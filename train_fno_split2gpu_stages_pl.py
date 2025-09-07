import os
import argparse
import datetime
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary, Callback

from neuralop.models import FNO
from neuralop.losses import LpLoss
from neuralop.layers.embeddings import GridEmbeddingND

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision('high')


# --------------------------
# Data augmentation 3D
# --------------------------
def augment_3d(input_tensor, output_tensor):
    for dims in [(1, 2), (1, 3), (2, 3)]:
        k = random.randint(0, 3)
        input_tensor = torch.rot90(input_tensor, k, dims=dims)
        output_tensor = torch.rot90(output_tensor, k, dims=[d - 1 for d in dims])

    scale_factor = random.choice([0.125, 0.25, 0.5, 1])
    if scale_factor == 1:
        return input_tensor, output_tensor

    input_tensor = F.interpolate(input_tensor.unsqueeze(0), scale_factor=scale_factor, mode='nearest').squeeze(0)
    output_tensor = F.interpolate(
        output_tensor.unsqueeze(0).unsqueeze(0),
        scale_factor=scale_factor, mode='nearest'
    ).squeeze(0).squeeze(0)
    return input_tensor, output_tensor


# --------------------------
# Dataset 3D
# --------------------------
class Dataset3D(Dataset):
    def __init__(self, data, augment=False):
        self.input_raw = data[..., 0]
        self.output = data[..., 1]
        self.augment = augment

        self.data_size, self.size_x, self.size_y, self.size_z = self.input_raw.shape
        self.grid = self._generate_grid()

        self.input = self.input_raw.unsqueeze(-1)
        self.input = torch.cat((self.grid, self.input), dim=-1).permute(0, 4, 1, 2, 3)

    def _generate_grid(self):
        x = torch.linspace(0, 1, self.size_x)
        y = torch.linspace(0, 1, self.size_y)
        z = torch.linspace(0, 1, self.size_z)
        grid = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
        grid = grid.unsqueeze(0).repeat(self.data_size, 1, 1, 1, 1)
        return grid

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        x = self.input[idx]
        y = self.output[idx]
        if self.augment:
            x, y = augment_3d(x.clone(), y.clone())
        return x, y


# --------------------------
# Manual FNO stages (model parallel)
# --------------------------
class FNOStage0(nn.Module):
    def __init__(self, fno, split_k):
        super().__init__()
        fb = fno.fno_blocks
        self.pos_emb = fno.positional_embedding
        self.lifting = fno.lifting
        self.convs = nn.ModuleList([fb.convs[i] for i in range(split_k)])
        self.fno_skips = nn.ModuleList([fb.fno_skips[i] for i in range(split_k)]) if hasattr(fb, "fno_skips") else None
        self.channel_mlps = nn.ModuleList([fb.channel_mlp[i] for i in range(split_k)]) if hasattr(fb, "channel_mlp") else None
        self.channel_mlp_skips = nn.ModuleList([fb.channel_mlp_skips[i] for i in range(split_k)]) if hasattr(fb, "channel_mlp_skips") else None
        self.act = nn.GELU()

    def _block(self, x, i):
        y = self.convs[i](x)
        if self.fno_skips is not None:
            y = y + self.fno_skips[i](x)
        x = self.act(y)
        if self.channel_mlps is not None:
            y2 = self.channel_mlps[i](x)
            if self.channel_mlp_skips is not None:
                y2 = y2 + self.channel_mlp_skips[i](x)
            x = self.act(y2)
        return x

    def forward(self, x):
        if self.pos_emb is not None:
            x = self.pos_emb(x)
        x = self.lifting(x)
        for i in range(len(self.convs)):
            x = self._block(x, i)
        return x


class FNOStage1(nn.Module):
    def __init__(self, fno, split_k):
        super().__init__()
        fb = fno.fno_blocks
        n_all = len(fb.convs)
        self.convs = nn.ModuleList([fb.convs[i] for i in range(split_k, n_all)])
        self.fno_skips = nn.ModuleList([fb.fno_skips[i] for i in range(split_k, n_all)]) if hasattr(fb, "fno_skips") else None
        self.channel_mlps = nn.ModuleList([fb.channel_mlp[i] for i in range(split_k, n_all)]) if hasattr(fb, "channel_mlp") else None
        self.channel_mlp_skips = nn.ModuleList([fb.channel_mlp_skips[i] for i in range(split_k, n_all)]) if hasattr(fb, "channel_mlp_skips") else None
        self.projection = fno.projection
        self.act = nn.GELU()

    def _block(self, x, i):
        y = self.convs[i](x)
        if self.fno_skips is not None:
            y = y + self.fno_skips[i](x)
        x = self.act(y)
        if self.channel_mlps is not None:
            y2 = self.channel_mlps[i](x)
            if self.channel_mlp_skips is not None:
                y2 = y2 + self.channel_mlp_skips[i](x)
            x = self.act(y2)
        return x

    def forward(self, x):
        for i in range(len(self.convs)):
            x = self._block(x, i)
        x = self.projection(x)
        return x


# --------------------------
# LightningModule wrapping the two stages
# --------------------------
class LitFNO2Stage(L.LightningModule):
    def __init__(self, stage0, stage1, lr=1e-3, chunks=4, devices="0,1"):
        super().__init__()
        self.save_hyperparameters(ignore=["stage0", "stage1"])  # keep CLI params
        self.stage0 = stage0
        self.stage1 = stage1
        self.lr = lr
        self.chunks = int(chunks)
        self.lp = LpLoss(d=3)

        # parse devices string like "0,1"
        dev_ids = [int(x) for x in str(devices).split(',')]
        assert len(dev_ids) == 2, "devices must be like '0,1'"
        self.dev0_id, self.dev1_id = dev_ids

        # manual optimization to implement microbatching and single-step per batch
        self.automatic_optimization = False

    def on_fit_start(self):
        # after Lightning moves the module to root_device, place stages on requested GPUs
        dev0 = torch.device(f"cuda:{self.dev0_id}")
        dev1 = torch.device(f"cuda:{self.dev1_id}")
        self.stage0.to(dev0)
        self.stage1.to(dev1)

    def forward(self, x):
        # not used in training loop directly; keep for inference convenience
        h = self.stage0(x.to(f"cuda:{self.dev0_id}", non_blocking=True))
        y_hat = self.stage1(h.to(f"cuda:{self.dev1_id}", non_blocking=True))
        if y_hat.ndim == 5 and y_hat.size(1) == 1:
            y_hat = y_hat[:, 0]
        return y_hat

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        x, y = batch
        B = x.size(0)
        chunks = max(1, min(self.chunks, B))
        if chunks == 1:
            xs = [x]
            ys = [y]
        else:
            m = max(1, B // chunks)
            sizes = [m] * (chunks - 1) + [B - m * (chunks - 1)] if B % chunks != 0 else [m] * chunks
            xs = list(torch.split(x, sizes, dim=0))
            ys = list(torch.split(y, sizes, dim=0))

        dev0 = torch.device(f"cuda:{self.dev0_id}")
        dev1 = torch.device(f"cuda:{self.dev1_id}")

        total_mse = 0.0
        total_lp = 0.0
        total_combined = 0.0
        count = 0

        for xm, ym in zip(xs, ys):
            h = self.stage0(xm.to(dev0, non_blocking=True))
            y_hat = self.stage1(h.to(dev1, non_blocking=True))
            if y_hat.ndim == 5 and y_hat.size(1) == 1:
                y_hat = y_hat[:, 0]
            ym1 = ym.to(dev1, non_blocking=True)

            mse = F.mse_loss(y_hat, ym1)
            lpl = self.lp(y_hat, ym1)
            loss = mse + lpl

            # accumulate gradients over microbatches
            self.manual_backward(loss)

            total_mse += mse.detach().float().item()
            total_lp += lpl.detach().float().item()
            total_combined += loss.detach().float().item()
            count += 1

        opt.step()

        # log averaged metrics for this batch
        if count > 0:
            avg_mse = total_mse / count
            avg_lp = total_lp / count
            avg_total = total_combined / count
            # mirror naming from train_fno_vlasov.py and add combined
            self.log_dict(
                {
                    "train_loss": avg_lp,        # LpLoss
                    "train_mse": avg_mse,
                    "train_total": avg_total,    # mse + 5*Lp
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        dev0 = torch.device(f"cuda:{self.dev0_id}")
        dev1 = torch.device(f"cuda:{self.dev1_id}")
        with torch.no_grad():
            h = self.stage0(x.to(dev0, non_blocking=True))
            y_hat = self.stage1(h.to(dev1, non_blocking=True))
            if y_hat.ndim == 5 and y_hat.size(1) == 1:
                y_hat = y_hat[:, 0]
            y1 = y.to(dev1, non_blocking=True)
            mse = F.mse_loss(y_hat, y1)
            lpl = self.lp(y_hat, y1)

        # keep val_lp for checkpoint compatibility; also expose val_loss like the other script
        self.log("val_mse", mse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_lp", lpl, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss", lpl, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        params = list(self.stage0.parameters()) + list(self.stage1.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, betas=(0.9, 0.999))
        return opt


# --------------------------
# Prediction plot callback (similar to train_fno_vlasov.py)
# --------------------------
class PredictionPlotCallback(Callback):
    def __init__(self, output_dir: str, every_n_epochs: int = 10):
        super().__init__()
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Plot every N epochs
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        pl_module.eval()

        # Get first batch from validation loader
        if trainer.val_dataloaders is None:
            return
        val_loader = (
            trainer.val_dataloaders
            if isinstance(trainer.val_dataloaders, torch.utils.data.DataLoader)
            else trainer.val_dataloaders[0]
        )

        try:
            x, y = next(iter(val_loader))
        except StopIteration:
            return

        # Forward through the model-parallel network; forward handles device moves
        with torch.no_grad():
            y_hat = pl_module(x)

        # Move to CPU for plotting
        y_hat = y_hat.detach().cpu()
        y = y.detach().cpu()
        x = x.detach().cpu()

        # Support either (B, 1, X, Y, Z) or (B, X, Y, Z)
        if y_hat.ndim == 5 and y_hat.size(1) == 1:
            y_pred_mid = y_hat[0, 0]
        else:
            y_pred_mid = y_hat[0]

        mid_z = y_pred_mid.shape[-1] // 2
        y_pred_slice = y_pred_mid[:, :, mid_z]
        y_true_slice = y[0, :, :, mid_z]

        # Relative error (normalized 0-1 for visualization)
        diff = (y_pred_slice - y_true_slice).abs() / (y_true_slice.abs() + 1e-7)
        if torch.isfinite(diff).any():
            maxv = diff.max()
            if maxv > 0:
                diff = diff / maxv

        fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
        im0 = axs[0, 0].imshow(y_true_slice, cmap='viridis')
        axs[0, 0].set_title("Real Output")
        im1 = axs[0, 1].imshow(y_pred_slice, cmap='viridis')
        axs[0, 1].set_title("Predicted Output")
        im2 = axs[1, 0].imshow(diff, cmap='magma', vmin=0, vmax=1)
        axs[1, 0].set_title(r"Relative Error ($|y_{pred}-y|/|y|$)")
        # Input field channel (last input channel is the physical field)
        try:
            im3 = axs[1, 1].imshow(x[0, 3, :, :, mid_z], cmap='viridis')
        except Exception:
            im3 = axs[1, 1].imshow(x[0, -1, :, :, mid_z], cmap='viridis')
        axs[1, 1].set_title("Input Field")

        fig.colorbar(im0, ax=axs[0, 0], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(im1, ax=axs[0, 1], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(im2, ax=axs[1, 0], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(im3, ax=axs[1, 1], orientation='vertical', fraction=0.02, pad=0.04)

        plt.suptitle(f"Epoch {trainer.current_epoch}")
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, f"epoch_{trainer.current_epoch:04d}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved prediction plot: {out_path}", end='\r', flush=True)
        
# Data builders
# --------------------------
def build_data(data_path, batch_size):
    data = torch.tensor(np.load(data_path), dtype=torch.float32)
    data = F.interpolate(
        data.permute(0, 4, 1, 2, 3),
        size=(128, 128, 128),
        mode='trilinear',
        align_corners=True,
    ).permute(0, 2, 3, 4, 1)

    data_train_raw, data_eval_raw = data[:14], data[14:]
    dataset_train = ConcatDataset([Dataset3D(data_train_raw), Dataset3D(data_train_raw, augment=True)])
    dataset_eval = Dataset3D(data_eval_raw)

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )
    return train_loader, val_loader


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_name', type=str, required=True)
    ap.add_argument('--data_path', type=str, default='/home/ia/asantillan/Proyects/VlasovPoisson/vlasov_dataset.npy')
    ap.add_argument('--output_path', type=str, default='experiments/VlasovPoisson')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--chunks', type=int, default=4, help='microbatches per batch')
    ap.add_argument('--devices', type=str, default='0,1', help="GPU IDs like '0,1'")
    ap.add_argument('--split_k', type=int, default=-1, help='num FNO blocks in stage0 (auto=half)')
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(args.output_path, f"{ts}_{args.exp_name}")
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # data
    train_loader, val_loader = build_data(args.data_path, args.batch_size)

    # base FNO to extract submodules
    embeddings = GridEmbeddingND(
        in_channels=4, dim=3, grid_boundaries=[[0, 1], [0, 1], [0, 1]],
    )
    fno = FNO(
        n_modes=[64, 64, 64],  # type: ignore
        hidden_channels=64,
        n_layers=4,
        in_channels=4,
        lifting_channel_ratio=4,
        projection_channel_ratio=4,
        out_channels=1,
        factorization="tucker",
        fno_block_precision="full",
        stabilizer="tanh",
        rank=0.1,
        channel_mlp_dropout=0.1,
        preactivation=True,
        positional_embedding=embeddings,
    )

    # split
    n_blocks = len(fno.fno_blocks.convs)
    split_k = (n_blocks // 2) if args.split_k == -1 else max(1, min(args.split_k, n_blocks - 1))
    print(f"Split FNO: total_blocks={n_blocks} -> stage0={split_k}, stage1={n_blocks - split_k}")

    stage0 = FNOStage0(fno, split_k)
    stage1 = FNOStage1(fno, split_k)

    # lightning module
    lit = LitFNO2Stage(stage0, stage1, lr=args.lr, chunks=args.chunks, devices=args.devices)

    # callbacks & loggers
    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir,
        filename='best',
        monitor='val_lp',
        mode='min',
        save_top_k=1,
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval='epoch')
    plot_cb = PredictionPlotCallback(img_dir, every_n_epochs=10)
    csv_logger = CSVLogger(save_dir=out_dir, name='csv_logs')
    tb_logger = TensorBoardLogger(save_dir=out_dir, name='tb_logs')

    # trainer: single process uses both GPUs internally via model-parallel stages
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=1,  # run a single process; model uses 2 GPUs internally
        logger=[csv_logger, tb_logger],
        callbacks=[ckpt_cb, lr_cb, plot_cb, ModelSummary(max_depth=4)],
        gradient_clip_val=None,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(lit, train_loader, val_loader)

    # also export a compact .pt with just stage states for convenience
    torch.save(
        {
            'stage0': stage0.state_dict(),
            'stage1': stage1.state_dict(),
            'meta': {
                'best_val_lp': trainer.callback_metrics.get('val_lp', torch.tensor(float('nan'))).item()
            }
        },
        os.path.join(out_dir, 'best_split2gpu.pt')
    )
    print(f"Saved weights to {out_dir}")


if __name__ == "__main__":
    main()
