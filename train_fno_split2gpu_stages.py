import os
import argparse
import datetime
import warnings
import random
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from neuralop.models import FNO
from neuralop.layers.embeddings import GridEmbeddingND
from neuralop.losses import LpLoss

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision('high')

# --------------------------
# 🔁 Aumento de datos 3D
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
# 🧱 Etapas manuales del FNO
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
        # x: (B, C, X, Y, Z)
        if self.pos_emb is not None:
            x = self.pos_emb(x)
        x = self.lifting(x)
        for i in range(len(self.convs)):
            x = self._block(x, i)
        return x  # hidden features

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
        x = self.projection(x)  # (B, out_channels, X, Y, Z)
        return x

# --------------------------
# 🔧 Entrenamiento 2-GPU con microbatches
# --------------------------
def train_loop_2gpu(stage0, stage1, train_loader, val_loader, epochs, lr, out_dir, chunks):
    params = list(stage0.parameters()) + list(stage1.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    lp = LpLoss(d=3)

    dev0 = next(stage0.parameters()).device
    dev1 = next(stage1.parameters()).device

    best_val = float('inf')

    for ep in range(1, epochs + 1):
        stage0.train(); stage1.train()
        tr_lp = tr_mse = 0.0; nb = 0

        for x, y in train_loader:
            B = x.size(0)
            m = max(1, B // chunks)
            sizes = [m]*(chunks-1) + [B - m*(chunks-1)] if B % chunks != 0 else [m]*chunks

            xs = list(torch.split(x, sizes, dim=0))
            ys = list(torch.split(y, sizes, dim=0))

            opt.zero_grad(set_to_none=True)

            for xm, ym in zip(xs, ys):
                h = stage0(xm.to(dev0, non_blocking=True))
                h = h.to(dev1, non_blocking=True)
                y_hat = stage1(h)
                if y_hat.ndim == 5 and y_hat.size(1) == 1:
                    y_hat = y_hat[:, 0]
                ym1 = ym.to(dev1, non_blocking=True)

                mse = F.mse_loss(y_hat, ym1)
                lpl = lp(y_hat, ym1)
                loss = mse + 5*lpl
                loss.backward()

                tr_lp += lpl.item(); tr_mse += mse.item(); nb += 1

            opt.step()

        # ----- Validación -----
        stage0.eval(); stage1.eval()
        with torch.no_grad():
            vl_lp = vl_mse = 0.0; vb = 0
            for x, y in val_loader:
                h = stage0(x.to(dev0, non_blocking=True))
                h = h.to(dev1, non_blocking=True)
                y_hat = stage1(h)
                if y_hat.ndim == 5 and y_hat.size(1) == 1:
                    y_hat = y_hat[:, 0]
                y1 = y.to(dev1, non_blocking=True)
                mse = F.mse_loss(y_hat, y1)
                lpl = lp(y_hat, y1)
                vl_lp += lpl.item(); vl_mse += mse.item(); vb += 1

        tr_lp /= max(nb, 1); tr_mse /= max(nb, 1)
        vl_lp /= max(vb, 1); vl_mse /= max(vb, 1)
        print(f"🗓️ Epoch {ep:03d} | train_lp={tr_lp:.4e} train_mse={tr_mse:.4e} | val_lp={vl_lp:.4e} val_mse={vl_mse:.4e}")

        if vl_lp < best_val:
            best_val = vl_lp
            os.makedirs(out_dir, exist_ok=True)
            ck = os.path.join(out_dir, "best_split2gpu.pt")
            torch.save({
                'stage0': stage0.state_dict(),
                'stage1': stage1.state_dict(),
                'meta': {'best_val_lp': best_val, 'epoch': ep}
            }, ck)
            print(f"✅ Nuevo mejor ({best_val:.4e}). Guardado en: {ck}")

# --------------------------
# 🔢 Datos
# --------------------------
def build_data(data_path, batch_size):
    data = torch.tensor(np.load(data_path), dtype=torch.float32)
    data = F.interpolate(data.permute(0, 4, 1, 2, 3), size=(128, 128, 128),
                         mode='trilinear', align_corners=True)
    data = data.permute(0, 2, 3, 4, 1)  # (N, X, Y, Z, C)

    data_train_raw, data_eval_raw = data[:14], data[14:]
    dataset_train = ConcatDataset([Dataset3D(data_train_raw), Dataset3D(data_train_raw, augment=True)])
    dataset_eval = Dataset3D(data_eval_raw)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                              num_workers=8, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False,
                            num_workers=8, persistent_workers=True, pin_memory=True)
    return train_loader, val_loader

# --------------------------
# 🚀 Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_name', type=str, required=True)
    ap.add_argument('--data_path', type=str, default='/home/ia/asantillan/Proyects/VlasovPoisson/vlasov_dataset.npy')
    ap.add_argument('--output_path', type=str, default='experiments/VlasovPoisson')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--chunks', type=int, default=4, help="microbatches por batch")
    ap.add_argument('--devices', type=str, default='0,1', help="IDs de GPU: '0,1'")
    ap.add_argument('--split_k', type=int, default=-1, help="nº de bloques FNO en la etapa0 (auto=mitad)")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(args.output_path, f"{ts}_{args.exp_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Datos
    train_loader, val_loader = build_data(args.data_path, args.batch_size)

    # Modelo base (solo para extraer submódulos)
    embeddings = GridEmbeddingND(
        in_channels=4, dim=3, grid_boundaries=[[0,1], [0,1], [0,1]],
    )
    fno = FNO(
        n_modes=[128, 128, 128], # type: ignore
        hidden_channels=32,
        n_layers=4,
        in_channels=4,
        out_channels=1,
        factorization="tucker",
        fno_block_precision="mixed",
        stabilizer="tanh",
        rank=0.1,
        positional_embedding=embeddings
    )

    # Decidir split
    n_blocks = len(fno.fno_blocks.convs)
    split_k = (n_blocks // 2) if args.split_k == -1 else max(1, min(args.split_k, n_blocks-1))
    print(f"🔪 Partiendo FNO: total_blocks={n_blocks} -> stage0={split_k}, stage1={n_blocks - split_k}")

    # Construir etapas
    stage0 = FNOStage0(fno, split_k)
    stage1 = FNOStage1(fno, split_k)

    dev_ids = [int(x) for x in args.devices.split(',')]
    assert len(dev_ids) == 2, "Este script divide en exactamente 2 GPUs; usa --devices 0,1 por ejemplo."

    dev0 = torch.device(f'cuda:{dev_ids[0]}')
    dev1 = torch.device(f'cuda:{dev_ids[1]}')
    stage0.to(dev0)
    stage1.to(dev1)

    # Entrenar
    train_loop_2gpu(stage0, stage1, train_loader, val_loader,
                    epochs=args.epochs, lr=args.lr, out_dir=out_dir, chunks=args.chunks)

if __name__ == "__main__":
    main()
