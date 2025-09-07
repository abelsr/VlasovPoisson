import os
import time
import json
import random
import argparse
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelSummary

from neuralop.models import FNO, UNO
from neuralop.layers.embeddings import GridEmbeddingND

from loguru import logger

torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------
# 🔁 Augment 3D seguro (sin cambiar resolución)
# --------------------------
def augment_no_rescale(x, y):
    # x: (C,X,Y,Z), y: (X,Y,Z)
    # flips aleatorios
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(-3,))
        y = torch.flip(y, dims=(-3,))
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(-2,))
        y = torch.flip(y, dims=(-2,))
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(-1,))
        y = torch.flip(y, dims=(-1,))

    # rotaciones 90° en planos
    for dims_x, dims_y in [((-3, -2), (-2, -1)), ((-3, -1), (-3, -1)), ((-2, -1), (-2, -1))]:
        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, dims=dims_x)
        y = torch.rot90(y, k, dims=dims_y)

    # shifts periódicos (roll)
    sx = torch.randint(0, x.shape[-3], (1,)).item()
    sy = torch.randint(0, x.shape[-2], (1,)).item()
    sz = torch.randint(0, x.shape[-1], (1,)).item()
    x = x.roll((sx, sy, sz), dims=(-3, -2, -1))
    y = y.roll((sx, sy, sz), dims=(-3, -2, -1))
    return x, y


# --------------------------
# Dataset 3D (sin duplicar grid posicional)
# --------------------------
class Dataset3D(Dataset):
    def __init__(self, data, augment=False):
        # data: (N,X,Y,Z,C) con C = [input, output, z0, z1, ...]
        self.input_raw = data[..., 0]   # (N,X,Y,Z)
        self.output = data[..., 1]      # (N,X,Y,Z)
        self.z0s = data[..., 2]         # (N,X,Y,Z) casi constante por muestra
        self.z1s = data[..., 3]         # (N,X,Y,Z) casi constante por muestra
        self.augment = augment

        self.N, self.X, self.Y, self.Z = self.input_raw.shape
        # canaliza el campo de entrada
        self.input = self.input_raw.unsqueeze(1)  # (N,1,X,Y,Z)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.input[idx]       # (1,X,Y,Z)
        y = self.output[idx]      # (X,Y,Z)

        if self.augment:
            x, y = augment_no_rescale(x.clone(), y.clone())

        # valores z0/z1 estables (evita unique() en float)
        z0 = self.z0s[idx][0, 0, 0].item()
        z1 = self.z1s[idx][0, 0, 0].item()
        z0c = torch.full_like(x[:1], fill_value=z0)  # (1,X,Y,Z)
        z1c = torch.full_like(x[:1], fill_value=z1)  # (1,X,Y,Z)

        # x final: [campo, z0, z1]
        x = torch.cat([x, z0c, z1c], dim=0)          # (3,X,Y,Z)
        return x, y


# --------------------------
# ⚡ Modelo Lightning
# --------------------------
class LitFNO(L.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, **kwargs):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # MSE base (mean)
        mse = F.mse_loss(y_hat, y)

        # Focal + relativa (para regiones pequeñas)
        eps, tau, gamma = 1e-4, 1e-2, 2.0
        err = y_hat - y
        w = (tau / (y.abs() + tau)).pow(gamma)
        loss_rel = ((err / (y.abs() + eps)) ** 2).mean()
        loss_focal = (w * err.pow(2)).sum() / (w.sum() + 1e-8)
        focused = 0.5 * loss_rel + 0.5 * loss_focal

        # Pesos (balance homogéneo)
        λ_mse, λ_foc = 1.0, 0.1
        loss = λ_mse * mse + λ_foc * focused

        # Logs por ÉPOCA (para Plateau) + contribuciones
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mse", mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/focused", focused, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/contr_mse", λ_mse * mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/contr_focused", λ_foc * focused, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        mse = F.mse_loss(y_hat, y)
        eps, tau, gamma = 1e-4, 1e-2, 2.0
        err = y_hat - y
        w = (tau / (y.abs() + tau)).pow(gamma)
        loss_rel = ((err / (y.abs() + eps)) ** 2).mean()
        loss_focal = (w * err.pow(2)).sum() / (w.sum() + 1e-8)
        focused = 0.5 * loss_rel + 0.5 * loss_focal

        λ_mse, λ_foc = 1.0, 0.1
        loss = λ_mse * mse + λ_foc * focused

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mse", mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/focused", focused, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Warmup 5% de épocas + Cosine (por ÉPOCA)
        max_epochs = int(getattr(self.trainer, "max_epochs", 200))
        warmup_epochs = max(1, int(0.05 * max_epochs))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6
        )
        seq = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )

        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

        # ✅ Formato recomendado: ([optimizers], [lr_schedulers])
        schedulers = [
            {"scheduler": seq, "interval": "epoch", "frequency": 1},
            {"scheduler": plateau, "monitor": "val/loss", "interval": "epoch", "frequency": 1},
        ]
        return [optimizer], schedulers


    def forward(self, x):
        return self.model(x)


# --------------------------
# 📸 Callback de visualización (DDP-safe)
# --------------------------
class PredictionPlotCallback(Callback):
    def __init__(self, output_dir, every_n_epochs=10):
        super().__init__()
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if trainer.val_dataloaders is None:
            return

        val_loader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, (list, tuple)) else trainer.val_dataloaders
        x, y = next(iter(val_loader))
        x = x.to(pl_module.device, non_blocking=True)
        y = y.to(pl_module.device, non_blocking=True)

        with torch.no_grad():
            y_hat = pl_module(x)

        # preparar para imshow en CPU
        x = x.detach().cpu()
        y = y.detach().cpu()
        y_hat = y_hat.detach().cpu()

        mid_z = y.shape[-1] // 2
        eps = 1e-6
        den = y[0, :, :, mid_z].abs().clamp_min(eps)
        diff = (y_hat[0, 0, :, :, mid_z] - y[0, :, :, mid_z]).abs() / den
        diff = diff / (diff.max().clamp_min(eps))

        fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
        im0 = axs[0, 0].imshow(y[0, :, :, mid_z], cmap='viridis')
        axs[0, 0].set_title("Real Output")
        im1 = axs[0, 1].imshow(y_hat[0, 0, :, :, mid_z], cmap='viridis')
        axs[0, 1].set_title("Predicted Output")
        im2 = axs[1, 0].imshow(diff, cmap='magma', vmin=0, vmax=1)
        axs[1, 0].set_title(r"Relative Error ($|y_{\mathrm{pred}}-y|/|y|$)")
        im3 = axs[1, 1].imshow(x[0, 0, :, :, mid_z], cmap='viridis')
        axs[1, 1].set_title("Input Field (channel 0)")

        for im, ax in zip([im0, im1, im2, im3], axs.ravel()):
            fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)

        plt.suptitle(f"Epoch {trainer.current_epoch}")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"epoch_{trainer.current_epoch:04d}.png")
        plt.savefig(path)
        plt.close()
        print(f"📸 Imagen guardada: {path}", end='\r', flush=True)


# --------------------------
# 🚀 Entrenamiento principal
# --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--n_nodes', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--data_path', type=str, default='/home/ia/asantillan/Proyects/VlasovPoisson/vlasov_dataset_with_z_values.npy')
    parser.add_argument('--output_path', type=str, default='experiments/VlasovPoisson')
    args = parser.parse_args()

    # 🗂️ Subdirectorio del experimento
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_dir = os.path.join(args.output_path, f"{timestamp}_{args.exp_name}")
    os.makedirs(experiment_dir, exist_ok=True)
    img_dir = os.path.join(experiment_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"📁 Carpeta del experimento: {experiment_dir}")

    # 🧠 Config
    config = {
        "experiment": args.exp_name,
        "timestamp": timestamp,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "hardware": {"n_gpus": args.n_gpus, "n_nodes": args.n_nodes},
        "model": {
            "n_modes": [64, 64, 64],
            "hidden_channels": 64,
            "n_layers": 4,
            "in_channels": 3,           # <- campo + z0 + z1
            "out_channels": 1,
            "factorization": "tucker",
            "fno_block_precision": "full",
            "stabilizer": "tanh",
            "rank": 0.01,
            "channel_mlp_dropout": 0.05,
            "channel_mlp_expansion": 1,
            "lifting_channel_ratio": 3,
            "projection_channel_ratio": 3,
        }
    }
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print("📝 Configuración guardada en config.json")

    # 🔢 Cargar datos
    data = torch.tensor(np.load(args.data_path), dtype=torch.float32)  # (N,X,Y,Z,C)

    # Normalización estable (log1p) para input/output si procede
    # (si tus campos pueden ser negativos < -1, ajusta/omite esta parte)
    data[..., 0] = torch.log1p(data[..., 0].clamp_min(-0.999))
    data[..., 1] = torch.log1p(data[..., 1].clamp_min(-0.999))

    # Asegura resolución 128^3 si quieres unificar
    data = F.interpolate(data.permute(0, 4, 1, 2, 3), size=(128, 128, 128),
                         mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)

    # Split aleatorio (mantiene ~14/3 si N>=17)
    N = data.shape[0]
    if N >= 17:
        n_train = 14
    else:
        n_train = max(1, int(0.8 * N))
    idx = torch.randperm(N)
    train_idx, eval_idx = idx[:n_train], idx[n_train:]
    data_train_raw, data_eval_raw = data[train_idx], data[eval_idx]

    # Dataset y DataLoaders (duplicamos dataset con augment para ampliar iteraciones)
    dataset_train = ConcatDataset([
        Dataset3D(data_train_raw),
        Dataset3D(data_train_raw, augment=True),
        Dataset3D(data_train_raw, augment=True),
        Dataset3D(data_train_raw, augment=True),
    ])
    dataset_eval = Dataset3D(data_eval_raw)

    num_workers = min(4, (os.cpu_count() or 2))
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        dataset_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
    )

    # Positional embedding (sin inyectar grid como canales)
    embeddings = GridEmbeddingND(
        in_channels=config['model']['in_channels'],
        dim=3,
        grid_boundaries=[[0, 1], [0, 1], [0, 1]],
    )
    model = FNO(**config['model'], positional_embedding=embeddings)

    lit_model = LitFNO(model, lr=args.lr, weight_decay=args.weight_decay, **config['model'])

    # 📈 Loggers y callbacks
    logger_csv = CSVLogger(save_dir=experiment_dir, name="csv_logs")
    logger_tb = TensorBoardLogger(save_dir=experiment_dir, name="tb_logs")
    plot_callback = PredictionPlotCallback(img_dir, every_n_epochs=10)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 🏋️ Entrenador
    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=[logger_csv, logger_tb],
        accelerator='gpu',
        devices=args.n_gpus,
        strategy='ddp' if args.n_gpus > 1 or args.n_nodes > 1 else 'auto',
        num_nodes=args.n_nodes,
        callbacks=[plot_callback, lr_monitor, ModelSummary(max_depth=3)],
        accumulate_grad_batches=2,
        enable_model_summary=False,
        # precision='16',
    )

    # 🚀 Entrena
    t0 = time.time()
    trainer.fit(lit_model, train_loader, eval_loader)
    tf = time.time()
    print(f"⏱️ Entrenamiento finalizado en {pd.to_datetime(tf - t0, unit='s').strftime('%H:%M:%S')}")

    # 💾 Guarda pesos del modelo
    torch.save(model.state_dict(), os.path.join(experiment_dir, f"{args.exp_name}.pt"))
    print("✅ Pesos del modelo guardados.")

    # 🔮 Predicción del último sample
    lit_model.eval().to(trainer.strategy.root_device)
    if len(dataset_eval) > 0:
        x_last, y_last = dataset_eval[-1]
        x_last = x_last.unsqueeze(0).to(trainer.strategy.root_device)
        with torch.no_grad():
            y_hat_last = lit_model.model(x_last).cpu().squeeze(0).numpy()
        pred_path = os.path.join(experiment_dir, f"{args.exp_name}_prediction.npy")
        np.save(pred_path, y_hat_last)
        print(f"📦 Predicción guardada en {pred_path}")

    # 🔮 Predicciones para todo el dataset (train + eval)
    dataset_train_plain = Dataset3D(data_train_raw)
    dataset_eval_plain = Dataset3D(data_eval_raw)

    train_loader_plain = DataLoader(dataset_train_plain, batch_size=args.batch_size, shuffle=False,
                                    num_workers=num_workers, persistent_workers=False, pin_memory=True)
    eval_loader_plain = DataLoader(dataset_eval_plain, batch_size=args.batch_size, shuffle=False,
                                   num_workers=num_workers, persistent_workers=False, pin_memory=True)

    lit_model.eval().to(trainer.strategy.root_device)
    all_predictions = []
    for loader, tag in [(train_loader_plain, "train"), (eval_loader_plain, "eval")]:
        for batch in loader:
            x, _ = batch
            x = x.to(trainer.strategy.root_device)
            with torch.no_grad():
                y_hat = lit_model.model(x).cpu().numpy()
            print(f"Dimensiones de y_hat ({tag}): {y_hat.shape}")
            all_predictions.append(y_hat)

    if all_predictions:
        all_predictions = np.concatenate(all_predictions, axis=0)  # (N,1,128,128,128)
        print(f"Total de predicciones: {len(all_predictions)}. Con shape {all_predictions.shape}")
        all_pred_path = os.path.join(experiment_dir, f"{args.exp_name}_all_predictions.npy")
        np.save(all_pred_path, all_predictions)
        print(f"📦 Todas las predicciones guardadas en {all_pred_path}")

        # Plot saneado (eps en denominador)
        eps = 1e-6
        os.makedirs(os.path.join(experiment_dir, "final_predictions"), exist_ok=True)
        data_cpu = data.cpu().numpy()
        for i, pred in enumerate(all_predictions):
            mid_z = pred.shape[2] // 2
            y_pred = pred[0, :, :, mid_z]
            y_true = data_cpu[i, :, :, mid_z, 1]  # ya normalizado como el modelo
            den = np.maximum(np.abs(y_true), eps)
            diff = np.abs(y_pred - y_true) / den
            diff = diff / max(diff.max(), eps)

            fig, ax = plt.subplots(1, 3, figsize=(10, 8), dpi=300)
            ax[0].imshow(y_true, cmap='viridis')
            ax[0].set_title("Real Output")
            ax[1].imshow(y_pred, cmap='viridis')
            ax[1].set_title("Predicted Output")
            im = ax[2].imshow(diff, cmap='magma', vmin=0, vmax=1)
            ax[2].set_title("Relative Error")
            plt.colorbar(im, ax=ax[2], orientation='vertical', fraction=0.02, pad=0.04)
            plt.suptitle(f"Prediction {i+1} - Epoch {trainer.current_epoch}")
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_dir, "final_predictions", f"prediction_{i+1:04d}.png"))
            plt.close()
