import os
import json
import random
import argparse
import warnings
from pathlib import Path


import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import lightning as L
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelSummary

from neuralop.models import FNO
from neuralop.layers.embeddings import GridEmbeddingND
from neuralop.losses import LpLoss, H1Loss, ICLoss

from loguru import logger

torch.set_float32_matmul_precision('high')

# Disable UserWarning for deprecated features
warnings.filterwarnings("ignore", category=UserWarning)



# --------------------------
# 🔁 Aumento de datos 3D
# --------------------------
def augment_3d(input_tensor, output_tensor):
    for dims in [(1, 2), (1, 3), (2, 3)]:
        k = random.randint(0, 3)
        input_tensor = torch.rot90(input_tensor, k, dims=dims)
        output_tensor = torch.rot90(output_tensor, k, dims=[d - 1 for d in dims])
            
    # Randomly upscale/downscale the input tensor [0.125, 0.25, 0.5, 1, 2]
    scale_factor = random.choice([0.125, 0.25, 0.5, 1])
    
    if scale_factor == 1:
        return input_tensor, output_tensor
    input_tensor = F.interpolate(input_tensor.unsqueeze(0), scale_factor=scale_factor, mode='nearest').squeeze(0)
    output_tensor = F.interpolate(output_tensor.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode='nearest').squeeze(0).squeeze(0)
    
    return input_tensor, output_tensor


# --------------------------
# Dataset 3D personalizado
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
# ⚡ Modelo Lightning
# --------------------------
class LitFNO(L.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, **kwargs):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.loss = LpLoss(d=3)
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        self.log_dict({'train_loss': loss, 'train_mse': mse}, prog_bar=True, sync_dist=True)
        return mse + loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        self.log_dict({'val_loss': loss, 'val_mse': mse}, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.1)
        return [opt], [sched]
    
    def forward(self, x):
        return self.model(x)


# --------------------------
# 📸 Callback de visualización
# --------------------------
class PredictionPlotCallback(Callback):
    def __init__(self, output_dir, every_n_epochs=10):
        super().__init__()
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        pl_module.eval()
        if trainer.val_dataloaders is None:
            return
        val_loader = trainer.val_dataloaders if isinstance(trainer.val_dataloaders, torch.utils.data.DataLoader) else trainer.val_dataloaders[0]
        x, y = next(iter(val_loader))
        x = x.cuda()
        with torch.no_grad():
            y_hat = pl_module(x).cpu()
        y = y.cpu()
        x = x.cpu()

        mid_z = y.shape[2] // 2
        diff = (y_hat[0, 0, :, :, mid_z] - y[0, :, :, mid_z]).abs() / y[0, :, :, mid_z].abs()
        diff = diff / diff.max() # Escalar normalizado entre 0 y 1

        fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
        im0 = axs[0, 0].imshow(y[0, :, :, mid_z], cmap='viridis')
        axs[0, 0].set_title("Real Output")
        # im1 = axs[0, 1].imshow(y_hat[0, 0, :, :, mid_z], cmap='viridis', vmin=y.min(), vmax=y.max())
        im1 = axs[0, 1].imshow(y_hat[0, 0, :, :, mid_z], cmap='viridis')
        axs[0, 1].set_title("Predicted Output")
        im2 = axs[1, 0].imshow(diff, cmap='magma')
        axs[1, 0].set_title(r"Percentage Error ($|y_{pred} - y_{real}| / |y_{real}|$)")
        im3 = axs[1, 1].imshow(x[0, 3, :, :, mid_z], cmap='viridis')
        axs[1, 1].set_title("Input Field")
        fig.colorbar(im0, ax=axs[0, 0], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(im1, ax=axs[0, 1], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(im2, ax=axs[1, 0], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(im3, ax=axs[1, 1], orientation='vertical', fraction=0.02, pad=0.04)
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
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset (.npy file) to infer')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs')
    parser.add_argument('--forward_steps', type=int, default=1, help='Number of forward steps to take')
    args = parser.parse_args()

    # Crear directorios de salida
    os.makedirs(args.output_dir, exist_ok=True)

    # Cargar modelo
    model_config = json.load(open(os.path.join(args.checkpoint_path, 'config.json')))
    checkpoint = torch.load(os.path.join(args.checkpoint_path, 'new_arch.pt'), map_location='cuda', weights_only=False)
    embeddings = GridEmbeddingND(
        in_channels=4,
        dim=3,
        grid_boundaries=[[0,1], [0,1], [0,1]],
    )
    model = FNO(**model_config['model'], positional_embedding=embeddings)
    model.load_state_dict(checkpoint)
    lit_model = LitFNO(model, lr=0.001, weight_decay=0.0001, **model_config['model'])
    lit_model.model.eval().cuda()
    
    # Cargar datos
    data_raw = np.load(args.data_path)
    if not (data_raw.ndim >= 3 and data_raw.ndim <= 4):
        raise ValueError(f"Invalid data shape: {data_raw.shape}")
    if data_raw.ndim == 3:
        data_raw = np.expand_dims(data_raw, axis=0)
    data_raw = np.expand_dims(data_raw, axis=-1)
    data_raw = np.concatenate((data_raw[..., 0:1], data_raw[..., -1:]), axis=-1)
    print(f"Data shape after concatenation: {data_raw.shape}")
    assert len(set(data_raw.shape[1:-1])) == 1, "All dimensions except batch must match"
    data = torch.tensor(data_raw, dtype=torch.float32)
    data_set = Dataset3D(data, augment=False)
    eval_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=4)
    
    # Make predictions
    all_preds = []
    for batch in eval_loader:
        x = batch[0].cuda()
        with torch.no_grad():
            y_hat = lit_model(x)
        all_preds.append(y_hat[0].cpu())
    all_preds = torch.cat(all_preds, dim=0)

    # Save predictions
    output_path = os.path.join(args.output_dir, 'predictions.npy')
    np.save(output_path, all_preds.cpu().numpy())
    print(f"Predictions saved to {output_path}")

    # Plot predictions
    for i in range(all_preds.shape[0]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        im1 = ax1.imshow(data[i, 0, :, :, 0].numpy(), cmap='viridis')
        ax1.set_title("Input Field")
        plt.colorbar(im1, ax=ax1)
        data_plot = all_preds[i, 0, :, :].numpy()
        print(f"Shape to plot {data_plot.shape}")
        im2 = ax2.imshow(data_plot, cmap='viridis')
        ax2.set_title("Predicted Output")
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"prediction_{i}.png"))
        plt.close()
        
    # Forward steps
    all_preds_steps = []
    if args.forward_steps > 1:
        for batch in eval_loader:
            all_preds_steps = [batch[0].cpu()[0, 0, ...]]  # Inicializar con la primera entrada
            for step in range(args.forward_steps):
                if step == 0:
                    previous_step = batch[0].cuda()
                else:
                    item_infer = all_preds_steps[-1].unsqueeze(0)
                    item_infer = item_infer.unsqueeze(-1)
                    item_infer = torch.cat((item_infer, item_infer), dim=-1)
                    item_infer = next(iter(DataLoader(Dataset3D(item_infer, augment=False), batch_size=1)))
                    previous_step = item_infer[0]
                with torch.no_grad():
                    current_step = lit_model(previous_step.cuda())
                all_preds_steps.append(current_step[0, 0, ...].cpu())
    
    all_preds_steps = torch.stack(all_preds_steps, dim=0)
    print(all_preds_steps.shape)
    
    # Create a gif with [b, 64, 64, 64] but only use z=0
    all_preds_steps_z0 = all_preds_steps[:, ..., 32].numpy()

    gif_path = os.path.join(args.output_dir, "predictions_z0.gif")

    frames = []
    for i in range(all_preds_steps_z0.shape[0]):  # iterar sobre los pasos
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(all_preds_steps_z0[i], cmap="viridis")
        ax.set_title(f"Predicted Output (z=0), step={i}")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        # Guardar imagen temporal
        temp_path = os.path.join(args.output_dir, f"frame_{i}.png")
        plt.savefig(temp_path)
        plt.close(fig)

        # Leer imagen y añadir al gif
        frames.append(imageio.imread(temp_path))

    # Guardar GIF
    imageio.mimsave(gif_path, frames, duration=1)  # duración = 1s por frame
    print(f"GIF guardado en {gif_path}")