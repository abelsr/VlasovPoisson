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
        return mse + 5*loss
    
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
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--n_nodes', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--data_path', type=str, default='/home/ia/asantillan/Proyects/VlasovPoisson/vlasov_dataset.npy')
    parser.add_argument('--output_path', type=str, default='experiments/VlasovPoisson')
    args = parser.parse_args()

    # 🗂️ Subdirectorio del experimento
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_dir = os.path.join(args.output_path, f"{timestamp}_{args.exp_name}")
    os.makedirs(experiment_dir, exist_ok=True)
    img_dir = os.path.join(experiment_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"📁 Carpeta del experimento: {experiment_dir}")

    # 🧠 Guarda configuración
    config = {
        "experiment": args.exp_name,
        "timestamp": timestamp,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "hardware": {
            "n_gpus": args.n_gpus,
            "n_nodes": args.n_nodes
        },
        "model": {
            "n_modes": [64, 64, 64],
            "hidden_channels": 64,
            "n_layers": 3,
            "in_channels": 4,
            "out_channels": 1,
            "factorization": "tucker",
            "fno_block_precision": "mixed",
            "stabilizer": "tanh",
            "rank": 0.1
        }
    }
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print("📝 Configuración guardada en config.json")
    

    # 🔢 Cargar datos
    data = torch.tensor(np.load(args.data_path), dtype=torch.float32)
    # Normalización datos
    # data = (data - data.min()) / (data.max() - data.min())
    data = F.interpolate(data.permute(0, 4, 1, 2, 3), size=(128, 128, 128), mode='trilinear', align_corners=True)
    data = data.permute(0, 2, 3, 4, 1)  # Volver a la forma original (N, X, Y, Z, C)
    data_train_raw, data_eval_raw = data[:14], data[14:]
    dataset_train = ConcatDataset([Dataset3D(data_train_raw), Dataset3D(data_train_raw, augment=True)])
    dataset_eval = Dataset3D(data_eval_raw)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8,
                              persistent_workers=True, pin_memory=True)
    eval_loader = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=8,
                             persistent_workers=True, pin_memory=True)
    
    embeddings = GridEmbeddingND(
        in_channels=4,
        dim=3,
        grid_boundaries=[[0,1], [0,1], [0,1]],
    )
    model = FNO(**config['model'], positional_embedding=embeddings)
    lit_model = LitFNO(model, lr=args.lr, weight_decay=args.weight_decay, **config['model'])

    # 📈 Loggers
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
        strategy='ddp',
        num_nodes=args.n_nodes,
        callbacks=[plot_callback, lr_monitor, ModelSummary(max_depth=3)],
        accumulate_grad_batches=2,
        # precision='16'
    )

    # 🚀 Entrena
    t0 = time.time()
    trainer.fit(lit_model, train_loader, eval_loader)
    tf = time.time()
    print(f"⏱️ Entrenamiento finalizado en {pd.to_datetime(tf - t0, unit='s').strftime('%H:%M:%S')}")

    # 💾 Guarda pesos del modelo
    torch.save(model.state_dict(), os.path.join(experiment_dir, f"{args.exp_name}.pt"))
    print("✅ Pesos del modelo guardados.")

    # 🔮 Predicción del último sample sin modificación
    lit_model.model.eval().cuda()
    x_last, y_last = dataset_eval[-1]
    x_last = x_last.unsqueeze(0).cuda()
    with torch.no_grad():
        y_hat_last = lit_model.model(x_last).cpu().squeeze(0).numpy()
    pred_path = os.path.join(experiment_dir, f"{args.exp_name}_prediction.npy")
    np.save(pred_path, y_hat_last)
    print(f"📦 Predicción guardada en {pred_path}")
    
    # Predecimos todo el dataset de train y eval
    # data = torch.tensor(np.load(args.data_path), dtype=torch.float32)
    # data_train_raw, data_eval_raw = data[:14], data[14:]
    dataset_train = Dataset3D(data_train_raw)
    dataset_eval = Dataset3D(data_eval_raw)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=8,
                              persistent_workers=True, pin_memory=True)
    eval_loader = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=8,
                             persistent_workers=True, pin_memory=True)
    lit_model.model.eval().cuda()
    all_predictions = []
    for batch in train_loader:
        x, _ = batch
        x = x.cuda()
        with torch.no_grad():
            y_hat = lit_model.model(x).cpu().numpy()
        # Checar dimensiones de y_hat
        print(f"Dimensiones de y_hat (train): {y_hat.shape}")
        all_predictions.append(y_hat)
        
    # breakpoint()
    for batch in eval_loader:
        x, _ = batch
        x = x.cuda()
        with torch.no_grad():
            y_hat = lit_model.model(x).cpu().numpy()
        # Checar dimensiones de y_hat
        print(f"Dimensiones de y_hat (eval): {y_hat.shape}")
        all_predictions.append(y_hat)
    all_predictions = np.concatenate(all_predictions, axis=0)
    print(f"Total de predicciones: {len(all_predictions)}. Con shape {all_predictions.shape}") # (17, 1, 128, 128, 128)
    all_pred_path = os.path.join(experiment_dir, f"{args.exp_name}_all_predictions.npy")
    np.save(all_pred_path, all_predictions)
    print(f"📦 Todas las predicciones guardadas en {all_pred_path}")
    
    # Plot de las predicciones
    for i, pred in enumerate(all_predictions):
        mid_z = pred.shape[2] // 2
        y_pred = pred[0, :, :, mid_z]
        y_true = data[i, :, :, mid_z, 1] + 1e-7 # To avoid zero-division
        y_true = y_true.numpy()
        diff = np.abs(y_pred - y_true) / np.abs(y_true)
        diff = diff / diff.max() # Escalar normalizado entre 0 y 1
        fig, ax = plt.subplots(1, 3, figsize=(10, 8), dpi=300)
        ax[0].imshow(y_true, cmap='viridis')
        ax[0].set_title("Real Output")
        ax[1].imshow(y_pred, cmap='viridis')
        ax[1].set_title("Predicted Output")
        ax[2].imshow(diff, cmap='magma')
        ax[2].set_title("Relative Error")
        # Color bar for error from 0 to 1
        plt.colorbar(
            ax[2].imshow(diff, cmap='magma', vmin=0, vmax=1), ax=ax[2], 
            orientation='vertical', 
            fraction=0.02, 
            pad=0.04
        )
        plt.suptitle(f"Prediction {i+1} - Epoch {trainer.current_epoch}")
        plt.tight_layout()
        
        # Guardar imagen
        os.makedirs(os.path.join(experiment_dir, "final_predictions"), exist_ok=True)
        plt.savefig(os.path.join(experiment_dir, "final_predictions", f"prediction_{i+1:04d}.png"))
        plt.close()