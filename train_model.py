import os
import time
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from neuralop.models import FNO
from neuralop.losses import LpLoss, H1Loss, ICLoss

torch.set_float32_matmul_precision('medium')


# ------------------------------------------
# 🔁 Augmentation 3D (rotaciones y flips)
# ------------------------------------------
def augment_3d(input_tensor, output_tensor):
    for dims in [(1, 2), (1, 3), (2, 3)]:
        k = random.randint(0, 3)
        input_tensor = torch.rot90(input_tensor, k, dims=dims)
        output_tensor = torch.rot90(output_tensor, k, dims=[d - 1 for d in dims])

    for dim in [1, 2, 3]:
        if random.random() > 0.5:
            input_tensor = torch.flip(input_tensor, dims=[dim])
            output_tensor = torch.flip(output_tensor, dims=[dim - 1])
    return input_tensor, output_tensor


# ------------------------------------------
# 📦 Dataset 3D
# ------------------------------------------
class Dataset3D(Dataset):
    def __init__(self, data, augment=False, add_noise=False, noise_std=0.01):
        self.raw_input = data[..., 0]
        self.output = data[..., 1]
        self.augment = augment
        self.add_noise = add_noise
        self.noise_std = noise_std

        self.data_size, self.size_x, self.size_y, self.size_z = self.raw_input.shape
        self.grid = self._generate_grid()

        # Entrada = (grid + campo físico)
        self.input = self.raw_input.unsqueeze(-1)
        self.input = torch.cat((self.grid, self.input), dim=-1).permute(0, 4, 1, 2, 3)

    def _generate_grid(self):
        x = torch.linspace(0, 1, self.size_x)
        y = torch.linspace(0, 1, self.size_y)
        z = torch.linspace(0, 1, self.size_z)
        grid = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)  # (X,Y,Z,3)
        grid = grid.unsqueeze(0).repeat(self.data_size, 1, 1, 1, 1)
        return grid

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        x = self.input[idx]
        y = self.output[idx]
        if self.augment:
            x, y = augment_3d(x.clone(), y.clone())
        if self.add_noise:
            x = x + torch.randn_like(x) * self.noise_std
        return x, y


# ------------------------------------------
# ⚙️ Modelo Lightning
# ------------------------------------------
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
        loss = self.loss(y_hat.squeeze(0), y)
        mse = F.mse_loss(y_hat.squeeze(0), y)
        rel_error = self.loss.rel(y_hat.squeeze(0), y)
        self.log_dict({'train_loss': loss, 'train_mse': mse, 'train_rel_error': rel_error}, prog_bar=True)
        return 5*loss + mse

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat.squeeze(0), y)
        mse = F.mse_loss(y_hat.squeeze(0), y)
        rel_error = self.loss.rel(y_hat.squeeze(0), y)
        self.log_dict({'val_loss': loss, 'val_mse': mse, 'val_rel_error': rel_error}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return [optimizer], [scheduler]


# ------------------------------------------
# 🧪 Script principal
# ------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--n_gpus', type=int, default=-1)
    parser.add_argument('--n_nodes', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--data_path', type=str, default='vlasov_dataset.npy')
    parser.add_argument('--output_path', type=str, default='experiments/VlasovPoisson')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # Carga de datos
    raw_data = torch.tensor(np.load(args.data_path), dtype=torch.float32) # Shape: (17, 256, 256, 256, 2) [N, X, Y, Z, (input, output)]
    print("🔄 Shape of data: {}".format(raw_data.shape))
    
    # Normalización datos
    raw_data = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min())

    # Reducción de tamaño 256 -> 128
    raw_data = F.interpolate(raw_data.permute(0, 4, 1, 2, 3), size=(128, 128, 128), mode='trilinear', align_corners=True)
    raw_data = raw_data.permute(0, 2, 3, 4, 1)  # Volver a la forma original (N, X, Y, Z, C)
    print("🔄 Shape of data after interpolation: {}".format(raw_data.shape))

    # División
    train_raw = raw_data[:14]
    eval_raw = raw_data[14:]
    

    # Dataset base
    base_train = Dataset3D(train_raw, augment=False)
    dataset_train = ConcatDataset([base_train])  # 🔁 duplicamos con augmentation
    augmented = Dataset3D(train_raw, augment=True)
    dataset_train = ConcatDataset([base_train, augmented])  # 🔁 duplicamos con augmentation
    dataset_eval = Dataset3D(eval_raw, augment=False)

    # Dataloaders
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=25, persistent_workers=True, pin_memory=True)
    eval_loader = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False,
                             num_workers=25, persistent_workers=True, pin_memory=True)

    # FNO modelo
    fno_params = dict(
        n_modes=(64, 64, 64),
        hidden_channels=20,
        n_layers=4,
        in_channels=4,
        lifting_channels=64,
        projection_channels=64,
        out_channels=1,
        factorization='tucker',
        implementation='factorized',
        fno_block_precision='mixed',
        stabilizer='tanh',
        rank=0.5,
    )
    model = FNO(**fno_params)
    lit_model = LitFNO(model, lr=args.lr, weight_decay=args.weight_decay, **fno_params)
    print("✅ Modelo FNO creado.")

    # Loggers
    logger_csv = CSVLogger(args.output_path, name=f"csv_logs_{args.exp_name}")
    logger_tb = TensorBoardLogger(args.output_path, name=f"tb_logs_{args.exp_name}")

    # Entrenamiento
    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=[logger_csv, logger_tb],
        devices=args.n_gpus,
        num_nodes=args.n_nodes,
        accumulate_grad_batches=8,
    )
    t0 = time.time()
    
    try:
        trainer.fit(lit_model, train_loader, eval_loader)
    except KeyboardInterrupt:
        print("⏹️ Entrenamiento interrumpido por el usuario.")   
        pass
        
    tf = time.time()  
    print(f"⏱️ Tiempo de entrenamiento: {pd.to_datetime(tf - t0, unit='s').strftime('%H:%M:%S')}")

    # Guardar modelo
    model_path = f"{args.output_path}/{args.exp_name}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Modelo guardado en {model_path}")

    # Visualización
    lit_model.model.eval().cuda()
    
    x, y = next(iter(eval_loader))
    with torch.no_grad():
        y_hat = lit_model.model(x.cuda()).cpu()

    # Mostrar plano medio en Z
    mid_z = y.shape[2] // 2
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(y[0, :, :, mid_z], cmap='viridis')
    axs[0].set_title("🟢 Real Output")
    axs[1].imshow(y_hat[0, 0, :, :, mid_z], cmap='viridis')
    axs[1].set_title("🔵 Predicted Output")
    axs[2].imshow(x[0, 3, :, :, mid_z], cmap='viridis')
    axs[2].set_title("🟠 Input (Field)")
    plt.tight_layout()
    plt.savefig(f"{args.output_path}/{args.exp_name}_comparison.png")
    print("📸 Resultados guardados.")