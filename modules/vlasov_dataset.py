import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

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
            x, y = self.augment_3d(x.clone(), y.clone())
        return x, y
    
    def augment_3d(self, input_tensor, output_tensor):
        for dims in [(1, 2), (1, 3), (2, 3)]:
            k = np.random.randint(0, 3)
            input_tensor = torch.rot90(input_tensor, k, dims=dims)
            output_tensor = torch.rot90(output_tensor, k, dims=[d - 1 for d in dims])

        scale_factor = np.random.choice([0.125, 0.25, 0.5, 1])
        if scale_factor == 1:
            return input_tensor, output_tensor

        input_tensor = F.interpolate(input_tensor.unsqueeze(0), scale_factor=scale_factor, mode='nearest').squeeze(0)
        output_tensor = F.interpolate(
            output_tensor.unsqueeze(0).unsqueeze(0),
            scale_factor=scale_factor, mode='nearest'
        ).squeeze(0).squeeze(0)
        return input_tensor, output_tensor