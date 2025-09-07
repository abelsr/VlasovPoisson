# Vlasov-Poisson with Fourier Neural Operators (3D)

This project trains 3D Neural Operators (FNO/UNO) to learn mappings on Vlasov–Poisson-like data. It includes single-GPU and two‑GPU model‑parallel training options, logging/visualization via PyTorch Lightning, and inference utilities for generating predictions and GIFs.


## Features
- 3D FNO/UNO models with grid positional embeddings (x, y, z + field).
- PyTorch Lightning training with CSV/TensorBoard logs and LR monitor.
- Data augmentation: 3D rotations and random scaling; optional flips/noise in `train_model.py`.
- Two-GPU model-parallel training alternatives (`train_fno_split2gpu_stages*.py`).
- Periodic prediction snapshots saved as images during validation.
- Inference script with optional multi-step forward and GIF export.


## Repository Structure
- `train_fno_vlasov.py`: Main Lightning trainer for 3D FNO/UNO; saves config, weights, images, and predictions.
- `train_fno_split2gpu_stages.py`: Manual 2‑GPU split + microbatching (no Lightning wrapper).
- `train_fno_split2gpu_stages_pl.py`: Lightning module that wraps the two stages (model-parallel on 2 GPUs).
- `inference.py`: Loads a trained model/config and runs inference, exports `.npy`, `.png`, and an animated `.gif`.
- `modules/vlasov_dataset.py`: 3D dataset helper (grid embedding + optional augmentation).
- `notebooks/`: Data analysis and plotting utilities.
- `experiments/`: Auto-created per run, contains logs, images, weights, predictions.
- `outputs/`: Default folder for inference artifacts.


## Data
- Expected dataset file: `vlasov_dataset.npy` (large; e.g., multiple GB).
- Expected shape: `(N, X, Y, Z, C)` with `C=2` where:
  - `[..., 0]` = input field
  - `[..., 1]` = target field
- Training scripts internally interpolate volumes to `(128, 128, 128)` by default.

Place the dataset anywhere and pass its path via `--data_path` (recommended: avoid using the absolute default in scripts).


## Environment
- Python 3.10+ recommended
- NVIDIA GPU with CUDA for training/inference

Quick start using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick your CUDA
pip install lightning neuralop numpy pandas matplotlib imageio loguru
```

Notes:
- The import namespace for the operator library is `neuralop` (installed via `pip install neuralop`).
- If installing CUDA wheels, choose the index URL matching your CUDA version.


## Single‑GPU Training
Example run (override the absolute default paths):

```bash
python train_fno_vlasov.py \
  --exp_name normalize \
  --n_gpus 1 \
  --batch_size 1 \
  --epochs 200 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --data_path /path/to/vlasov_dataset.npy \
  --output_path experiments/VlasovPoisson
```

Outputs are written to `experiments/VlasovPoisson/<YYYY-MM-DD_HH-MM-SS>_<exp_name>/`, including:
- `config.json`: Saved hyperparameters/model config.
- `csv_logs/` and `tb_logs/`: Metrics.
- `images/epoch_XXXX.png`: Validation snapshots (e.g., `.../images/epoch_0040.png`).
- `<exp_name>.pt`: Model weights (FNO state dict).
- `<exp_name>_prediction.npy`, `<exp_name>_all_predictions.npy`: Saved predictions.
- `final_predictions/*.png`: Per-sample visualization at the end of training.

Tip: Reduce memory by lowering `n_modes`, `hidden_channels`, or `batch_size` in the script/config.


## Two‑GPU Model‑Parallel Training
You can split the FNO into two stages that run on separate GPUs. Two alternatives are provided.

1) Manual training loop (no Lightning wrapper):
```bash
python train_fno_split2gpu_stages.py \
  --exp_name myrun \
  --data_path /path/to/vlasov_dataset.npy \
  --output_path experiments/VlasovPoisson \
  --epochs 200 \
  --batch_size 1 \
  --lr 1e-3 \
  --devices 0,1 \
  --chunks 4
```

2) Lightning-wrapped two-stage trainer:
```bash
python train_fno_split2gpu_stages_pl.py \
  --exp_name myrun \
  --data_path /path/to/vlasov_dataset.npy \
  --output_path experiments/VlasovPoisson \
  --epochs 200 \
  --batch_size 1 \
  --lr 5e-4 \
  --devices 0,1 \
  --chunks 4
```

Both variants create a timestamped folder under `experiments/VlasovPoisson/` and write images/logs. The split depth (`--split_k`) is auto-chosen to half the FNO blocks unless overridden.


## Inference
Run inference on a saved experiment directory (it must contain `config.json` and a model `.pt`).

```bash
python inference.py \
  --checkpoint_path experiments/VlasovPoisson/<YYYY-MM-DD_HH-MM-SS>_<exp_name> \
  --data_path /path/to/data_to_infer.npy \
  --output_dir outputs \
  --forward_steps 1
```

Artifacts:
- `outputs/predictions.npy`: Stacked predictions.
- `outputs/prediction_*.png`: Side-by-side input/prediction plots.
- `outputs/predictions_z0.gif`: GIF of repeated forward steps (if `--forward_steps > 1`).

Important: `inference.py` currently looks for a weights file named `new_arch.pt` inside `--checkpoint_path`. If your training script saved a different name (e.g., `<exp_name>.pt`), either:
- rename your weights file to `new_arch.pt`, or
- edit `inference.py` to load the actual filename you saved.


## Logging & Visualization
- TensorBoard logs: `experiments/.../tb_logs/`
- CSV logs: `experiments/.../csv_logs/`
- Validation images: `experiments/.../images/epoch_XXXX.png` (e.g., `experiments/VlasovPoisson/2025-09-04_23-33-58_normalize/images/epoch_0040.png`).


## Tips & Troubleshooting
- Memory pressure: Use smaller `n_modes`, `hidden_channels`, or `batch_size`. Interpolation to `128^3` is already applied; you can change that in the scripts.
- Lightning/accelerator: Ensure a working CUDA setup; pass `--n_gpus 1` or use the model‑parallel scripts for 2 GPUs.
- Package imports: If `neuralop` import fails, ensure `pip install neuralop` completed successfully. Lightning is imported as `import lightning as L`.
- Data shape: Must be `(N, X, Y, Z, 2)`; the dataset class constructs 4 input channels = `(x, y, z, field)` and predicts one target channel.


## Acknowledgements
- Built with PyTorch, Lightning, and the `neuralop` library for Neural Operators/FNOs.


## License
Specify a license of your choice in a `LICENSE` file if you plan to distribute this code or models.

