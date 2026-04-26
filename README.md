# PINN_2 (Turbulence PINN Prediction)
Use PINN models to predict turbulence data from reduced coefficients or reconstructed fields.

## Setup
```bash
pip install -r requirements.txt
```

## Data expectations
- `data.npy` (for example, `test_reconstructed(1).npy`) should be a 2D array with shape `[time, features]`.
- If you have spatial coordinates, provide `--coords` (shape `[points, dim]`).
- If your data is reduced (coefficients), provide `--basis` (shape `[features, points * vars]`)
  and optional `--mean` (shape `[points * vars]`) to reconstruct physical fields.

## Run (time-only coefficient prediction)
```bash
python pinn_turbulence.py \
  --data /path/to/data.npy \
  --train-time-steps 400 \
  --test-time-steps 200 \
  --plots
```

## Run (PINN with Navier-Stokes residuals)
```bash
python pinn_turbulence.py \
  --data /path/to/data.npy \
  --coords /path/to/coords.npy \
  --basis /path/to/basis.npy \
  --mean /path/to/mean.npy \
  --vars 3 \
  --train-time-steps 400 \
  --test-time-steps 200 \
  --pde-weight 1.0 \
  --plots
```

## Outputs
- `outputs/predictions.npy`: predictions for the test window.
- `outputs/metrics.json`: mean MSE and relative error.
- `outputs/mse_over_time.npy` and `outputs/rel_error_over_time.npy`.
- Optional plots if `--plots` is provided.

## Autoencoder decode to VTU (Paraview)
If your `predictions.npy` contains latent coefficients from a trained autoencoder, you can decode
them to full fields and export VTU files for Paraview.

```bash
python autoencoder_to_vtu.py \
  --predictions outputs/predictions.npy \
  --autoencoder /path/to/autoencoder.pt \
  --coords /path/to/coords.npy \
  --vars 3 \
  --output-dir vtu
```

Notes:
- `coords.npy` should be shape `[points, dim]` (dim=2 or 3). For 2D it is padded to 3D in VTU.
- `--vars` is the number of variables per point (e.g., 3 for u,v,p).
- Use `--var-names` to label variables in the VTU file (comma-separated).
