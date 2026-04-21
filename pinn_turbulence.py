#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - torch is required for execution
    raise SystemExit("PyTorch is required. Install with `pip install torch`.") from exc


@dataclass
class Normalizer:
    min_value: torch.Tensor
    max_value: torch.Tensor

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        scale = self.max_value - self.min_value
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        return (values - self.min_value) / scale * 2.0 - 1.0


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: int, hidden_width: int) -> None:
        super().__init__()
        layers = []
        layer_sizes = [in_dim] + [hidden_width] * hidden_layers + [out_dim]
        for idx in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[idx], layer_sizes[idx + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def load_npy(path: str, name: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} file not found: {path}")
    return np.load(path)


def describe_array(name: str, array: np.ndarray) -> None:
    print(f"{name}: shape={array.shape}, dtype={array.dtype}, min={array.min():.6f}, max={array.max():.6f}")


def prepare_field_data(
    data: np.ndarray,
    vars_count: Optional[int],
    basis: Optional[np.ndarray],
    mean: Optional[np.ndarray],
) -> Tuple[np.ndarray, int]:
    if data.ndim != 2:
        raise ValueError("Data array must be 2D [time, features].")

    if basis is not None:
        if basis.ndim != 2:
            raise ValueError("Basis array must be 2D [features, flattened_field].")
        if data.shape[1] != basis.shape[0]:
            raise ValueError("Basis first dimension must match data feature dimension.")
        reconstructed = data @ basis
    else:
        reconstructed = data

    if mean is not None:
        if mean.ndim != 1 or mean.shape[0] != reconstructed.shape[1]:
            raise ValueError("Mean array must be 1D and match flattened field dimension.")
        reconstructed = reconstructed + mean

    feature_count = reconstructed.shape[1]
    if vars_count is None:
        if feature_count % 3 == 0:
            vars_count = 3
        elif feature_count % 4 == 0:
            vars_count = 4
        else:
            raise ValueError("Unable to infer variable count. Provide --vars.")

    if feature_count % vars_count != 0:
        raise ValueError("Feature dimension must be divisible by variable count.")

    points = feature_count // vars_count
    field = reconstructed.reshape(reconstructed.shape[0], points, vars_count)
    return field.astype(np.float32), vars_count


def prepare_time_series(data: np.ndarray, time_values: Optional[np.ndarray]) -> np.ndarray:
    if time_values is None:
        return np.arange(data.shape[0], dtype=np.float32)
    if time_values.ndim != 1 or time_values.shape[0] != data.shape[0]:
        raise ValueError("Time array must be 1D and match data time dimension.")
    return time_values.astype(np.float32)


def sample_data_batch(
    field: torch.Tensor,
    coords: torch.Tensor,
    time_norm: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    time_idx = torch.randint(0, field.shape[0], (batch_size,), device=field.device)
    point_idx = torch.randint(0, coords.shape[0], (batch_size,), device=field.device)
    coord_batch = coords[point_idx]
    t_batch = time_norm[time_idx].unsqueeze(1)
    inputs = torch.cat([coord_batch, t_batch], dim=1)
    targets = field[time_idx, point_idx]
    return inputs, targets


def gradients(target: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        target,
        inputs,
        grad_outputs=torch.ones_like(target),
        create_graph=True,
        retain_graph=True,
    )[0]


def navier_stokes_residual(
    model: nn.Module, inputs: torch.Tensor, viscosity: float, coord_dim: int, vars_count: int
) -> Tuple[torch.Tensor, ...]:
    inputs.requires_grad_(True)
    prediction = model(inputs)

    if coord_dim == 2 and vars_count == 3:
        u, v, p = prediction[:, 0], prediction[:, 1], prediction[:, 2]
        grads_u = gradients(u, inputs)
        grads_v = gradients(v, inputs)
        grads_p = gradients(p, inputs)

        u_x, u_y, u_t = grads_u[:, 0], grads_u[:, 1], grads_u[:, 2]
        v_x, v_y, v_t = grads_v[:, 0], grads_v[:, 1], grads_v[:, 2]
        p_x, p_y = grads_p[:, 0], grads_p[:, 1]

        u_xx = gradients(u_x, inputs)[:, 0]
        u_yy = gradients(u_y, inputs)[:, 1]
        v_xx = gradients(v_x, inputs)[:, 0]
        v_yy = gradients(v_y, inputs)[:, 1]

        res_u = u_t + u * u_x + v * u_y + p_x - viscosity * (u_xx + u_yy)
        res_v = v_t + u * v_x + v * v_y + p_y - viscosity * (v_xx + v_yy)
        res_c = u_x + v_y
        return (res_u, res_v, res_c)

    if coord_dim == 3 and vars_count == 4:
        u, v, w, p = prediction[:, 0], prediction[:, 1], prediction[:, 2], prediction[:, 3]
        grads_u = gradients(u, inputs)
        grads_v = gradients(v, inputs)
        grads_w = gradients(w, inputs)
        grads_p = gradients(p, inputs)

        u_x, u_y, u_z, u_t = grads_u[:, 0], grads_u[:, 1], grads_u[:, 2], grads_u[:, 3]
        v_x, v_y, v_z, v_t = grads_v[:, 0], grads_v[:, 1], grads_v[:, 2], grads_v[:, 3]
        w_x, w_y, w_z, w_t = grads_w[:, 0], grads_w[:, 1], grads_w[:, 2], grads_w[:, 3]
        p_x, p_y, p_z = grads_p[:, 0], grads_p[:, 1], grads_p[:, 2]

        u_xx = gradients(u_x, inputs)[:, 0]
        u_yy = gradients(u_y, inputs)[:, 1]
        u_zz = gradients(u_z, inputs)[:, 2]
        v_xx = gradients(v_x, inputs)[:, 0]
        v_yy = gradients(v_y, inputs)[:, 1]
        v_zz = gradients(v_z, inputs)[:, 2]
        w_xx = gradients(w_x, inputs)[:, 0]
        w_yy = gradients(w_y, inputs)[:, 1]
        w_zz = gradients(w_z, inputs)[:, 2]

        res_u = u_t + u * u_x + v * u_y + w * u_z + p_x - viscosity * (u_xx + u_yy + u_zz)
        res_v = v_t + u * v_x + v * v_y + w * v_z + p_y - viscosity * (v_xx + v_yy + v_zz)
        res_w = w_t + u * w_x + v * w_y + w * w_z + p_z - viscosity * (w_xx + w_yy + w_zz)
        res_c = u_x + v_y + w_z
        return (res_u, res_v, res_w, res_c)

    raise ValueError("Coordinate dimension and variable count are incompatible with Navier-Stokes residuals.")


def sample_collocation(
    coords: torch.Tensor, time_norm: torch.Tensor, batch_size: int
) -> torch.Tensor:
    point_idx = torch.randint(0, coords.shape[0], (batch_size,), device=coords.device)
    time_idx = torch.randint(0, time_norm.shape[0], (batch_size,), device=coords.device)
    coord_batch = coords[point_idx]
    t_batch = time_norm[time_idx].unsqueeze(1)
    return torch.cat([coord_batch, t_batch], dim=1)


def predict_field(
    model: nn.Module,
    coords: torch.Tensor,
    time_value: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, coords.shape[0], batch_size):
            coord_batch = coords[start : start + batch_size]
            t_batch = time_value.repeat(coord_batch.shape[0], 1)
            inputs = torch.cat([coord_batch, t_batch], dim=1)
            outputs.append(model(inputs).cpu())
    return torch.cat(outputs, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PINN for turbulence prediction.")
    parser.add_argument("--data", required=True, help="Path to reduced or reconstructed data .npy file.")
    parser.add_argument("--coords", help="Path to coordinates .npy file with shape [points, dim].")
    parser.add_argument("--time", help="Optional time values .npy file with shape [time].")
    parser.add_argument("--basis", help="Optional basis .npy file for reconstruction.")
    parser.add_argument("--mean", help="Optional mean .npy file for reconstruction.")
    parser.add_argument("--vars", type=int, help="Number of variables per spatial point (e.g., 3 for u,v,p).")
    parser.add_argument(
        "--train-time-steps",
        dest="train_steps",
        type=int,
        default=400,
        help="Number of consecutive time steps for training.",
    )
    parser.add_argument(
        "--test-time-steps",
        dest="test_steps",
        type=int,
        default=200,
        help="Number of consecutive time steps for prediction.",
    )
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs.")
    parser.add_argument("--data-batch", type=int, default=4096, help="Batch size for data loss.")
    parser.add_argument("--collocation-batch", type=int, default=4096, help="Batch size for PDE residual.")
    parser.add_argument("--hidden-layers", type=int, default=6, help="Number of hidden layers.")
    parser.add_argument("--hidden-width", type=int, default=64, help="Width of hidden layers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--viscosity", type=float, default=0.01, help="Kinematic viscosity.")
    parser.add_argument("--data-weight", type=float, default=1.0, help="Weight for data loss.")
    parser.add_argument("--pde-weight", type=float, default=1.0, help="Weight for PDE residual loss.")
    parser.add_argument("--ic-weight", type=float, default=0.5, help="Weight for initial condition loss.")
    parser.add_argument("--bc-indices", help="Optional boundary indices .npy file.")
    parser.add_argument("--bc-weight", type=float, default=0.2, help="Weight for boundary condition loss.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save outputs.")
    parser.add_argument("--predict-batch", type=int, default=4096, help="Batch size for prediction.")
    parser.add_argument("--log-every", type=int, default=200, help="Log every N epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--plots", action="store_true", help="Save matplotlib plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_npy(args.data, "data")
    describe_array("data", data)

    time_values = load_npy(args.time, "time") if args.time else None
    if time_values is not None:
        describe_array("time", time_values)

    basis = load_npy(args.basis, "basis") if args.basis else None
    if basis is not None:
        describe_array("basis", basis)

    mean = load_npy(args.mean, "mean") if args.mean else None
    if mean is not None:
        describe_array("mean", mean)

    coords = None
    if args.coords:
        coords = load_npy(args.coords, "coords")
        if coords.ndim == 1:
            coords = coords[:, None]
        describe_array("coords", coords)

    time_series = prepare_time_series(data, time_values)

    train_steps = args.train_steps
    test_steps = args.test_steps
    if train_steps + test_steps > data.shape[0]:
        raise ValueError("Train + test steps exceed available time steps.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if coords is None:
        print("No coordinates provided. Falling back to coefficient prediction (time-only).")
        data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
        time_tensor = torch.tensor(time_series, dtype=torch.float32, device=device)
        time_norm = (time_tensor - time_tensor.min()) / (time_tensor.max() - time_tensor.min() + 1e-8)
        time_norm = time_norm * 2.0 - 1.0
        model = MLP(1, data_tensor.shape[1], args.hidden_layers, args.hidden_width).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.5)
        mse = nn.MSELoss()

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            idx = torch.randint(0, train_steps, (args.data_batch,), device=device)
            t_batch = time_norm[idx].unsqueeze(1)
            target = data_tensor[idx]
            pred = model(t_batch)
            loss = mse(pred, target) * args.data_weight
            loss.backward()
            optimizer.step()
            scheduler.step()
            if epoch % args.log_every == 0 or epoch == 1:
                print(f"Epoch {epoch}: loss={loss.item():.6f}")

        os.makedirs(args.output_dir, exist_ok=True)
        predictions = []
        model.eval()
        with torch.no_grad():
            for t_idx in range(train_steps, train_steps + test_steps):
                t_val = time_norm[t_idx].view(1, 1)
                predictions.append(model(t_val).cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        np.save(os.path.join(args.output_dir, "predictions.npy"), predictions)

        true = data[train_steps : train_steps + test_steps]
        mse_per_t = np.mean((predictions - true) ** 2, axis=1)
        rel_err = np.linalg.norm(predictions - true, axis=1) / (np.linalg.norm(true, axis=1) + 1e-8)
        metrics = {
            "mse_mean": float(mse_per_t.mean()),
            "rel_error_mean": float(rel_err.mean()),
        }
        with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        np.save(os.path.join(args.output_dir, "mse_over_time.npy"), mse_per_t)
        np.save(os.path.join(args.output_dir, "rel_error_over_time.npy"), rel_err)

        if args.plots:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib not installed; skipping plots.")
            else:
                plt.figure()
                plt.plot(mse_per_t)
                plt.title("Test MSE over time")
                plt.xlabel("Test step")
                plt.ylabel("MSE")
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, "mse_over_time.png"))

        return

    field, vars_count = prepare_field_data(data, args.vars, basis, mean)
    coord_dim = coords.shape[1]
    if coord_dim not in (2, 3):
        raise ValueError("Coordinate dimension must be 2 or 3 for PINN mode.")

    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    coords_normalizer = Normalizer(coords_tensor.min(dim=0).values, coords_tensor.max(dim=0).values)
    coords_norm = coords_normalizer.normalize(coords_tensor)

    time_tensor = torch.tensor(time_series, dtype=torch.float32, device=device)
    time_normalizer = Normalizer(time_tensor.min(), time_tensor.max())
    time_norm = time_normalizer.normalize(time_tensor).unsqueeze(1)

    field_tensor = torch.tensor(field, dtype=torch.float32, device=device)
    mean_field = field_tensor[:train_steps].mean(dim=(0, 1), keepdim=True)
    std_field = field_tensor[:train_steps].std(dim=(0, 1), keepdim=True) + 1e-6
    field_tensor = (field_tensor - mean_field) / std_field

    model = MLP(coord_dim + 1, vars_count, args.hidden_layers, args.hidden_width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.5)
    mse = nn.MSELoss()

    bc_indices = None
    if args.bc_indices:
        bc_indices = load_npy(args.bc_indices, "bc_indices").astype(int)
        if bc_indices.ndim != 1:
            raise ValueError("Boundary indices must be 1D.")

    use_pde = args.pde_weight > 0.0
    if coord_dim == 2 and vars_count != 3:
        raise ValueError("For 2D PINN mode, vars must be 3 (u,v,p).")
    if coord_dim == 3 and vars_count != 4:
        raise ValueError("For 3D PINN mode, vars must be 4 (u,v,w,p).")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        inputs, targets = sample_data_batch(field_tensor[:train_steps], coords_norm, time_norm[:train_steps], args.data_batch)
        preds = model(inputs)
        data_loss = mse(preds, targets)

        ic_inputs = torch.cat([coords_norm, time_norm[0].repeat(coords_norm.shape[0], 1)], dim=1)
        ic_targets = field_tensor[0]
        ic_preds = model(ic_inputs)
        ic_loss = mse(ic_preds, ic_targets)

        bc_loss = torch.tensor(0.0, device=device)
        if bc_indices is not None and bc_indices.size > 0:
            bc_idx = torch.tensor(bc_indices, device=device)
            bc_coords = coords_norm[bc_idx]
            t_idx = torch.randint(0, train_steps, (bc_coords.shape[0],), device=device)
            bc_times = time_norm[t_idx]
            bc_inputs = torch.cat([bc_coords, bc_times], dim=1)
            bc_targets = field_tensor[t_idx, bc_idx]
            bc_preds = model(bc_inputs)
            bc_loss = mse(bc_preds, bc_targets)

        pde_loss = torch.tensor(0.0, device=device)
        if use_pde:
            collocation = sample_collocation(coords_norm, time_norm[:train_steps], args.collocation_batch)
            residuals = navier_stokes_residual(model, collocation, args.viscosity, coord_dim, vars_count)
            pde_loss = sum(torch.mean(res ** 2) for res in residuals)

        loss = (
            args.data_weight * data_loss
            + args.ic_weight * ic_loss
            + args.bc_weight * bc_loss
            + args.pde_weight * pde_loss
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch}: loss={loss.item():.6f} data={data_loss.item():.6f} "
                f"ic={ic_loss.item():.6f} bc={bc_loss.item():.6f} pde={pde_loss.item():.6f}"
            )

    os.makedirs(args.output_dir, exist_ok=True)
    predictions = []
    for t_idx in range(train_steps, train_steps + test_steps):
        t_val = time_norm[t_idx]
        pred = predict_field(model, coords_norm, t_val, args.predict_batch)
        predictions.append(pred)
    predictions = torch.stack(predictions, dim=0)
    predictions = predictions * std_field + mean_field
    predictions_np = predictions.cpu().numpy()
    np.save(os.path.join(args.output_dir, "predictions.npy"), predictions_np)

    true = field[train_steps : train_steps + test_steps]
    mse_per_t = np.mean((predictions_np - true) ** 2, axis=(1, 2))
    rel_err = np.linalg.norm(predictions_np - true, axis=(1, 2)) / (
        np.linalg.norm(true, axis=(1, 2)) + 1e-8
    )
    metrics = {
        "mse_mean": float(mse_per_t.mean()),
        "rel_error_mean": float(rel_err.mean()),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    np.save(os.path.join(args.output_dir, "mse_over_time.npy"), mse_per_t)
    np.save(os.path.join(args.output_dir, "rel_error_over_time.npy"), rel_err)

    if args.plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plots.")
        else:
            plt.figure()
            plt.plot(mse_per_t)
            plt.title("Test MSE over time")
            plt.xlabel("Test step")
            plt.ylabel("MSE")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "mse_over_time.png"))


if __name__ == "__main__":
    main()
