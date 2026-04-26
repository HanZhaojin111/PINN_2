#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - torch is optional for raw predictions
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if nn is not None:

    class MLP(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, hidden_layers: int, hidden_width: int) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            layer_sizes = [in_dim] + [hidden_width] * hidden_layers + [out_dim]
            for idx in range(len(layer_sizes) - 2):
                layers.append(nn.Linear(layer_sizes[idx], layer_sizes[idx + 1]))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
            self.net = nn.Sequential(*layers)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.net(inputs)

else:

    class MLP:  # type: ignore[misc]
        def __init__(self, *_: object, **__: object) -> None:
            raise SystemExit("PyTorch is required for autoencoder decoding. Install with `pip install torch`.")


def load_npy(path: str, name: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} file not found: {path}")
    return np.load(path)


def describe_array(name: str, array: np.ndarray) -> None:
    print(f"{name}: shape={array.shape}, dtype={array.dtype}, min={array.min():.6f}, max={array.max():.6f}")


def load_checkpoint(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    checkpoint = torch.load(path, map_location="cpu")
    config: Dict[str, int] = {}
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("config"), dict):
            config = checkpoint["config"]
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
            if not state_dict:
                state_dict = checkpoint  # type: ignore[assignment]
    else:
        state_dict = checkpoint  # type: ignore[assignment]
    return state_dict, config


def extract_decoder_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(key.startswith("decoder.") for key in state_dict):
        return {key[len("decoder.") :]: value for key, value in state_dict.items() if key.startswith("decoder.")}
    return state_dict


def infer_decoder_arch(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int]:
    indices = sorted(
        {
            int(match.group(1))
            for key in state_dict
            if (match := re.match(r"net\.(\d+)\.weight", key))
        }
    )
    if not indices:
        raise ValueError("Unable to infer decoder architecture from checkpoint weights.")
    first = state_dict[f"net.{indices[0]}.weight"]
    last = state_dict[f"net.{indices[-1]}.weight"]
    latent_dim = first.shape[1]
    hidden_width = first.shape[0]
    output_dim = last.shape[0]
    hidden_layers = len(indices) - 1
    return int(latent_dim), int(hidden_layers), int(hidden_width), int(output_dim)


def resolve_config_int(value: Optional[int], config: Dict[str, int], keys: Iterable[str]) -> Optional[int]:
    if value is not None:
        return value
    for key in keys:
        if key in config:
            return int(config[key])
    return None


def format_ascii(array: np.ndarray) -> str:
    flat = array.ravel()
    if flat.size == 0:
        return ""
    if np.issubdtype(flat.dtype, np.floating):
        text = np.array2string(
            flat,
            separator=" ",
            max_line_width=1_000_000,
            threshold=flat.size,
            formatter={"float_kind": lambda value: f"{value:.6e}"},
        )
    else:
        text = np.array2string(
            flat,
            separator=" ",
            max_line_width=1_000_000,
            threshold=flat.size,
            formatter={"int_kind": lambda value: str(int(value))},
        )
    return text.strip("[]")


def write_vtu(path: str, points: np.ndarray, point_data: Dict[str, np.ndarray]) -> None:
    num_points = points.shape[0]
    connectivity = np.arange(num_points, dtype=np.int32)
    offsets = np.arange(1, num_points + 1, dtype=np.int32)
    types = np.ones(num_points, dtype=np.uint8)
    lines = [
        '<?xml version="1.0"?>\n',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n',
        "  <UnstructuredGrid>\n",
        f'    <Piece NumberOfPoints="{num_points}" NumberOfCells="{num_points}">\n',
        "      <Points>\n",
        '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n',
        f"          {format_ascii(points.astype(np.float32))}\n",
        "        </DataArray>\n",
        "      </Points>\n",
        "      <Cells>\n",
        '        <DataArray type="Int32" Name="connectivity" format="ascii">\n',
        f"          {format_ascii(connectivity)}\n",
        "        </DataArray>\n",
        '        <DataArray type="Int32" Name="offsets" format="ascii">\n',
        f"          {format_ascii(offsets)}\n",
        "        </DataArray>\n",
        '        <DataArray type="UInt8" Name="types" format="ascii">\n',
        f"          {format_ascii(types)}\n",
        "        </DataArray>\n",
        "      </Cells>\n",
        "      <PointData>\n",
    ]
    for name, values in point_data.items():
        lines.append(f'        <DataArray type="Float32" Name="{name}" format="ascii">\n')
        lines.append(f"          {format_ascii(values.astype(np.float32))}\n")
        lines.append("        </DataArray>\n")
    lines.extend(
        [
            "      </PointData>\n",
            "    </Piece>\n",
            "  </UnstructuredGrid>\n",
            "</VTKFile>\n",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode PINN predictions with an autoencoder and export VTU files.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.npy.")
    parser.add_argument("--autoencoder", help="Path to autoencoder checkpoint (.pt/.pth) for latent predictions.")
    parser.add_argument(
        "--coords",
        help="Path to coordinates .npy with shape [points, dim]. If omitted, index-based coordinates are used.",
    )
    parser.add_argument("--vars", type=int, help="Number of variables per spatial point.")
    parser.add_argument("--var-names", help="Comma-separated variable names for VTU output.")
    parser.add_argument("--latent-dim", type=int, help="Latent dimension (overrides checkpoint config).")
    parser.add_argument("--hidden-layers", type=int, help="Hidden layers in decoder (overrides checkpoint config).")
    parser.add_argument("--hidden-width", type=int, help="Hidden width in decoder (overrides checkpoint config).")
    parser.add_argument("--output-dim", type=int, help="Decoder output dimension (overrides checkpoint config).")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for decoding.")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N VTU files (0 to disable).")
    parser.add_argument("--output-dir", default="vtu", help="Directory to write VTU files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictions = load_npy(args.predictions, "predictions")
    describe_array("predictions", predictions)

    if args.autoencoder:
        if torch is None:
            raise SystemExit("PyTorch is required for autoencoder decoding. Install with `pip install torch`.")
        if predictions.ndim != 2:
            raise ValueError("predictions.npy must be 2D [time, latent_dim] when using --autoencoder.")
        if args.vars is None:
            raise ValueError("--vars is required when decoding with an autoencoder.")
        vars_count = args.vars

        state_dict, config = load_checkpoint(args.autoencoder)
        decoder_state = extract_decoder_state(state_dict)
        inferred_latent, inferred_layers, inferred_width, inferred_output = infer_decoder_arch(decoder_state)

        latent_dim = resolve_config_int(args.latent_dim, config, ["latent_dim", "z_dim"]) or inferred_latent
        hidden_layers = resolve_config_int(args.hidden_layers, config, ["hidden_layers"]) or inferred_layers
        hidden_width = resolve_config_int(args.hidden_width, config, ["hidden_width"]) or inferred_width
        output_dim = resolve_config_int(args.output_dim, config, ["output_dim", "out_dim"]) or inferred_output

        if predictions.shape[1] != latent_dim:
            raise ValueError("Predictions feature dimension does not match decoder latent dimension.")

        if output_dim is None:
            raise ValueError("Decoder output dimension could not be inferred; provide --output-dim.")
        if output_dim % vars_count != 0:
            raise ValueError("Output dimension must be divisible by variable count.")
        points = output_dim // vars_count

        decoder = MLP(latent_dim, output_dim, hidden_layers, hidden_width)
        decoder.load_state_dict(decoder_state)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Decoding on device: {device}")
        decoder.to(device)
        decoder.eval()

        pred_tensor = torch.from_numpy(predictions.astype(np.float32))
        decoded = []
        with torch.no_grad():
            for start in range(0, pred_tensor.shape[0], args.batch_size):
                batch = pred_tensor[start : start + args.batch_size].to(device)
                decoded.append(decoder(batch).cpu().numpy())
        decoded_np = np.concatenate(decoded, axis=0)
        field = decoded_np.reshape(decoded_np.shape[0], points, vars_count)
    else:
        if predictions.ndim == 3:
            field = predictions
            vars_count = field.shape[2]
            if args.vars is not None and args.vars != vars_count:
                raise ValueError("--vars does not match the last dimension of predictions.npy.")
        elif predictions.ndim == 2:
            vars_count = args.vars or 1
            if predictions.shape[1] % vars_count != 0:
                raise ValueError("predictions.npy feature dimension must be divisible by --vars.")
            points = predictions.shape[1] // vars_count
            field = predictions.reshape(predictions.shape[0], points, vars_count)
        else:
            raise ValueError("predictions.npy must be 2D [time, features] or 3D [time, points, vars].")

    points = field.shape[1]
    if args.coords:
        coords = load_npy(args.coords, "coords")
        if coords.ndim == 1:
            coords = coords[:, None]
        if coords.ndim != 2:
            raise ValueError("coords.npy must be 2D [points, dim].")
        if coords.shape[0] != points:
            raise ValueError("coords point count does not match decoded output dimension.")
        describe_array("coords", coords)
    else:
        coords = np.column_stack(
            [
                np.arange(points, dtype=np.float32),
                np.zeros(points, dtype=np.float32),
                np.zeros(points, dtype=np.float32),
            ]
        )
        print("coords not provided; using index-based coordinates.")

    coord_dim = coords.shape[1]
    if coord_dim == 2:
        coords = np.column_stack([coords, np.zeros(points, dtype=coords.dtype)])
        coord_dim = 3
    elif coord_dim != 3:
        raise ValueError("Coordinates must have 2 or 3 dimensions.")

    if args.var_names:
        names = [name.strip() for name in args.var_names.split(",") if name.strip()]
        if len(names) != vars_count:
            raise ValueError("Number of --var-names entries must match --vars.")
        var_names = names
    else:
        var_names = [f"var{idx}" for idx in range(vars_count)]

    os.makedirs(args.output_dir, exist_ok=True)
    for step in range(field.shape[0]):
        point_data = {name: field[step, :, idx] for idx, name in enumerate(var_names)}
        path = os.path.join(args.output_dir, f"frame_{step:04d}.vtu")
        write_vtu(path, coords, point_data)
        if args.log_every > 0 and step % args.log_every == 0:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
