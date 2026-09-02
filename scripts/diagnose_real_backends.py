#!/usr/bin/env python
"""Diagnose real-backend loading and forward execution for Uni3D / PTv3.

This script is meant to answer two questions:
1. Does the checkpoint match the model architecture closely enough to load
   without a large key mismatch?
2. If the model loads, which stage is the last one reached before a forward
   pass fails with a CUDA / floating point exception?
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_retrieval.retrieval.dataset import ExportedEpisode
from action_retrieval.retrieval.encoders import (  # noqa: E402
    PointTransformerV3Encoder,
    Uni3DEncoder,
    _best_state_dict_remap,
    _extract_checkpoint_state_dict,
    _load_checkpoint,
    _repo_release_summary,
    _state_dict_alignment_summary,
)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["uni3d", "ptv3", "both"], default="both")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cuda or cpu.")
    parser.add_argument(
        "--uni3d-sample-count",
        type=int,
        default=None,
        help="Override the number of points sampled for the Uni3D smoke test.",
    )
    parser.add_argument(
        "--ptv3-sample-count",
        type=int,
        default=None,
        help="Override the number of points sampled for the PTv3 smoke test.",
    )
    parser.add_argument(
        "--forward-smoke",
        action="store_true",
        help="Run a tiny synthetic forward pass after loading each backend.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "diagnostics" / "real_backend_diagnosis.json",
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def _print_header(title: str) -> None:
    """Implement the _print_header operation used by this module."""
    print("=" * 88)
    print(title)
    print("=" * 88)


def _checkpoint_report(checkpoint_path: Path) -> dict[str, Any]:
    """Implement the _checkpoint_report operation used by this module."""
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = _extract_checkpoint_state_dict(checkpoint)
    top_level_keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    prefix_counts: dict[str, int] = {}
    if isinstance(state_dict, dict):
        for key in state_dict.keys():
            if isinstance(key, str):
                prefix = key.split(".", 1)[0]
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    return {
        "checkpoint_type": str(type(checkpoint)),
        "top_level_keys": top_level_keys,
        "num_tensors": len(state_dict) if isinstance(state_dict, dict) else None,
        "prefix_counts": dict(sorted(prefix_counts.items(), key=lambda item: (-item[1], item[0]))),
        "state_dict": state_dict,
    }


def _make_ptv3_sample(sample_count: int, device: torch.device) -> dict[str, Any]:
    """Implement the _make_ptv3_sample operation used by this module."""
    sampled = torch.randn(sample_count, 6, dtype=torch.float32, device=device)
    return {
        "coord": sampled[:, :3].contiguous(),
        "feat": sampled.contiguous(),
        "batch": torch.zeros((sampled.shape[0],), dtype=torch.long, device=device),
        "grid_size": 0.01,
    }


def _make_uni3d_sample(sample_count: int, device: torch.device) -> torch.Tensor:
    """Implement the _make_uni3d_sample operation used by this module."""
    return torch.randn(1, sample_count, 6, dtype=torch.float32, device=device)


def _summarize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Implement the _summarize_tensor operation used by this module."""
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(tensor.min().item()) if tensor.numel() else None,
        "max": float(tensor.max().item()) if tensor.numel() else None,
        "mean": float(tensor.float().mean().item()) if tensor.numel() else None,
        "std": float(tensor.float().std(unbiased=False).item()) if tensor.numel() else None,
    }


def _attach_stage_hooks(model: torch.nn.Module, label: str) -> list[Any]:
    """Implement the _attach_stage_hooks operation used by this module."""
    hooks = []

    def make_pre_hook(name: str):
        """Implement the make_pre_hook operation used by this module."""
        def _hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            """Implement the _hook operation used by this module."""
            print(f"[{label}] -> enter {name}", flush=True)

        return _hook

    def make_post_hook(name: str):
        """Implement the make_post_hook operation used by this module."""
        def _hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            """Implement the _hook operation used by this module."""
            print(f"[{label}] <- exit {name}", flush=True)

        return _hook

    for name, module in model.named_children():
        hooks.append(module.register_forward_pre_hook(make_pre_hook(name)))
        hooks.append(module.register_forward_hook(make_post_hook(name)))
    return hooks


def _print_alignment_summary(model: torch.nn.Module, checkpoint_state_dict: dict[str, Any], label: str) -> dict[str, Any]:
    """Implement the _print_alignment_summary operation used by this module."""
    model_state_dict = model.state_dict()
    summary = _state_dict_alignment_summary(model_state_dict, checkpoint_state_dict)
    remap_name, remapped_state_dict, remap_summary = _best_state_dict_remap(model_state_dict, checkpoint_state_dict)
    print(f"[{label}] model key count: {summary['model_key_count']}")
    print(f"[{label}] checkpoint key count: {summary['checkpoint_key_count']}")
    print(f"[{label}] raw shared keys: {summary['shared_key_count']}")
    print(f"[{label}] raw shape matches: {summary['shape_match_count']}")
    print(
        f"[{label}] best remap: {remap_name} | "
        f"shared={remap_summary['shared_key_count']} | "
        f"shape_matches={remap_summary['shape_match_count']}"
    )
    if remap_summary["shape_mismatch_count"]:
        preview = remap_summary["shape_mismatches"][:8]
        print(f"[{label}] shape mismatches preview: {preview}")
    return {
        "raw": summary,
        "best_remap": {
            "name": remap_name,
            "summary": remap_summary,
            "remapped_key_count": len(remapped_state_dict),
        },
    }


def _diagnose_uni3d(device: torch.device, forward_smoke: bool, sample_count: int | None) -> dict[str, Any]:
    """Implement the _diagnose_uni3d operation used by this module."""
    _print_header("Uni3D diagnosis")
    encoder = Uni3DEncoder(use_real=True, device=str(device))
    if encoder.checkpoint is None:
        raise FileNotFoundError("UNI3D_CHECKPOINT is not set.")
    print(f"repo_root: {encoder.repo_root}")
    print(f"checkpoint: {encoder.checkpoint}")
    print(f"release expectation: tag={encoder.expected_release_tag!r}, commit={encoder.expected_release_commit!r}")
    print(f"release strict: {encoder.strict_release}")
    print(f"repo layout: {encoder.repo_layout}")
    print(f"repo release summary: {_repo_release_summary(encoder.repo_root)}")

    checkpoint_info = _checkpoint_report(encoder.checkpoint)
    print(f"checkpoint type: {checkpoint_info['checkpoint_type']}")
    print(f"top-level keys: {checkpoint_info['top_level_keys']}")
    print(f"num tensors: {checkpoint_info['num_tensors']}")
    print(f"prefix counts: {checkpoint_info['prefix_counts']}")

    model = encoder._build_real_backend()
    print(f"loaded model type: {type(model)}")
    alignment = _print_alignment_summary(model, checkpoint_info["state_dict"], "uni3d")

    if forward_smoke:
        hooks = _attach_stage_hooks(model, "uni3d")
        try:
            smoke_sample_count = sample_count or encoder.sample_count
            sample = _make_uni3d_sample(smoke_sample_count, device)
            print(f"[uni3d] smoke sample count: {smoke_sample_count}")
            print(f"[uni3d] smoke tensor summary: {_summarize_tensor(sample)}")
            print("[uni3d] running synthetic encode_pc smoke test...", flush=True)
            with torch.inference_mode():
                output = model.encode_pc(sample)
            if isinstance(output, (tuple, list)):
                output = output[0]
            print(f"[uni3d] output type: {type(output)}")
            if isinstance(output, torch.Tensor):
                print(f"[uni3d] output shape: {tuple(output.shape)}")
        finally:
            for hook in hooks:
                hook.remove()

    return alignment


def _diagnose_ptv3(device: torch.device, forward_smoke: bool, sample_count: int | None) -> dict[str, Any]:
    """Implement the _diagnose_ptv3 operation used by this module."""
    _print_header("PTv3 diagnosis")
    encoder = PointTransformerV3Encoder(use_real=True, device=str(device))
    if encoder.checkpoint is None:
        raise FileNotFoundError("PTV3_CHECKPOINT is not set.")
    print(f"repo_root: {encoder.repo_root}")
    print(f"checkpoint: {encoder.checkpoint}")
    print(f"device: {encoder.device}")
    print(f"allow_key_mismatch: {encoder.allow_key_mismatch}")
    print(f"release expectation: tag={encoder.expected_release_tag!r}, commit={encoder.expected_release_commit!r}")
    print(f"release strict: {encoder.strict_release}")
    print(f"repo layout: {encoder.repo_layout}")
    print(f"repo release summary: {_repo_release_summary(encoder.repo_root)}")
    print(f"enable_flash: {encoder.enable_flash}")
    print(f"enable_rpe: {encoder.enable_rpe}")

    checkpoint_info = _checkpoint_report(encoder.checkpoint)
    print(f"checkpoint type: {checkpoint_info['checkpoint_type']}")
    print(f"top-level keys: {checkpoint_info['top_level_keys']}")
    print(f"num tensors: {checkpoint_info['num_tensors']}")
    print(f"prefix counts: {checkpoint_info['prefix_counts']}")

    model = encoder._build_real_backend()
    print(f"loaded model type: {type(model)}")
    alignment = _print_alignment_summary(model, checkpoint_info["state_dict"], "ptv3")

    if forward_smoke:
        hooks = _attach_stage_hooks(model, "ptv3")
        try:
            smoke_sample_count = sample_count or encoder.sample_count
            sample = _make_ptv3_sample(smoke_sample_count, device)
            print(f"[ptv3] smoke sample count: {smoke_sample_count}")
            print(f"[ptv3] coord summary: {_summarize_tensor(sample['coord'])}")
            print(f"[ptv3] feat summary: {_summarize_tensor(sample['feat'])}")
            print(f"[ptv3] batch summary: {_summarize_tensor(sample['batch'])}")
            print(f"[ptv3] grid_size: {sample['grid_size']}")
            print("[ptv3] running synthetic forward smoke test...", flush=True)
            with torch.inference_mode():
                output = model(sample)
            print(f"[ptv3] output type: {type(output)}")
            if isinstance(output, dict):
                print(f"[ptv3] output keys: {list(output.keys())[:20]}")
            elif isinstance(output, torch.Tensor):
                print(f"[ptv3] output shape: {tuple(output.shape)}")
        finally:
            for hook in hooks:
                hook.remove()

    return alignment


def main() -> int:
    """Run the command-line entry point."""
    faulthandler.enable()
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "backends": {},
    }

    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"device: {device}")
    print()

    if args.backend in {"uni3d", "both"}:
        try:
            report["backends"]["uni3d"] = _diagnose_uni3d(device, args.forward_smoke, args.uni3d_sample_count)
        except Exception as exc:
            report["backends"]["uni3d_error"] = f"{exc.__class__.__name__}: {exc}"
            print(f"[uni3d] ERROR: {exc.__class__.__name__}: {exc}")

    if args.backend in {"ptv3", "both"}:
        try:
            report["backends"]["ptv3"] = _diagnose_ptv3(device, args.forward_smoke, args.ptv3_sample_count)
        except Exception as exc:
            report["backends"]["ptv3_error"] = f"{exc.__class__.__name__}: {exc}"
            print(f"[ptv3] ERROR: {exc.__class__.__name__}: {exc}")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"diagnostic report written to: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
