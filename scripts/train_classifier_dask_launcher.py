#!/usr/bin/env python3
"""Launcher cross-platform para entrenamiento distribuido con Dask.

Detecta automáticamente el sistema operativo y configura:
  - Linux:   GPU (CUDA) + workers Dask para entrenamiento real.
  - Windows: CPU-only + workers reducidos para prueba académica.

Reemplaza a ``train_classifier_dask_linux.sh``. Compatible con conda y venv.

Uso:
    python scripts/train_classifier_dask_launcher.py
    python scripts/train_classifier_dask_launcher.py --epochs 30 --lr 5e-5
    python scripts/train_classifier_dask_launcher.py --dask-workers 4 --batch-size 16
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_SCRIPT = _PROJECT_ROOT / "scripts" / "train_classifier_dask.py"

_HEADER = """
=================================================================
   Clasificador de Heridas — Entrenamiento Distribuido
   Dask + PyTorch Lightning
=================================================================
"""


def _detect_platform() -> dict[str, str | int | bool]:
    """Detecta SO y devuelve configuración específica.

    Returns:
        Dict con claves: name, is_linux, is_windows, cuda_expected,
        default_workers, threads_per_worker, extra_env.
    """
    system = platform.system()
    is_linux = system == "Linux"
    is_windows = system == "Windows"

    if is_linux:
        cpu_count = os.cpu_count() or 8
        return {
            "name": f"Linux ({platform.release()})",
            "is_linux": True,
            "is_windows": False,
            "cuda_expected": True,
            "mode": "GPU (entrenamiento real)",
            "default_workers": max(1, cpu_count - 1),
            "threads_per_worker": 1,
            "extra_env": {
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "NUMEXPR_NUM_THREADS": "4",
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
        }
    elif is_windows:
        cpu_count = os.cpu_count() or 4
        return {
            "name": f"Windows ({platform.release()})",
            "is_linux": False,
            "is_windows": True,
            "cuda_expected": False,
            "mode": "CPU-only (prueba académica)",
            "default_workers": max(1, (cpu_count or 4) // 2),
            "threads_per_worker": 1,
            "extra_env": {
                "OMP_NUM_THREADS": "2",
                "CUDA_VISIBLE_DEVICES": "",
            },
        }
    else:
        cpu_count = os.cpu_count() or 4
        return {
            "name": f"{system} ({platform.release()})",
            "is_linux": False,
            "is_windows": False,
            "cuda_expected": False,
            "mode": "CPU-only (SO no soportado oficialmente)",
            "default_workers": max(1, (cpu_count or 4) // 2),
            "threads_per_worker": 1,
            "extra_env": {},
        }


def _print_banner(cfg: dict) -> None:
    """Imprime banner informativo con la configuración detectada."""
    print(_HEADER)
    print(f"  Sistema     : {cfg['name']}")
    print(f"  Modo        : {cfg['mode']}")
    print(f"  Dask workers: {cfg['default_workers']} (auto-detectado)")
    print(f"  Hilos/worker: {cfg['threads_per_worker']}")
    if cfg["extra_env"]:
        print("  Env vars    :")
        for k, v in cfg["extra_env"].items():
            print(f"    {k}={v}")
    print()


def _resolve_dask_workers(argv: list[str], cfg: dict) -> int:
    """Extrae --dask-workers de argv si existe, sino usa default."""
    for i, arg in enumerate(argv):
        if arg == "--dask-workers" and i + 1 < len(argv):
            return int(argv[i + 1])
        if arg.startswith("--dask-workers="):
            return int(arg.split("=", 1)[1])
    return cfg["default_workers"]


def main() -> None:
    cfg = _detect_platform()
    _print_banner(cfg)

    # --- Validación cruzada de CUDA en Windows ---
    if cfg["is_windows"]:
        try:
            import torch
            if torch.cuda.is_available():
                print(
                    "[!] ADVERTENCIA: CUDA detectada en Windows. "
                    "El entrenamiento real debe ejecutarse en Linux.\n"
                    "   Este launcher forza CPU. Si necesitás GPU en Windows, "
                    "editá train_classifier_dask.py directamente.\n"
                )
        except ImportError:
            pass  # torch no instalado — sin GPU obviamente

    # --- Aplicar environment variables ---
    for key, val in cfg["extra_env"].items():
        if val:  # solo setear si no es string vacío
            os.environ[key] = val

    # --- Armar comando ---
    # Pasamos --dask-workers con el default del SO si el usuario no lo setea
    user_argv = sys.argv[1:]
    has_dask_workers = any(
        arg.startswith("--dask-workers") for arg in user_argv
    )
    dask_args = []
    if not has_dask_workers:
        dask_args = [
            "--dask-workers", str(cfg["default_workers"]),
            "--dask-threads", str(cfg["threads_per_worker"]),
        ]

    cmd = [
        sys.executable,
        str(_TRAIN_SCRIPT),
        *user_argv,
        *dask_args,
    ]

    # --- Ejecutar ---
    print(f">>> Ejecutando: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
