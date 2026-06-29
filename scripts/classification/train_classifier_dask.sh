#!/bin/bash
# ----------------------------------------------------------------- #
# Launcher Linux para entrenamiento distribuido del clasificador.
#
# Configura variables de entorno para rendimiento óptimo en CPU
# (OMP/MKL threads) y ejecuta con estrategia ddp nativa (fork).
#
# Uso:
#     bash scripts/train_classifier_dask_linux.sh
#     bash scripts/train_classifier_dask_linux.sh --epochs 30 --lr 5e-5
# ----------------------------------------------------------------- #
set -e

# Limitar threads de BLAS para evitar contención con workers Dask
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

# Directorio del proyecto (relativo a la ubicación del script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== Entrenamiento Distribuido (Linux) ==="
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "Directorio: $PROJECT_DIR"
echo ""

exec python scripts/train_classifier_dask.py "$@"
