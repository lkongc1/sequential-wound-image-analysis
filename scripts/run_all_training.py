#!/usr/bin/env python3
"""
Ejecuta todos los entrenamientos pendientes en secuencia.
Cada entrenamiento tiene su propio log.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("="*80)
    print("EJECUTANDO TODOS LOS ENTRENAMIENTOS PENDIENTES")
    print("="*80)
    
    # Scripts de entrenamiento
    training_scripts = [
        "train_deeplabv3plus_resnet101.py",
        "train_unet_resnet101.py",
        "train_fpn_resnet101.py",
        "train_deeplabv3plus_efficientnet.py",
        "train_unet_efficientnet.py",
        "train_fpn_efficientnet.py",
        "train_unet_segformer.py",
        "train_fpn_segformer.py",
    ]
    
    scripts_dir = PROJECT_ROOT / "scripts"
    
    for i, script_name in enumerate(training_scripts, 1):
        script_path = scripts_dir / script_name
        
        print(f"\n{'='*80}")
        print(f"[{i}/{len(training_scripts)}] Ejecutando: {script_name}")
        print(f"{'='*80}\n")
        
        if not script_path.exists():
            print(f"ERROR: Script no encontrado: {script_path}")
            continue
        
        # Ejecutar script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT)
        )
        
        if result.returncode != 0:
            print(f"\n[ERROR] {script_name} falló con código {result.returncode}")
            print("Continuando con el siguiente...\n")
        else:
            print(f"\n[OK] {script_name} completado exitosamente\n")
    
    print("\n" + "="*80)
    print("TODOS LOS ENTRENAMIENTOS COMPLETADOS")
    print("="*80)
    print("\nPara ver resultados finales:")
    print("  python scripts/monitor_training.py")


if __name__ == "__main__":
    main()
