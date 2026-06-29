#!/usr/bin/env python3
"""
Monitor de logs de entrenamiento en tiempo real.
Muestra el progreso actual de todos los modelos.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def tail_log(log_file: Path, lines: int = 20):
    """Lee las últimas N líneas de un archivo de log."""
    if not log_file.exists():
        return []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        return all_lines[-lines:] if len(all_lines) >= lines else all_lines


def get_latest_log(model_name: str):
    """Encuentra el log más reciente para un modelo."""
    log_dir = PROJECT_ROOT / "logs" / "screening"
    if not log_dir.exists():
        return None
    
    logs = list(log_dir.glob(f"{model_name}_*.log"))
    if not logs:
        return None
    
    return max(logs, key=lambda p: p.stat().st_mtime)


def show_progress():
    """Muestra el progreso de todos los modelos."""
    print("="*80)
    print(f"MONITOR DE ENTRENAMIENTO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Leer resultados CSV
    results_csv = PROJECT_ROOT / "models" / "screening" / "screening_results.csv"
    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)
        print(f"\nModelos completados: {len(df)}/12")
        print("\nRanking actual:")
        df_sorted = df.sort_values('best_dice', ascending=False)
        for idx, row in df_sorted.iterrows():
            print(f"  {row['name']:40s} | Dice: {row['best_dice']:.4f} | IoU: {row['best_iou']:.4f}")
    
    # Modelos pendientes
    all_models = [
        "DeepLabV3_ResNeXt50",
        "DeepLabV3Plus_ResNeXt50",
        "UNet_ResNeXt50",
        "FPN_ResNeXt50",
        "DeepLabV3Plus_ResNet101",
        "UNet_ResNet101",
        "FPN_ResNet101",
        "DeepLabV3Plus_EfficientNetB3",
        "UNet_EfficientNetB3",
        "FPN_EfficientNetB3",
        "UNet_SegFormer",
        "FPN_SegFormer",
    ]
    
    completed = set(df['name'].values) if results_csv.exists() else set()
    pending = [m for m in all_models if m not in completed]
    
    print(f"\nModelos pendientes: {len(pending)}")
    for model in pending:
        log_file = get_latest_log(model)
        if log_file:
            print(f"\n  [{model}] - Log: {log_file.name}")
            lines = tail_log(log_file, lines=5)
            for line in lines:
                print(f"    {line.rstrip()}")
        else:
            print(f"\n  [{model}] - Sin log aún")


def monitor_realtime(model_name: str, interval: int = 5):
    """Monitorea un modelo específico en tiempo real."""
    print(f"\n{'='*80}")
    print(f"MONITOREO EN TIEMPO REAL: {model_name}")
    print(f"Intervalo: {interval}s | Ctrl+C para detener")
    print(f"{'='*80}\n")
    
    log_file = get_latest_log(model_name)
    if not log_file:
        print(f"No se encontró log para {model_name}")
        return
    
    print(f"Log: {log_file}\n")
    
    last_pos = 0
    try:
        while True:
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    if new_lines:
                        for line in new_lines:
                            print(line.rstrip())
                        last_pos = f.tell()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitoreo detenido.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor de entrenamiento")
    parser.add_argument("--model", type=str, help="Monitorear modelo específico en tiempo real")
    parser.add_argument("--interval", type=int, default=5, help="Intervalo de actualización en segundos")
    args = parser.parse_args()
    
    if args.model:
        monitor_realtime(args.model, args.interval)
    else:
        show_progress()
