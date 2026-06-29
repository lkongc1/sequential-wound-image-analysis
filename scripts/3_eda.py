#!/usr/bin/env python3
"""Analisis Exploratorio de Datos (EDA) — SOLO CONSOLA.

Genera un reporte completo del dataset de heridas: distribucion por fuente,
estadisticas de imagenes, caracteristicas de heridas, deteccion de outliers
y metricas de calidad. Ideal para incluir como ANEXO en documentacion
regulatoria o trabajo academico.

Uso:
    python scripts/eda.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# ================================================================== #
# CONFIGURACION
# ================================================================== #

DEFAULT_CSV = PROJECT_ROOT / "data" / "processed" / "dataset_final.csv"
SEP = "=" * 78
SEP2 = "-" * 78


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analisis Exploratorio de Datos (EDA) — Dataset de heridas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/3_eda.py
  python scripts/3_eda.py --csv data/processed/dataset_final.csv --output reports/eda.txt
        """.strip(),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Ruta al CSV del dataset (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Guardar reporte a archivo (default: solo consola)",
    )
    return parser


# ================================================================== #
# RENDERERS
# ================================================================== #

def header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def subheader(title: str) -> None:
    print(f"\n  {title}")
    print(f"  {SEP2}")


def bar(val: float, max_val: float = 100.0, width: int = 30) -> str:
    pct = min(val / max_val * width, width) if max_val > 0 else 0
    return "|" + "=" * int(pct) + "-" * (width - int(pct)) + "|"


def fmt_pct(n: float, total: int) -> str:
    return f"{n/total*100:.1f}%" if total > 0 else "N/A"


# ================================================================== #
# 1. DESCRIPCION GENERAL DEL DATASET
# ================================================================== #

def analisis_general(df: pd.DataFrame) -> None:
    header("1. DESCRIPCION GENERAL DEL DATASET")

    total = len(df)
    print(f"""
    Total de muestras:           {total:,}
    Imagenes de entrenamiento:   {len(df[df['split']=='train']):,}  ({fmt_pct(len(df[df['split']=='train']), total)})
    Imagenes de test:            {len(df[df['split']=='test']):,}   ({fmt_pct(len(df[df['split']=='test']), total)})
    """)

    # Por fuente
    subheader("Distribucion por fuente de datos")
    print(f"    {'Fuente':<15} {'Muestras':>10}  {'%':>8}  {'Train':>8}  {'Test':>8}")
    print(f"    {'-'*15} {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
    for source in sorted(df["source"].unique()):
        subset = df[df["source"] == source]
        n = len(subset)
        n_train = len(subset[subset["split"] == "train"])
        n_test = len(subset[subset["split"] == "test"])
        print(f"    {source:<15} {n:>10,}  {n/total*100:>7.1f}%  {n_train:>8,}  {n_test:>8,}")

    # Fuentes conocidas en segmentacion de heridas
    print(f"""
    Fuentes:
      - FUSC (Foot Ulcer Segmentation Challenge): imagenes de ulceras de pie diabetico
      - Medetec: base de datos clinica de heridas de diversa etiologia
      - WSNet (Wound Segmentation Network): dataset publico de heridas cronicas

    Split patient-aware (GroupShuffleSplit): todas las imagenes del mismo
    paciente van al mismo split (train o test). Esto previene DATA LEAKAGE
    y asegura que la evaluacion refleje rendimiento en pacientes NO VISTOS.
    """)


# ================================================================== #
# 2. ESTADISTICAS DE IMAGENES
# ================================================================== #

def analisis_imagenes(df: pd.DataFrame) -> None:
    header("2. ESTADISTICAS DE IMAGENES")

    cols = ["brightness_mean", "brightness_std", "contrast_rms"]
    names = ["Brillo medio", "Desviacion estandar brillo", "Contraste RMS"]
    units = ["[0-255]", "[0-255]", "RMS"]

    subheader("Metricas globales de calidad de imagen")
    print(f"    {'Metrica':<30} {'Media':>10} {'Mediana':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
    print(f"    {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for col, name, unit in zip(cols, names, units):
        vals = df[col].dropna()
        print(f"    {name + ' ' + unit:<30} {vals.mean():>10.2f} {vals.median():>10.2f} "
              f"{vals.min():>10.2f} {vals.max():>10.2f} {vals.std():>10.2f}")

    # Distribucion de brillo por fuente
    subheader("Brillo medio por fuente")
    print(f"    {'Fuente':<15} {'Media':>10} {'Mediana':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
    print(f"    {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for source in sorted(df["source"].unique()):
        vals = df[df["source"] == source]["brightness_mean"]
        print(f"    {source:<15} {vals.mean():>10.2f} {vals.median():>10.2f} "
              f"{vals.min():>10.2f} {vals.max():>10.2f} {vals.std():>10.2f}")

    # Imagenes con problemas de brillo
    dark = df[df["brightness_mean"] < 30]
    bright = df[df["brightness_mean"] > 230]
    print(f"""
    Control de calidad de iluminacion:
      - Imagenes muy oscuras (brillo < 30):  {len(dark):>4}  ({fmt_pct(len(dark), len(df))})
      - Imagenes muy claras (brillo > 230):  {len(bright):>4}  ({fmt_pct(len(bright), len(df))})
      - Imagenes en rango aceptable:         {len(df)-len(dark)-len(bright):>4}  ({fmt_pct(len(df)-len(dark)-len(bright), len(df))})
    """)


# ================================================================== #
# 3. CARACTERISTICAS DE HERIDAS
# ================================================================== #

def analisis_heridas(df: pd.DataFrame) -> None:
    header("3. CARACTERISTICAS DE HERIDAS")

    # Wound percentage
    wp = df["wound_percentage"].dropna()
    subheader("Porcentaje de herida en la imagen (wound_percentage)")

    bins = [0, 1, 5, 10, 25, 50, 100]
    labels = ["<1%", "1-5%", "5-10%", "10-25%", "25-50%", ">50%"]
    df["wp_bin"] = pd.cut(df["wound_percentage"], bins=bins, labels=labels, include_lowest=True)
    wp_dist = df["wp_bin"].value_counts().sort_index()

    print(f"    {'Rango':<12} {'Muestras':>10}  {'%':>8}  {'Distribucion'}")
    print(f"    {'-'*12} {'-'*10}  {'-'*8}  {'-'*32}")
    for label, count in wp_dist.items():
        print(f"    {str(label):<12} {count:>10,}  {count/len(df)*100:>7.1f}%  {bar(count, wp_dist.max())}")

    print(f"""
    Estadisticas de wound_percentage:
      Media:     {wp.mean():.2f}%     Mediana:   {wp.median():.2f}%
      Minimo:    {wp.min():.2f}%      Maximo:    {wp.max():.2f}%
      Desv std:  {wp.std():.2f}%

    Interpretacion clinica:
      - La mayoria de heridas ocupan <5% de la imagen -> desbalance de clases severo
      - Esto justifica usar F2-score (penaliza FN) y TverskyLoss en entrenamiento
      - Los pixeles de herida son ~{wp.mean():.1f}% del total -> clase minoritaria extrema
    """)

    # Mask area
    ma = df["mask_area_pixels"]
    subheader("Area de herida en pixeles")
    print(f"      Media:     {ma.mean():>10,.0f} px     Mediana:   {ma.median():>10,.0f} px")
    print(f"      Minimo:    {ma.min():>10,.0f} px     Maximo:    {ma.max():>10,.0f} px")
    print(f"      Desv std:  {ma.std():>10,.0f} px")

    # Empty masks
    empties = df[df["is_empty"] == True]
    print(f"\n    Mascaras vacias (sin herida): {len(empties)}  ({fmt_pct(len(empties), len(df))})")

    # Por fuente
    subheader("Wound percentage por fuente")
    print(f"    {'Fuente':<15} {'Media':>10} {'Mediana':>10} {'Min':>10} {'Max':>10}")
    print(f"    {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for source in sorted(df["source"].unique()):
        vals = df[df["source"] == source]["wound_percentage"]
        print(f"    {source:<15} {vals.mean():>10.2f}% {vals.median():>10.2f}% "
              f"{vals.min():>10.2f}% {vals.max():>10.2f}%")

    # Edge density
    ed = df["mask_edge_density"]
    subheader("Densidad de bordes de herida (complejidad morfologica)")
    print(f"      Media:     {ed.mean():.6f}     Mediana:   {ed.median():.6f}")
    print(f"      Minimo:    {ed.min():.6f}     Maximo:    {ed.max():.6f}")
    print(f"\n    Mayor densidad = bordes mas irregulares (heridas complejas)")
    print(f"    Menor densidad = bordes mas suaves (heridas regulares/redondeadas)")


# ================================================================== #
# 4. DETECCION DE OUTLIERS
# ================================================================== #

def analisis_outliers(df: pd.DataFrame) -> None:
    header("4. DETECCION DE OUTLIERS")

    n_outliers = len(df[df["is_outlier"] == True])
    n_clean = len(df[df["is_outlier"] == False])

    print(f"""
    Total de muestras:      {len(df):,}
    Muestras normales:      {n_clean:,}  ({fmt_pct(n_clean, len(df))})
    Outliers detectados:    {n_outliers:,}   ({fmt_pct(n_outliers, len(df))})
    """)

    if n_outliers > 0:
        subheader("Outliers por razon")
        reasons = df[df["is_outlier"] == True]["outlier_reason"].value_counts()
        for reason, count in reasons.items():
            print(f"    {str(reason):<50} {count:>5}  ({fmt_pct(count, n_outliers)})")

        subheader("Outliers por fuente")
        for source in sorted(df["source"].unique()):
            src_out = len(df[(df["source"] == source) & (df["is_outlier"] == True)])
            src_total = len(df[df["source"] == source])
            print(f"    {source:<15} {src_out:>4}/{src_total:<6}  ({fmt_pct(src_out, src_total)})")

        subheader("Outliers por split")
        for split in ["train", "test"]:
            sp_out = len(df[(df["split"] == split) & (df["is_outlier"] == True)])
            sp_total = len(df[df["split"] == split])
            print(f"    {split:<15} {sp_out:>4}/{sp_total:<6}  ({fmt_pct(sp_out, sp_total)})")

    # Wound percentage outliers (IQR method)
    subheader("Outliers de wound_percentage (metodo IQR)")
    q1 = df["wound_percentage"].quantile(0.25)
    q3 = df["wound_percentage"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    wp_outliers = df[(df["wound_percentage"] < lower) | (df["wound_percentage"] > upper)]

    print(f"    Q1={q1:.2f}%  Q3={q3:.2f}%  IQR={iqr:.2f}%")
    print(f"    Limite inferior: {max(0, lower):.2f}%")
    print(f"    Limite superior: {upper:.2f}%")
    print(f"    Outliers por IQR: {len(wp_outliers)}  ({fmt_pct(len(wp_outliers), len(df))})")


# ================================================================== #
# 5. CONTROL DE CALIDAD
# ================================================================== #

def analisis_calidad(df: pd.DataFrame) -> None:
    header("5. CONTROL DE CALIDAD DEL DATASET")

    # Completitud
    total_celdas = len(df) * len(df.columns)
    nulos = df.isnull().sum().sum()
    completitud = (1 - nulos / total_celdas) * 100

    print(f"""
    Completitud de datos: {completitud:.2f}%
      - Celdas totales:   {total_celdas:,}
      - Valores nulos:    {nulos:,}
    """)

    # Valores nulos por columna
    subheader("Valores nulos por columna")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        for col, n in nulls.items():
            print(f"    {col:<30} {n:>6}  ({fmt_pct(n, len(df))})")
    else:
        print(f"    Sin valores nulos en columnas principales.")

    # Balance de clases
    subheader("Balance wound vs background (a nivel de pixel)")
    mean_wp = df["wound_percentage"].mean()
    print(f"""
    Porcentaje medio de herida:  {mean_wp:.2f}%
    Porcentaje medio de fondo:   {100-mean_wp:.2f}%
    Ratio herida:fondo:          1:{100/mean_wp:.0f}

    DESBALANCE SEVERO: Solo ~{mean_wp:.1f}% de los pixeles son herida.
    Esto justifica:
      - Uso de F2-score y TverskyLoss (penalizan mas los falsos negativos)
      - Metricas como Dice/IoU en vez de accuracy
      - Data augmentation dirigido a heridas pequenas
    """)

    # Duplicados
    dups = df["filename"].duplicated().sum()
    print(f"    Archivos duplicados: {dups}")

    # Mascaras vacias por fuente
    subheader("Mascaras vacias (sin herida detectable)")
    for source in sorted(df["source"].unique()):
        src_empty = len(df[(df["source"] == source) & (df["is_empty"] == True)])
        src_total = len(df[df["source"] == source])
        print(f"    {source:<15} {src_empty:>4}/{src_total:<6}  ({fmt_pct(src_empty, src_total)})")


# ================================================================== #
# 6. RESUMEN PARA DOCUMENTACION
# ================================================================== #

def resumen_final(df: pd.DataFrame) -> None:
    header("6. RESUMEN EJECUTIVO — ANEXO EDA")

    total = len(df)
    n_train = len(df[df["split"] == "train"])
    n_test = len(df[df["split"] == "test"])
    n_outliers = len(df[df["is_outlier"] == True])
    mean_wp = df["wound_percentage"].mean()
    mean_bright = df["brightness_mean"].mean()

    print(f"""
    +{'-'*74}+
    | {'DATASET DE SEGMENTACION DE HERIDAS — FICHA TECNICA':<72} |
    +{'-'*74}+
    | {'Muestras totales:':<30} {total:>8,} imagenes con mascara binaria       |
    | {'Entrenamiento:':<30} {n_train:>8,}  ({fmt_pct(n_train, total)})                       |
    | {'Test (held-out):':<30} {n_test:>8,}   ({fmt_pct(n_test, total)})                       |
    | {'Fuentes:':<30} {', '.join(sorted(df['source'].unique())):<40} |
    | {'':<30} {'':<40} |
    | {'Herida media (%):':<30} {mean_wp:>8.2f}%  (desbalance severo, ~1:{100/mean_wp:.0f})  |
    | {'Brillo medio:':<30} {mean_bright:>8.1f}  [0-255]                         |
    | {'Outliers detectados:':<30} {n_outliers:>8,}  ({fmt_pct(n_outliers, total)})                       |
    | {'':<30} {'':<40} |
    | {'Split:':<30} {'Patient-aware (GroupShuffleSplit)':<40} |
    | {'Uso:':<30} {'Entrenamiento U-Net variantes + eval':<40} |
    +{'-'*74}+

    NOTAS METODOLOGICAS:
      - El split es patient-aware: todas las imagenes de un mismo paciente
        estan en el mismo split (train o test). Esto previene fuga de datos.
      - Las mascaras son binarias (0=fondo, 1=herida).
      - Las fuentes tienen diferentes protocolos de adquisicion (iluminacion,
        resolucion, tipo de herida) -> el modelo debe generalizar entre fuentes.
      - El desbalance wound/background (~{mean_wp:.1f}%/{(100-mean_wp):.1f}%) requiere
        metricas especializadas (Dice, F2) y funciones de perdida ponderadas.
    """)


# ================================================================== #
# MAIN
# ================================================================== #

def main() -> None:
    args = build_parser().parse_args()
    csv_path: Path = args.csv

    # Validar existencia del CSV antes de cargar nada
    if not csv_path.exists():
        print(f"\n[ERROR] No se encontro el archivo: {csv_path}", file=sys.stderr)
        print(f"        Asegurate de que el dataset fue generado (e.g., ejecuta 2_preprocess.py primero).", file=sys.stderr)
        print(f"        Puedes especificar otra ruta con: python scripts/3_eda.py --csv <ruta>", file=sys.stderr)
        sys.exit(1)

    # Redirigir salida a archivo si se pidio
    out_file = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(args.output, "w", encoding="utf-8")
        sys.stdout = out_file

    try:
        print()
        print(SEP)
        print("  ANALISIS EXPLORATORIO DE DATOS (EDA)")
        print("  Dataset: Wound Segmentation — FUSC + Medetec + WSNet")
        print("  Proposito: ANEXO — Documentacion regulatoria / Trabajo academico")
        print(SEP)

        # Cargar datos
        df = pd.read_csv(csv_path)
        print(f"\n  Archivo: {csv_path}")
        print(f"  Registros: {len(df):,}  |  Columnas: {len(df.columns)}")

        # Ejecutar todos los analisis
        analisis_general(df)
        analisis_imagenes(df)
        analisis_heridas(df)
        analisis_outliers(df)
        analisis_calidad(df)
        resumen_final(df)

        print(f"\n{SEP}")
        print(f"  FIN DEL REPORTE EDA")
        print(f"  Dataset: {len(df):,} muestras  |  {len(df['source'].unique())} fuentes")
        print(f"  Listo para incluir como ANEXO en documentacion.")
        print(SEP)
        print()
    finally:
        if out_file:
            out_file.close()
            sys.stdout = sys.__stdout__
            print(f"\n[OK] Reporte guardado en: {args.output}")


if __name__ == "__main__":
    main()
