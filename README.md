# Sequential Wound Image Analysis

Sistema de deteccion, segmentacion y clasificacion de heridas mediante tecnicas de vision computacional y aprendizaje profundo.

Tesis: *"Deteccion y clasificacion de los tipos de heridas mediante tecnicas de vision computacional"*

## Pipeline

```
                         SEQUENTIAL WOUND IMAGE ANALYSIS
                         ==============================

  ┌──────────┐      ┌─────────────────┐      ┌──────────────────┐      ┌──────────────┐
  │          │      │                 │      │                  │      │              │
  │  INPUT   │      │  SEGMENTACION   │      │  CLASIFICACION   │      │  PREDICCION  │
  │          │      │                 │      │                  │      │              │
  │  Imagen  │ ───> │  U-Net ResNet50 │ ───> │  EfficientNet-B3 │ ───> │  Tipo herida │
  │  clinica │      │  FPN / DeepLab  │      │  7 categorias:   │      │  Area (px)   │
  │  (RGB)   │      │  YOLO11 / SAM2  │      │  raspon          │      │  Confianza   │
  │          │      │                 │      │  hematoma        │      │  Overlay     │
  │          │      │  Salida: mascara │      │  quemadura       │      │  azul        │
  │          │      │  binaria (0/1)   │      │  corte           │      │              │
  │          │      │                 │      │  laceracion      │      │              │
  │          │      │                 │      │  puncion         │      │              │
  │          │      │                 │      │  piel sana       │      │              │
  └──────────┘      └─────────────────┘      └──────────────────┘      └──────────────┘

  Modelos: 12 arquitecturas evaluadas  |  Mejor: UNet+SegFormer Dice=0.8893  |  150 epocas finales
```


## Caracteristicas

- Segmentacion semantica con 12 arquitecturas (U-Net, FPN, DeepLabV3+ combinadas con ResNet, ResNeXt, EfficientNet, SegFormer)
- Segmentacion de instancias: Watershed, SAM2, YOLO11-seg
- Clasificador de 7 tipos de herida: raspon, hematoma, quemadura, corte, laceracion, puncion, piel sana
- Inferencia hibrida YOLO + U-Net: deteccion de region seguida de segmentacion fina
- API REST con FastAPI para diagnostico
- Entrenamiento distribuido con Dask
- Pipeline de datos reproducible: descarga, construccion, EDA, integracion de datasets externos

## Resultados

### Screening de arquitecturas (20 epocas)

| Arquitectura | Encoder | Dice | IoU | Parametros | Tiempo (min) |
|-------------|---------|------|-----|------------|-------------|
| UNet + SegFormer | mit_b2 | 0.8893 | 0.8061 | 27.5M | 34.3 |
| FPN + SegFormer | mit_b2 | 0.8816 | 0.7934 | 26.1M | 28.7 |
| UNet + EfficientNet-B3 | efficientnet-b3 | 0.8840 | 0.7981 | 13.2M | 34.5 |
| FPN + EfficientNet-B3 | efficientnet-b3 | 0.8749 | 0.7827 | 12.5M | 26.6 |
| DeepLabV3Plus + EfficientNet-B3 | efficientnet-b3 | 0.8699 | 0.7757 | 11.7M | 34.0 |
| FPN + ResNeXt50 | resnext50_32x4d | 0.8796 | 0.7907 | 25.6M | 32.1 |
| DeepLabV3 + ResNeXt50 | resnext50_32x4d | 0.8772 | 0.7867 | 39.1M | 133.7 |
| UNet + ResNeXt50 | resnext50_32x4d | 0.8766 | 0.7864 | 32.0M | 39.3 |
| DeepLabV3Plus + ResNeXt50 | resnext50_32x4d | 0.8754 | 0.7845 | 26.1M | 34.5 |
| DeepLabV3Plus + ResNet101 | resnet101 | 0.8705 | 0.7767 | 45.7M | 38.5 |
| FPN + ResNet101 | resnet101 | 0.8675 | 0.7721 | 45.1M | 35.5 |
| UNet + ResNet101 | resnet101 | 0.8650 | 0.7692 | 51.5M | 41.9 |

### Modelo final (150 epocas)

| Modelo | Encoder | Dice | F1 | Accuracy | Precision | Recall |
|--------|---------|------|-----|----------|-----------|--------|
| U-Net | ResNet50 (ImageNet) | 0.8927 | 0.8927 | 0.9927 | 0.9137 | 0.8726 |

## Estructura del proyecto

```
src/
  core/           Clases base, constantes, excepciones
  config.py       Configuracion centralizada (dataclasses + YAML)
  data/           Auditoria, EDA, preprocesamiento, calidad, fuentes, validadores
  datasets/       Datasets de heridas y clasificacion
  models/         U-Net, Attention U-Net, Nested U-Net, clasificador, fabrica
  training/       Trainer con DI, EarlyStopping, CheckpointManager
  evaluation/     Analisis comparativo, reportes, visualizacion
  losses/         Dice, Focal, Tversky
  metrics/        Segmentacion, clinicas, matriz de confusion
  inference/      Predictor, ensemble, SAM2, Watershed
  api/            FastAPI: salud, diagnostico, pacientes
  utils/          Logger estructurado, deteccion de GPU

scripts/
  pipeline/       download_dataset, build_dataset, explore_data,
                  prepare_yolo, add_negatives, integrate_datasets
  training/       train_unet, screen_architectures, train_single, train_yolo
  classification/ train_classifier, train_classifier_dask, download_data, generate_masks
  evaluation/     evaluate_model
  inference/      predict, predict_yolo, predict_hybrid, demo_watershed,
                  compare_models, predict_presentation
  utils/          monitor

config/           YAMLs de configuracion por entorno
tests/            Unitarios, integracion, clinicos
data/             Datasets procesados (rutas relativas)
models/           Pesos preentrenados (Git LFS)
```

## Requisitos

- Python 3.10+
- CUDA 12.1 (recomendado para entrenamiento)
- Git LFS (para pesos de modelos)

```bash
pip install -r requirements.txt
```

## Instalacion

```bash
git clone https://github.com/lkongc1/sequential-wound-image-analysis.git
cd sequential-wound-image-analysis
git lfs pull
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

Copiar `.env.example` a `.env` y configurar las variables de entorno necesarias.

Git LFS es obligatorio: los pesos de los modelos preentrenados se almacenan con LFS. Sin `git lfs pull`, los archivos `.pth` seran punteros vacios.

## Flujo de trabajo

Este repositorio usa un flujo simple con dos ramas:

```
main       <- codigo estable, listo para produccion
develop    <- rama de integracion para nuevos cambios
```

Para contribuir:
1. Crear rama desde `develop`: `git checkout -b feat/mi-cambio develop`
2. Hacer cambios y commits con mensajes descriptivos
3. Abrir PR hacia `develop`
4. Una vez mergeado y validado, se promueve a `main`

## Inicio rapido

### Inferencia con modelo preentrenado

```bash
python scripts/inference/predict.py imagen.png
python scripts/inference/predict.py imagen.png --umbral 0.6 --limpiar 3
```

### Segmentacion de instancias (Watershed)

```bash
python scripts/inference/demo_watershed.py imagen.png --threshold 0.4
```

### Inferencia hibrida YOLO + U-Net

```bash
python scripts/inference/predict_hybrid.py imagen.png
```

### Prediccion para presentaciones (overlay azul)

```bash
python scripts/inference/predict_presentation.py imagen.jpg
python scripts/inference/predict_presentation.py imagen.jpg --title "Resultado"
```

Genera una imagen con overlay azul, bounding box, panel de metricas y diagrama del pipeline. Ideal para presentaciones y documentacion.

### API REST

```bash
uvicorn src.api.main:app --reload
```

Endpoints disponibles en `http://localhost:8000/docs`.

## Pipeline de datos

Ejecutar en orden:

```bash
python scripts/pipeline/download_dataset.py      # Descarga desde Kaggle
python scripts/pipeline/build_dataset.py          # Construye dataset paired
python scripts/pipeline/explore_data.py           # Analisis exploratorio
python scripts/pipeline/add_negatives.py          # Agrega negativos
python scripts/pipeline/prepare_yolo.py           # Prepara formato YOLO
python scripts/pipeline/integrate_datasets.py     # Integra datasets externos
```

## Entrenamiento

### Screening de arquitecturas

```bash
python scripts/training/screen_architectures.py
```

Entrena 12 combinaciones (arquitectura + encoder) con 20 epocas cada una para identificar los mejores candidatos.

### Entrenamiento individual

```bash
python scripts/training/train_single.py
python scripts/training/train_unet.py --encoder resnet50
python scripts/training/train_unet.py --encoder resnet18
python scripts/training/train_yolo.py
```

### Clasificador de heridas

```bash
python scripts/classification/download_data.py       # Descarga datos
python scripts/classification/generate_masks.py      # Genera pseudo-mascaras
python scripts/classification/train_classifier.py    # Entrenamiento single
python scripts/classification/train_classifier_dask.py  # Entrenamiento distribuido
```

## Evaluacion

```bash
python scripts/evaluation/evaluate_model.py
```

## Modelos disponibles

| Modelo | Tipo | Parametros | Uso |
|--------|------|------------|-----|
| UNet + SegFormer | Segmentacion | 27.5M | Mejor Dice en screening |
| UNet + EfficientNet-B3 | Segmentacion | 13.2M | Mejor balance tamano/rendimiento |
| U-Net ResNet50 | Segmentacion | 32.6M | Modelo final (150 epocas) |
| U-Net ResNet18 | Segmentacion | 31.1M | Heridas finas/lineales |
| EfficientNet-B3 | Clasificacion | 12.2M | 7 tipos de herida |
| YOLO11-seg | Instancias | 2.7M | Deteccion de instancias |

12 checkpoints de screening en `models/screening/`. Pesos trackeados con Git LFS.

## Tests

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
```

## Licencia

MIT License. Ver archivo `LICENSE`.
