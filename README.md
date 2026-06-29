# Sequential Wound Image Analysis

Sistema de deteccion, segmentacion y clasificacion de heridas mediante tecnicas de vision computacional y aprendizaje profundo.

Tesis: *"Deteccion y clasificacion de los tipos de heridas mediante tecnicas de vision computacional"*

## Caracteristicas

- Segmentacion semantica con 12 arquitecturas (U-Net, FPN, DeepLabV3+ combinadas con ResNet, ResNeXt, EfficientNet, SegFormer)
- Segmentacion de instancias: Watershed, SAM2, YOLO11-seg
- Clasificador de 7 tipos de herida: raspon, hematoma, quemadura, corte, laceracion, puncion, piel sana
- Inferencia hibrida YOLO + U-Net: deteccion de region seguida de segmentacion fina
- API REST con FastAPI para diagnostico
- Entrenamiento distribuido con Dask
- Pipeline de datos reproducible: descarga, construccion, EDA, integracion de datasets externos

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
  pipeline/       Descarga, construccion, EDA, preparacion YOLO, integracion
  training/       U-Net, screening 12 arquitecturas, train individual, YOLO
  classification/ Clasificador single y Dask, descarga de datos, pseudo-mascaras
  evaluation/     Evaluacion del modelo final
  inference/      Prediccion, YOLO+U-Net hibrido, Watershed, comparativa
  utils/          Monitoreo de entrenamiento

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
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

Copiar `.env.example` a `.env` y configurar las variables de entorno necesarias.

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

## Model Zoo

| Modelo | Arquitectura | Encoder | Uso |
|--------|-------------|---------|-----|
| U-Net final | U-Net | ResNet50 | Segmentacion principal |
| U-Net R18 | U-Net | ResNet18 | Heridas finas/lineales |
| Clasificador | EfficientNet-B3 | - | 7 tipos de herida |
| YOLO11-seg | YOLO11 | - | Instancias |

12 modelos de screening disponibles en `models/screening/`.

## Tests

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
```

## Licencia

MIT License. Ver archivo `LICENSE`.
