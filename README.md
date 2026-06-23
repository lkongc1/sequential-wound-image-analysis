# Sequential Wound Image Analysis

Sistema de inteligencia artificial médica para detección, segmentación y clasificación de heridas mediante técnicas de visión computacional.

Tesis: *"Detección y clasificación de los tipos de heridas mediante técnicas de visión computacional"*

[![CI](https://github.com/lkongc1/sequential-wound-image-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/lkongc1/sequential-wound-image-analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

---

## Tabla de Contenidos

- [Características](#características)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos e Instalación](#requisitos-e-instalación)
- [Inicio Rápido](#inicio-rápido)
- [Pipeline de Datos](#pipeline-de-datos)
- [Entrenamiento](#entrenamiento)
- [Evaluación](#evaluación)
- [Inferencia](#inferencia)
- [Model Zoo](#model-zoo)
- [API](#api)
- [Tests](#tests)
- [Configuración](#configuración)
- [Documentación](#documentación)
- [Validación Clínica](#validación-clínica)
- [Guía Completa de Comandos](#guía-completa-de-comandos)
- [Licencia](#licencia)

---

## Características

### Segmentación Semántica (máscara binaria)
- **12 arquitecturas evaluadas**: FPN, U-Net, DeepLabV3, DeepLabV3+ con encoders ResNet-101, ResNeXt-50, EfficientNet-B3 y SegFormer
- **Modelo principal**: FPN + EfficientNet-B3 (mejor Dice en screening)
- Post-procesamiento: limpieza morfológica, relleno de huecos, filtrado por área
- Test-Time Augmentation (TTA) para mayor precisión

### Segmentación de Instancias (heridas individuales)
- **Watershed**: rápido, para casos simples (basado en distance transform)
- **SAM2**: Segment Anything Model 2.1 (Hiera-L, B+, S) — separación precisa de heridas pegadas
- **YOLO11-seg**: segmentación de instancias nativa con detección

### Clasificación de Tipos de Herida
- **7 clases**: raspón, hematoma, quemadura, corte, laceración, punción, piel sana
- Arquitectura: EfficientNet-B3 con entrada de 4 canales (RGB + máscara)
- Técnicas avanzadas: MixUp, CutMix, label smoothing, RandAugment
- Entrenamiento distribuido con Dask para datasets grandes

### Inferencia Híbrida
- Pipeline YOLO + U-Net: YOLO detecta regiones de interés, U-Net segmenta dentro de cada ROI

### API REST
- FastAPI con endpoints de diagnóstico, pacientes y health check
- Schemas Pydantic para validación de entrada/salida

---

## Estructura del Proyecto

```
sequential-wound-image-analysis/
│
├── src/                          # Código fuente del paquete principal
│   ├── api/                      # FastAPI — endpoints REST
│   │   ├── main.py               # Entry point de la aplicación
│   │   ├── middleware.py          # Middleware (CORS, logging, errores)
│   │   ├── schemas.py            # Modelos Pydantic
│   │   └── routes/
│   │       ├── diagnosis.py      # POST /diagnosis — segmentar y clasificar
│   │       ├── health.py         # GET /health — health check
│   │       └── patients.py       # CRUD de pacientes y estudios
│   │
│   ├── core/                     # Clases base y constantes
│   │   ├── base.py               # Clases abstractas
│   │   ├── constants.py          # Constantes del proyecto
│   │   └── exceptions.py         # Excepciones personalizadas
│   │
│   ├── config.py                 # Dataclasses de configuración centralizada
│   │
│   ├── data/                     # Pipeline de datos completo
│   │   ├── sources/              # Fuentes de datos (Kaggle, Roboflow)
│   │   ├── downloaders/          # Descarga de datasets
│   │   ├── quality/              # Control de calidad de imágenes
│   │   ├── preprocessing/        # Normalización y preprocesado
│   │   ├── transforms/           # Augmentaciones (Albumentations)
│   │   ├── validators/           # Validación de integridad de datos
│   │   ├── extractors/           # Extracción de features
│   │   ├── eda/                  # Análisis exploratorio
│   │   ├── audit/                # Auditoría de datasets
│   │   └── review/               # Revisión y reporte de outliers
│   │
│   ├── datasets/                 # PyTorch Datasets
│   │   ├── wound_dataset.py      # Dataset de segmentación
│   │   ├── classification_dataset.py  # Dataset de clasificación
│   │   └── split_strategy.py     # Estrategias de split train/val/test
│   │
│   ├── models/                   # Arquitecturas de modelos
│   │   ├── factory.py            # Factory pattern (OCP) con MODEL_REGISTRY
│   │   ├── unet.py               # U-Net (lightweight, 8GB VRAM)
│   │   ├── attention_unet.py     # U-Net con attention gates
│   │   ├── nested_unet.py        # U-Net++ (nested skip connections)
│   │   └── classifier.py         # Clasificador EfficientNet-B3 (timm)
│   │
│   ├── training/                 # Loop de entrenamiento
│   │   ├── trainer.py            # Trainer principal con mixed precision
│   │   ├── checkpoint_manager.py # Gestión de checkpoints
│   │   └── early_stopping.py     # Early stopping configurable
│   │
│   ├── losses/                   # Funciones de pérdida
│   │   ├── dice_loss.py          # Dice Loss
│   │   ├── bce_dice_loss.py      # BCE + Dice combinada
│   │   ├── tversky_loss.py       # Tversky (tradeoff FN/FP configurable)
│   │   └── focal_loss.py         # Focal Loss para desbalance severo
│   │
│   ├── metrics/                  # Métricas de evaluación
│   │   ├── segmentation.py       # Dice, IoU, Sensitivity, Specificity
│   │   ├── clinical_metrics.py   # Área de herida, cambio de coloración
│   │   ├── longitudinal.py       # Progresión de cicatrización
│   │   ├── confusion.py          # Matriz de confusión
│   │   └── reporting.py          # Generación de reportes clínicos
│   │
│   ├── inference/                # Motor de inferencia
│   │   ├── predictor.py          # Wrapper de inferencia unificado
│   │   ├── postprocessing.py     # Limpieza de máscaras y morfología
│   │   ├── ensemble.py           # Ensemble de múltiples modelos
│   │   └── sam2_seg.py           # Integración con SAM2
│   │
│   └── utils/                    # Utilidades
│       ├── gpu.py                # Detección y configuración de GPU
│       └── logging.py            # Configuración de logging
│
├── scripts/                      # Scripts ejecutables
│   ├── 1_download_dataset.py     # Descarga del dataset base
│   ├── 2_build_dataset.py        # Construcción del dataset unificado
│   ├── 3_eda.py                  # Análisis exploratorio
│   ├── 4_train_models.py         # Entrenamiento batch
│   ├── 5_evaluate.py             # Evaluación de todos los modelos
│   ├── 6_train_unet_final.py     # U-Net ResNet50 (entrenamiento largo)
│   ├── 6b_train_unet_r18.py      # U-Net ResNet18 (rápido)
│   ├── 7_evaluate_pretrained.py  # Evaluación de modelos preentrenados
│   ├── 8_prepare_yolo.py         # Preparar dataset en formato YOLO
│   ├── 9_add_negatives.py        # Añadir imágenes negativas
│   ├── 10_integrate_new_datasets.py  # Integración de nuevos datasets
│   ├── 11_screening_architectures.py # Screening de 12 arquitecturas
│   ├── download_classification_data.py  # Descarga datos de clasificación
│   ├── generate_pseudo_masks.py  # Generación de pseudo-máscaras
│   ├── train_classifier.py       # Entrenamiento del clasificador
│   ├── train_classifier_dask.py  # Entrenamiento distribuido con Dask
│   ├── train_yolo_seg.py         # Entrenamiento YOLO11-seg
│   ├── train_individual.py       # Entrenamiento individual por arquitectura
│   ├── train_fpn_efficientnet.py # FPN + EfficientNet-B3
│   ├── train_fpn_resnet101.py    # FPN + ResNet-101
│   ├── train_fpn_segformer.py    # FPN + SegFormer
│   ├── train_unet_efficientnet.py    # U-Net + EfficientNet-B3
│   ├── train_unet_resnet101.py   # U-Net + ResNet-101
│   ├── train_unet_segformer.py   # U-Net + SegFormer
│   ├── train_deeplabv3plus_efficientnet.py  # DeepLabV3+ + EfficientNet-B3
│   ├── train_deeplabv3plus_resnet101.py     # DeepLabV3+ + ResNet-101
│   ├── run_all_training.py       # Ejecutar todos los entrenamientos
│   ├── monitor_training.py       # Monitoreo de entrenamiento en vivo
│   └── inference/
│       ├── predecir.py           # Inferencia de segmentación semántica
│       ├── predecir_yolo_seg.py  # Inferencia YOLO11-seg
│       ├── predecir_yolo_unet.py # Inferencia híbrida YOLO + U-Net
│       ├── comparar_modelos_screening.py  # Comparativa de 12 modelos
│       ├── test_miimage.py       # Test rápido de una imagen
│       └── test_instancias.py    # Test de instancias (watershed + SAM2)
│
├── tests/                        # Suite de tests
│   ├── conftest.py               # Fixtures compartidos
│   ├── test_solid_final.py       # Verificación de estructura SOLID
│   ├── unit/                     # 16 tests unitarios
│   ├── integration/              # 3 tests de integración
│   ├── clinical/                 # Tests clínicos
│   └── fixtures/                 # Datos de prueba
│
├── config/                       # Archivos de configuración YAML
│   ├── data_config.yaml          # Fuentes de datos y preprocessing
│   ├── model_config.yaml         # Hiperparámetros de modelos
│   ├── clinical_config.yaml      # Configuración de validación clínica
│   └── environments/             # Desarrollo / Staging / Producción
│
├── docs/                         # Documentación
│   ├── technical/architecture.md # Arquitectura del sistema
│   ├── clinical/                 # Protocolo clínico
│   └── regulatory/               # CE marking, FDA, análisis de riesgos
│
├── models/                       # Checkpoints (principalmente gitignored)
│   ├── screening/                # 12 modelos del screening
│   │   └── FPN_EfficientNetB3_best.pth  ← MEJOR MODELO
│   ├── general/                  # Modelos de propósito general
│   ├── comparativa/              # Modelos clásicos
│   ├── classifier/               # Checkpoints del clasificador
│   │   └── best.pth              ← Mejor checkpoint (trackeado con LFS)
│   └── sam2/                     # Pesos de SAM2 (descarga automática)
│
├── data/                         # Datos (principalmente gitignored)
│   ├── raw/                      # Datos crudos descargados
│   ├── processed/                # CSVs de datasets procesados
│   └── reports/                  # Reportes generados
│
├── data-clasificador/            # Splits del clasificador (CSVs trackeados)
│
├── .github/workflows/ci.yml      # CI/CD con GitHub Actions
├── .env.example                  # Variables de entorno de ejemplo
├── requirements.txt              # Dependencias Python
├── comandos.txt                  # Guía completa de comandos
├── LICENSE                       # MIT License
└── README.md
```

---

## Requisitos e Instalación

### Requisitos del sistema
- Python 3.10+
- GPU NVIDIA con CUDA 12.1+ (recomendado para entrenamiento)
- 8GB+ VRAM (mínimo para entrenar U-Net ligero)
- 16GB+ RAM

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/lkongc1/sequential-wound-image-analysis.git
cd sequential-wound-image-analysis

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate   # Linux/Mac

# Instalar PyTorch con CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar dependencias
pip install -r requirements.txt
```

### Variables de entorno

```bash
# Copiar el archivo de ejemplo y ajustar según tu entorno
cp .env.example .env
```

---

## Inicio Rápido

### Inferencia inmediata (segmentación semántica)

```bash
# Predicción básica con el mejor modelo (FPN + EfficientNet-B3)
python scripts/inference/predecir.py ruta/imagen.jpg

# Ajustar sensibilidad (umbral más alto = más conservador)
python scripts/inference/predecir.py ruta/imagen.jpg --umbral 0.6

# Post-procesamiento completo
python scripts/inference/predecir.py ruta/imagen.jpg --umbral 0.5 --limpiar 3 --rellenar 7 --area-min 0.005
```

**Salidas generadas** junto a la imagen original:
- `imagen_original.png` — copia de la original
- `imagen_mascara.png` — máscara binaria (blanco = herida)
- `imagen_mascara_cruda.png` — máscara sin filtros (para comparar)
- `imagen_superpuesto.png` — overlay sobre la herida detectada

---

## Pipeline de Datos

El pipeline se ejecuta en orden numérico:

```bash
# 1. Descargar dataset base (Medetec + fuentes online)
python scripts/1_download_dataset.py

# 2. Construir dataset unificado (organizar + dividir train/val/test)
python scripts/2_build_dataset.py

# 3. Análisis exploratorio
python scripts/3_eda.py

# 4. Agregar imágenes negativas (sin herida) para reducir falsos positivos
python scripts/9_add_negatives.py

# 5. Integrar datasets adicionales (Roboflow, CO2)
python scripts/10_integrate_new_datasets.py
```

### Datasets utilizados

| Dataset | Fuente | Tipo | Uso |
|---------|--------|------|-----|
| Medetec | Kaggle | Heridas clínicas | Segmentación (base) |
| FUSC | Kaggle | Úlceras de pie | Segmentación |
| WSNet | Kaggle | Heridas quirúrgicas | Segmentación |
| Roboflow | API | Heridas diversas | Clasificación |
| CO2 Wounds | Externo | Heridas CO2 láser | Clasificación |

---

## Entrenamiento

### Screening de Arquitecturas (12 modelos)

Evalúa 4 arquitecturas con 3-4 encoders cada una:

```bash
# Screening COMPLETO (12 combinaciones, 20 épocas c/u)
python scripts/11_screening_architectures.py
```

| Arquitectura | Encoders evaluados |
|-------------|-------------------|
| U-Net | ResNet-101, ResNeXt-50, EfficientNet-B3, SegFormer (mit_b2) |
| FPN | ResNet-101, ResNeXt-50, EfficientNet-B3, SegFormer (mit_b2) |
| DeepLabV3+ | ResNet-101, ResNeXt-50, EfficientNet-B3 |
| DeepLabV3 | ResNeXt-50 |

### Entrenamiento Individual

```bash
# FPN + EfficientNet-B3 (MEJOR combinación)
python scripts/train_fpn_efficientnet.py

# DeepLabV3+ + ResNet-101
python scripts/train_deeplabv3plus_resnet101.py

# U-Net + SegFormer
python scripts/train_unet_segformer.py

# Script genérico para cualquier arquitectura
python scripts/train_individual.py --arch deeplabv3 --encoder resnext50_32x4d --name DeepLabV3_ResNeXt50
```

### Modelos Clásicos (entrenamiento largo, 50+ épocas)

```bash
# U-Net ResNet50 desde cero
python scripts/6_train_unet_final.py

# U-Net ResNet18 (rápido, menos preciso)
python scripts/6b_train_unet_r18.py
```

### YOLO11 — Segmentación de Instancias

```bash
# Preparar dataset en formato YOLO
python scripts/8_prepare_yolo.py

# Entrenar YOLO11n-seg
python scripts/train_yolo_seg.py
```

### Clasificador de Tipos de Herida

```bash
# Descargar datos de clasificación
python scripts/download_classification_data.py

# Generar pseudo-máscaras (domain adaptation)
python scripts/generate_pseudo_masks.py

# Entrenamiento estándar
python scripts/train_classifier.py

# Entrenamiento distribuido con Dask (datasets grandes)
python scripts/train_classifier_dask.py
```

---

## Evaluación

```bash
# Evaluar TODOS los modelos del screening (métricas + gráficos)
python scripts/5_evaluate.py

# Monitorear entrenamiento en vivo
python scripts/monitor_training.py

# Ejecutar TODOS los entrenamientos en secuencia
python scripts/run_all_training.py
```

### Métricas reportadas
- **Segmentación**: Dice Score, IoU (Jaccard), Sensitivity, Specificity, Precision
- **Clasificación**: Accuracy, Macro F1, Per-class Precision/Recall, Matriz de Confusión
- **Clínicas**: Área de herida (cm²), cambio de coloración, progresión temporal

---

## Inferencia

### Segmentación Semántica (máscara binaria única)

```bash
# Comando mínimo
python scripts/inference/predecir.py ruta/imagen.jpg

# Ajuste de umbral
python scripts/inference/predecir.py ruta/imagen.jpg --umbral 0.3   # más sensible
python scripts/inference/predecir.py ruta/imagen.jpg --umbral 0.7   # más conservador

# Post-procesamiento
python scripts/inference/predecir.py ruta/imagen.jpg --limpiar 3 --rellenar 7
python scripts/inference/predecir.py ruta/imagen.jpg --area-min 0.01    # filtrar blobs chicos
python scripts/inference/predecir.py ruta/imagen.jpg --solo-mayor       # solo la herida más grande

# Avanzado
python scripts/inference/predecir.py ruta/imagen.jpg --tta              # Test-Time Augmentation
python scripts/inference/predecir.py ruta/imagen.jpg --cruda            # máscara sin filtros
python scripts/inference/predecir.py ruta/imagen.jpg --color rojo       # cambiar color overlay

# Comparar los 12 modelos del screening
python scripts/inference/comparar_modelos_screening.py
python scripts/inference/comparar_modelos_screening.py --imagen ruta/imagen.jpg
```

### Segmentación de Instancias (heridas individuales)

```bash
# Watershed — rápido, casos simples
python scripts/inference/predecir.py ruta/imagen.jpg --instancias

# SAM2 — alta precisión, separa heridas pegadas
python scripts/inference/predecir.py ruta/imagen.jpg --sam
python scripts/inference/predecir.py ruta/imagen.jpg --sam --sam-model sam2.1_hiera_s    # rápido
python scripts/inference/predecir.py ruta/imagen.jpg --sam --sam-model sam2.1_hiera_b+   # balanceado
# Por defecto usa sam2.1_hiera_l (224M params, ~900MB)

# YOLO11-seg — segmentación de instancias nativa
python scripts/inference/predecir_yolo_seg.py ruta/imagen.jpg
python scripts/inference/predecir_yolo_seg.py ruta/imagen.jpg --conf 0.3 --guardar-mapa
```

### Inferencia Híbrida YOLO + U-Net

```bash
# YOLO detecta ROIs, U-Net segmenta dentro de cada ROI
python scripts/inference/predecir_yolo_unet.py ruta/imagen.jpg
```

---

## Model Zoo

### Segmentación Semántica — Screening (12 modelos)

| Arquitectura | Encoder | Checkpoint |
|-------------|---------|------------|
| **FPN** | **EfficientNet-B3** ⭐ | `models/screening/FPN_EfficientNetB3_best.pth` |
| FPN | ResNet-101 | `models/screening/FPN_ResNet101_best.pth` |
| FPN | ResNeXt-50 | `models/screening/FPN_ResNeXt50_best.pth` |
| FPN | SegFormer (mit_b2) | `models/screening/FPN_SegFormer_best.pth` |
| U-Net | EfficientNet-B3 | `models/screening/UNet_EfficientNetB3_best.pth` |
| U-Net | ResNet-101 | `models/screening/UNet_ResNet101_best.pth` |
| U-Net | ResNeXt-50 | `models/screening/UNet_ResNeXt50_best.pth` |
| U-Net | SegFormer (mit_b2) | `models/screening/UNet_SegFormer_best.pth` |
| DeepLabV3+ | ResNet-101 | `models/screening/DeepLabV3Plus_ResNet101_best.pth` |
| DeepLabV3+ | ResNeXt-50 | `models/screening/DeepLabV3Plus_ResNeXt50_best.pth` |
| DeepLabV3+ | EfficientNet-B3 | `models/screening/DeepLabV3Plus_EfficientNetB3_best.pth` |
| DeepLabV3 | ResNeXt-50 | `models/screening/DeepLabV3_ResNeXt50_best.pth` |

### Modelos Clásicos

| Arquitectura | Encoder | Checkpoint |
|-------------|---------|------------|
| U-Net | ResNet-50 | `models/general/unet_resnet50_v4.pth` |
| U-Net++ | — | `models/comparativa/nested_unet.pth` |
| Attention U-Net | — | `models/comparativa/attention_unet.pth` |
| DeepLabV3 | ResNet-50 | `models/comparativa/deeplabv3.pth` |

### Segmentación de Instancias

| Modelo | Checkpoint | Notas |
|--------|-----------|-------|
| YOLO11n-seg | `models/screening/yolo11_seg_best.pt` | Segmentación de instancias nativa |
| SAM2.1 Hiera-L | Descarga automática | 224M params, ~900MB, más preciso |
| SAM2.1 Hiera-B+ | Descarga automática | Balance precisión/velocidad |
| SAM2.1 Hiera-S | Descarga automática | Más rápido, menos preciso |

### Clasificación

| Modelo | Checkpoint | Clases |
|--------|-----------|--------|
| EfficientNet-B3 (4ch) | `models/classifier/best.pth` | 7 tipos de herida |

---

## API

API REST con FastAPI para diagnóstico y gestión de pacientes.

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Health check del servicio |
| `POST` | `/diagnosis` | Segmentar y clasificar una imagen de herida |
| `GET` | `/patients` | Listar pacientes |
| `POST` | `/patients` | Crear paciente |
| `GET` | `/patients/{id}` | Obtener paciente por ID |
| `GET` | `/patients/{id}/studies` | Historial de estudios del paciente |

### Iniciar la API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Tests

```bash
# Suite completa
python -m pytest tests/ -v

# Tests unitarios (16 tests)
python -m pytest tests/unit/ -v

# Tests de integración (3 tests)
python -m pytest tests/integration/ -v

# Tests de un área específica
python -m pytest tests/unit/test_instance_seg.py -v
python -m pytest tests/unit/test_classifier.py -v

# Verificación de estructura SOLID
python tests/test_solid_final.py
```

### Cobertura de tests

| Categoría | Cantidad | Descripción |
|-----------|----------|-------------|
| Unitarios | 16 tests | Modelos, losses, métricas, config, factory, pipeline |
| Integración | 3 tests | API, inferencia, pipeline end-to-end |
| Estructura | 1 test | Verificación SOLID de `src/` |

### CI/CD

GitHub Actions ejecuta automáticamente en cada push y PR a `main`:
1. Lint con Ruff
2. Type check con MyPy
3. Tests unitarios
4. Tests de integración
5. Validación clínica

---

## Configuración

El proyecto usa múltiples niveles de configuración:

| Archivo | Propósito |
|---------|-----------|
| `.env` | Variables de entorno (paths, device, API) |
| `src/config.py` | Dataclasses tipadas con validación |
| `config/data_config.yaml` | Fuentes de datos, splits, augmentaciones |
| `config/model_config.yaml` | Hiperparámetros de modelos y losses |
| `config/clinical_config.yaml` | Umbrales y parámetros clínicos |
| `config/environments/*.yaml` | Overrides por entorno (dev/staging/prod) |

---

## Documentación

Documentación adicional disponible en `docs/`:

| Documento | Descripción |
|-----------|-------------|
| `docs/technical/architecture.md` | Arquitectura del sistema, componentes y despliegue |
| `docs/clinical/clinical_protocol.md` | Protocolo clínico para validación |
| `docs/regulatory/ce_marking.md` | Requisitos para marcado CE |
| `docs/regulatory/fda_submission.md` | Guía de sumisión FDA |
| `docs/regulatory/risk_analysis.md` | Análisis de riesgos (ISO 14971) |

---

## Validación Clínica

El sistema incluye métricas de validación clínica en `src/metrics/clinical_metrics.py`:

- **Área de herida**: cálculo en cm² usando referencia de escala
- **Cambio de coloración**: seguimiento de evolución del tejido
- **Progresión temporal**: comparación de estudios consecutivos del mismo paciente

---

## Guía Completa de Comandos

Todos los comandos detallados están documentados en [`comandos.txt`](comandos.txt), incluyendo:

- Preparación de datos (scripts 1-3, 9-10)
- Entrenamiento de los 12 modelos del screening
- Entrenamiento de modelos clásicos (U-Net, U-Net++)
- YOLO11 segmentación de instancias
- Evaluación y monitoreo
- Inferencia semántica, instancias e híbrida
- Tests unitarios y de integración

---

## Licencia

MIT License — ver [LICENSE](LICENSE) para más detalles.

Copyright (c) 2026 lkongc1
