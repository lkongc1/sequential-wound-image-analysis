# Deteccion de heridas — U-Net ResNet50

Modelo de segmentacion para detectar heridas en imagenes clinicas.

Tesis: *"Deteccion y clasificacion de los tipos de heridas mediante tecnicas de vision computacional"*

## Requisitos

```bash
pip install torch torchvision opencv-python numpy albumentations segmentation-models-pytorch
```

## Uso

```bash
python inferir.py imagen.png
```

Genera 3 archivos junto a la imagen original:
- `imagen_original.png` — copia de la original
- `imagen_mascara.png` — mascara binaria (blanco = herida)
- `imagen_superpuesto.png` — overlay rojo sobre la herida detectada

### Opciones

```bash
python inferir.py imagen.png --umbral 0.3   # mas sensible
python inferir.py imagen.png --umbral 0.7   # mas conservador
```

## Modelo

| Arquitectura | Encoder | Entrenamiento | Dice |
|-------------|---------|---------------|------|
| U-Net | ResNet50 (ImageNet) | 50 epocas, 256x256 | 0.89 |

## Metricas (test set)

```
                 PRED NEG    PRED POS
  REAL NEG      33,685,719       99,802
  REAL POS         154,233    1,056,470

  Accuracy         0.9927
  Precision        0.9137
  Recall           0.8726
  Specificity      0.9970
  F1-Score         0.8927
  ROC-AUC          0.9963
```
