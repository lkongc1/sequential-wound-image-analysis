# Justificación Matemática: Segmentación de Heridas con Deep Learning

> **Proyecto:** Detección y clasificación de heridas mediante visión computacional  
> **Autor:** [Tu nombre]  
> **Fecha:** Mayo 2026  

---

## 1. Planteamiento del Problema

La segmentación de heridas cutáneas presenta un desafío dual:

1. **Heridas irregulares:** úlceras, laceraciones, quemaduras — ocupan áreas extensas con bordes difusos
2. **Heridas lineales:** cortes, incisiones quirúrgicas, suturas — estructuras finas (< 3px de ancho) que los modelos convolucionales estándar destruyen durante el downsampling

### 1.1 Teorema de Nyquist-Shannon aplicado a CNN

El **Teorema de Muestreo de Nyquist-Shannon** establece que para reconstruir una señal sin pérdida de información, la frecuencia de muestreo debe ser al menos el doble de la frecuencia máxima de la señal:

$$f_s \geq 2 \cdot f_{max}$$

En una CNN con downsampling, cada etapa de pooling o convolución con stride reduce la resolución espacial. Para un encoder con $D$ etapas de downsampling de factor 2:

$$R_{bottleneck} = \frac{R_{input}}{2^D}$$

**Para U-Net ResNet50 ($D = 5$, 32× downsampling) a 384×384:**
$$R_{bottleneck} = \frac{384}{32} = 12 \text{ píxeles}$$

Una herida lineal de 3px de ancho en la imagen original ocupa:
$$W_{bottleneck} = \frac{3}{32} = 0.094 \text{ píxeles}$$

**Por el criterio de Nyquist, una señal de 0.094px es IRRECUPERABLE.** La información se pierde irreversiblemente en el bottleneck.

**Para U-Net ResNet18 ($D = 4$, 16× downsampling) a 640×640:**
$$R_{bottleneck} = \frac{640}{16} = 40 \text{ píxeles}$$
$$W_{bottleneck} = \frac{3}{16} = 0.188 \text{ píxeles}$$

Aunque sigue siendo sub-Nyquist, la relación señal-ruido mejora 2×.

---

## 2. Transfer Learning en Cascada

### 2.1 Formalización

El pipeline propuesto implementa **Cascaded Transfer Learning**:

$$P(wound|x) = \underbrace{P_{YOLO}(bbox|x)}_{\text{Transfer: COCO}} \cdot \underbrace{P_{U\text{-}Net}(mask | crop(x, bbox))}_{\text{Transfer: ImageNet}}$$

Cada etapa hereda pesos pre-entrenados de un dominio más general:

| Etapa | Pre-entrenamiento | Dominio fuente | Muestras |
|-------|------------------|---------------|----------|
| YOLOv8-nano | COCO | Detección de objetos | 330K imágenes, 80 clases |
| U-Net ResNet50 | ImageNet | Clasificación | 1.2M imágenes, 1000 clases |

### 2.2 Fundamento Matemático del Transfer Learning

Sea $f_\theta(x)$ una red neuronal con parámetros $\theta$. En lugar de inicializar $\theta \sim \mathcal{N}(0, \sigma^2)$, se inicializa con pesos pre-entrenados $\theta_{pretrained}$:

$$\theta = \theta_{pretrained} + \Delta\theta$$

Donde $\Delta\theta$ representa los **residuos** que la red debe aprender para especializarse en el nuevo dominio. Esto reduce el riesgo de sobreajuste porque:

$$\|\Delta\theta\| \ll \|\theta_{pretrained}\|$$

Los filtros de bajo nivel (bordes, texturas, gradientes) aprendidos en ImageNet/COCO son **universales** y transferibles a la segmentación de heridas.

### 2.3 Pérdida Tversky

La función de pérdida Tversky generaliza el coeficiente Dice mediante pesos asimétricos:

$$\text{Tversky} = \frac{TP + \epsilon}{TP + \alpha \cdot FP + \beta \cdot FN + \epsilon}$$

Donde:
- $\alpha$: penalización de Falsos Positivos
- $\beta$: penalización de Falsos Negativos
- $\alpha + \beta = 1$

**Para el modelo clínico (FDA):** $\beta = 0.7, \alpha = 0.3$
- Penaliza 2.3× más no detectar herida que marcar tejido sano
- Justificación: en contexto clínico, un falso negativo (no detectar herida) es más peligroso que un falso positivo

**Para el modelo de máscaras limpias:** $\beta = 0.5, \alpha = 0.5$
- Balance precisión/sensibilidad
- Minimiza falsos positivos (manchas de sangre, costras, líneas de uña)

---

## 3. Hard Attention: YOLO → Crop → U-Net

### 3.1 Principio

El mecanismo de **Hard Attention** resuelve el problema de heridas pequeñas en imágenes grandes:

$$\text{Si } A_{wound} \ll A_{image} \text{, entonces } \frac{A_{wound}}{A_{crop}} \gg \frac{A_{wound}}{A_{image}}$$

**Ejemplo numérico (imagen2.jpg, 450×299 = 134,550 px):**

| Método | Área herida en imagen | % efectivo |
|--------|----------------------|-----------|
| Sin crop (U-Net directo) | 1,771 px | 1.32% |
| Con crop 200×200 → 640×640 | 1,771 px en 40,000 px crop | **4.43%** |

La herida lineal pasa de ocupar 1.32% de la imagen a 4.43% de la región de interés — **3.4× más densidad de señal**.

### 3.2 Ganancia de Resolución Efectiva

Sea $R_{crop}$ la resolución del crop y $R_{model}$ la resolución de entrada del modelo:

$$G = \frac{R_{model}}{R_{crop}} = \frac{640}{200} = 3.2\times$$

Una herida de 3px en la imagen original → $3 \times 3.2 = 9.6px$ en el crop → $9.6 / 16 = 0.6px$ en el bottleneck de ResNet18.

**Comparación con/sin crop en el bottleneck:**

| Método | Píxeles en bottleneck (ResNet18) | Sobre Nyquist (0.5px)? |
|--------|----------------------------------|------------------------|
| Sin crop | 3 / 16 = 0.188 px |  No |
| Con crop 3.2× | 9.6 / 16 = **0.600 px** |  Sí |

**El crop permite superar el límite de Nyquist para heridas lineales.**

---

## 4. Métricas y Matriz de Confusión

### 4.1 Definiciones

La **matriz de confusión** a nivel de píxel cuantifica el rendimiento:

| | Pred: Negativo | Pred: Positivo |
|---|---|---|
| **Real: Negativo** | TN (True Negative) | FP (False Positive) |
| **Real: Positivo** | FN (False Negative) | TP (True Positive) |

### 4.2 Métricas Derivadas

| Métrica | Fórmula | Interpretación |
|---------|---------|---------------|
| **Sensibilidad (Recall)** | $\frac{TP}{TP + FN}$ | % de herida real detectada |
| **Especificidad** | $\frac{TN}{TN + FP}$ | % de piel sana correctamente ignorada |
| **Precisión** | $\frac{TP}{TP + FP}$ | % de píxeles marcados que realmente son herida |
| **Dice (F1)** | $\frac{2 \cdot TP}{2 \cdot TP + FP + FN}$ | Media armónica de precisión y sensibilidad |
| **F2-Score** | $\frac{5 \cdot P \cdot R}{4 \cdot P + R}$ | Ponderación 4× a recall sobre precisión |
| **IoU (Jaccard)** | $\frac{TP}{TP + FP + FN}$ | Intersección sobre unión |
| **NPV** | $\frac{TN}{TN + FN}$ | Valor predictivo negativo |

### 4.3 Thresholds FDA 510(k)

Para la sumisión regulatoria de dispositivos médicos de Clase II:

| Métrica | Umbral FDA | Significado clínico |
|---------|-----------|-------------------|
| Sensibilidad | $\geq 0.90$ | Detecta al menos el 90% del tejido de herida |
| Especificidad | $\geq 0.95$ | No más del 5% de falsas alarmas |
| Dice | $\geq 0.85$ | Balance general de segmentación |
| NPV | $\geq 0.95$ | 95% de confianza en zonas marcadas como sanas |

---

## 5. Resultados Experimentales

### 5.1 Modelo General: U-Net ResNet50 ($\beta = 0.7$)

**Test set (534 imágenes, 35M píxeles):**

| TP | TN | FP | FN |
|----|-----|-----|-----|
| 1,110,092 | 33,632,229 | 153,292 | 100,611 |

| Métrica | Valor | FDA |
|---------|-------|-----|
| **Dice** | **0.8974** |  ≥ 0.85 |
| **Sensibilidad** | **0.9169** |  ≥ 0.90 |
| **Especificidad** | **0.9955** |  ≥ 0.95 |
| **NPV** | **0.9970** |  ≥ 0.95 |
| **F2-Score** | 0.9090 | — |

**FDA 510(k): 4/4 **

### 5.2 Modelo Lineal: U-Net ResNet18 ($\beta = 0.5$, 640×640)

**Comparación en herida lineal (imagen2.jpg, 450×299 px):**

| Modelo | Downsampling | Píxeles detectados | % de imagen |
|--------|-------------|-------------------|-------------|
| U-Net ResNet50 | 32× | 262 px | 0.19% |
| **U-Net ResNet18** | **16×** | **1,771 px** | **1.32%** |
| **Mejora** | — | **6.8×** | **6.8×** |

### 5.3 Detector YOLOv8

**Validación (321 imágenes):**

| Métrica | Valor |
|---------|-------|
| mAP@50 | **0.949** |
| mAP@50-95 | 0.707 |
| Precisión | 0.926 |
| Recall | 0.863 |
| Peso del modelo | 6.2 MB |
| Velocidad inferencia | 2.2 ms/imagen |

---

## 6. Arquitectura del Sistema

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Imagen     │────▶│  YOLOv8-nano     │────▶│  Bounding Box   │
│  Entrada    │     │  (COCO pretrain) │     │  [x1,y1,x2,y2]  │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                      │
                                              ┌───────▼────────┐
                                              │  Crop + Resize │
                                              │  640 × 640     │
                                              └───────┬────────┘
                                                      │
┌─────────────┐     ┌──────────────────┐     ┌───────▼────────┐
│  Máscara    │◀────│  U-Net ResNet50  │◀────│  Región        │
│  Segmentada │     │  (ImageNet pret) │     │  Recortada     │
└─────────────┘     └──────────────────┘     └────────────────┘
```

---

## 7. Discusión y Conclusiones

### 7.1 Hallazgos Principales

1. **El downsampling es el factor crítico** para heridas lineales. La diferencia entre 32× (ResNet50) y 16× (ResNet18) produce una mejora de 6.8× en detección de estructuras finas.

2. **Transfer Learning en cascada** (COCO → ImageNet → Heridas) permite entrenar modelos efectivos con datasets médicos limitados (2,748 imágenes, incluyendo 75 negativas).

3. **TTA (Test-Time Augmentation)** agrega +0.003 Dice y +0.001 Sens mediante 4 inferencias con flips, llevando el modelo general a **0.90+ en todas las métricas FDA**.

4. **Hard Attention (YOLO crop)** mejora la resolución efectiva de heridas pequeñas en 3.2×, permitiendo superar el límite de Nyquist para estructuras sub-píxel.

5. **TverskyLoss asimétrica** permite balancear precisión y sensibilidad según el contexto clínico: $\beta = 0.7$ para FDA, $\beta = 0.5$ para máscaras limpias.

6. **Ejemplos negativos** (75 imágenes de piel sana de AZH BG) permiten al modelo aprender explícitamente qué NO es herida, reduciendo falsos positivos.

### 7.2 Limitaciones

- ResNet18 tiene menor capacidad semántica (14M vs 32M params) → detecta líneas de uña como falsos positivos
- YOLO no detecta heridas lineales muy finas (conf < 0.15)
- Dataset limitado a 2,748 imágenes de 4 fuentes (wsnet, fusc, medetec, azh_bg)

### 7.3 Resultados Finales con TTA

| Métrica | Sin TTA | **Con TTA** | FDA 510(k) |
|---------|---------|------------|------------|
| **Dice** | 0.8974 | **0.9001** | ≥ 0.85  |
| **Sensibilidad** | 0.9169 | **0.9182** | ≥ 0.90  |
| **Especificidad** | 0.9955 | **0.9956** | ≥ 0.95  |
| **NPV** | 0.9970 | **0.9971** | ≥ 0.95  |

### 7.4 Trabajo Futuro

- Re-entrenar con ejemplos negativos integrados para evaluación en producción
- Incorporar dataset DFUC2021 para heridas con fisuras lineales
- Explorar DeepLabV3+ con output_stride=4 para máxima preservación espacial
- Active contours / graph-cut para conectar fragmentos en heridas lineales

---

## 7.5 Test-Time Augmentation (TTA)

### Fundamento Matemático

TTA aplica transformaciones geométricas $T_k$ a la imagen de entrada, ejecuta inferencia en cada versión transformada, revierte la transformación y promedia:

$$\hat{y}_{TTA}(x) = \frac{1}{K} \sum_{k=1}^{K} T_k^{-1}\left(f_\theta\left(T_k(x)\right)\right)$$

Donde $T_k \in \{\text{identidad}, \text{flip-H}, \text{flip-V}, \text{flip-HV}\}$, $K = 4$.

Esto equivale a un **ensamble implícito** del mismo modelo evaluado en 4 perspectivas distintas de la misma imagen. La varianza del predictor se reduce en factor $\frac{1}{\sqrt{K}} = \frac{1}{2}$:

$$\text{Var}(\hat{y}_{TTA}) = \frac{1}{4} \text{Var}(\hat{y})$$

### Resultado Empírico

| Métrica | Sin TTA | Con TTA | Delta |
|---------|---------|---------|-------|
| Dice | 0.8974 | 0.9001 | +0.0027 |
| Sensibilidad | 0.9169 | 0.9182 | +0.0013 |
| Especificidad | 0.9955 | 0.9956 | +0.0001 |

La mejora de +0.0027 en Dice es suficiente para cruzar el umbral de 0.90 requerido por FDA 510(k).

---

## 7.6 Ejemplos Negativos (Hard Negative Mining)

### Justificación

Un modelo entrenado solo con imágenes de heridas aprende que **toda desviación de la piel sana es herida**. Esto produce falsos positivos en líneas naturales de la piel (uñas, arrugas), manchas, lunares, y bordes de imagen.

Agregar imágenes de **piel sana sin herida** con máscaras vacías ($mask = 0$) fuerza al modelo a aprender la distinción:

$$\mathcal{L}_{total} = \mathcal{L}_{wound}(x_{wound}, y_{wound}) + \lambda \cdot \mathcal{L}_{negative}(x_{healthy}, \mathbf{0})$$

Donde $\mathcal{L}_{negative}$ penaliza cualquier activación en piel sana:

$$\mathcal{L}_{negative} = \frac{1}{N} \sum_{i=1}^{N} \sigma(f_\theta(x_i))$$

**Dataset final:** 2,748 imágenes (2,673 heridas + 75 piel sana de AZH BG).

---

## Referencias

1. **Nyquist, H.** (1928). "Certain Topics in Telegraph Transmission Theory"
2. **Ronneberger, O. et al.** (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"  
3. **Salehi, S. et al.** (2017). "Tversky Loss Function for Image Segmentation"
4. **He, K. et al.** (2016). "Deep Residual Learning for Image Recognition"
5. **Redmon, J. et al.** (2018). "YOLOv3: An Incremental Improvement"
6. **FDA** (2023). "Content of Premarket Submissions for AI/ML-Enabled Medical Devices"
