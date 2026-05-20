# Clinical Protocol

## Image Acquisition Standards

### Equipment Requirements
- Camera resolution: Minimum 512x512 pixels
- Color space: RGB (24-bit)
- Lighting: Standardized clinical lighting conditions
- Background: Clean, neutral background preferred

### Patient Positioning
- Wound should be centered in frame
- Minimum 2cm margin around wound edges
- Avoid obstructed views or shadows on wound bed

### Quality Criteria
- Image sharpness: Laplacian variance > 10.0
- Brightness range: 10-240 mean grayscale value
- Aspect ratio: < 5:1 (width/height)

## Segmentation Protocol

### Model Output Interpretation
- Threshold: 0.5 probability
- Minimum wound area: 50 pixels
- Post-processing: Morphological closing (kernel=5)

### Clinical Reporting
All segmentation results must include:
- Wound area in pixels
- Wound area as percentage of image
- Bounding box coordinates
- Confidence score

## Safety Thresholds

- Physician review required if Dice < 0.85
- Flag for re-evaluation if sensitivity < 0.80
- Auto-alert on ambiguous classifications (0.4-0.6 probability range)
