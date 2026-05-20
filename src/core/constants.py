"""Constants for the wound segmentation project."""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Image defaults
DEFAULT_IMAGE_SIZE = 384
DEFAULT_MIN_RESOLUTION = (256, 256)
DEFAULT_MAX_ASPECT_RATIO = 5.0

# Quality thresholds
DEFAULT_MIN_BRIGHTNESS = 10.0
DEFAULT_MAX_BRIGHTNESS = 240.0
DEFAULT_MIN_LAPLACIAN = 10.0
DEFAULT_MIN_WOUND_RATIO = 0.01
DEFAULT_MAX_WOUND_RATIO = 0.95

# Data splits
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_TEST_SPLIT = 0.1
DEFAULT_RANDOM_SEED = 42

# Training defaults
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 6
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 15
DEFAULT_ACCUMULATION_STEPS = 2

# Model types
MODEL_TYPES = [
    "unet_mini",
    "unet",
    "attention_unet",
    "nested_unet",
    "deeplabv3",
    "manet",
]

# Loss types
LOSS_TYPES = [
    "bce_dice",
    "tversky",
    "focal",
]

# Normalization
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
