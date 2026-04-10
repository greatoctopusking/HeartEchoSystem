"""类别常量定义"""

ALL_CLASS_NAMES = ['A4C', 'PL', 'PSAV', 'PSMV', 'Random', 'SC', 'A2C']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(ALL_CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(ALL_CLASS_NAMES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NUM_CLASSES = 7
DEFAULT_IMAGE_SIZE = 224