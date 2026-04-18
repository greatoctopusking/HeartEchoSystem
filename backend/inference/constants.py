"""类别常量定义"""

ALL_CLASS_NAMES = ['A4C', 'PL', 'PSAV', 'PSMV', 'Random', 'SC', 'A2C']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(ALL_CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(ALL_CLASS_NAMES)}

BINARY_CLASS_NAMES = ['A2C', 'A4C']
BINARY_CLASS_TO_IDX = {'A2C': 0, 'A4C': 1}
BINARY_IDX_TO_CLASS = {0: 'A2C', 1: 'A4C'}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NUM_CLASSES = 7
DEFAULT_IMAGE_SIZE = 224


def get_class_names(task_type: str = 'multi_class') -> list:
    """
    根据任务类型获取类别名称
    
    Args:
        task_type: 'multi_class' 或 'binary'
    
    Returns:
        类别名称列表
    """
    if task_type == 'binary':
        return BINARY_CLASS_NAMES
    return ALL_CLASS_NAMES


def get_class_to_idx(task_type: str = 'multi_class') -> dict:
    """根据任务类型获取类别到索引的映射"""
    if task_type == 'binary':
        return BINARY_CLASS_TO_IDX
    return CLASS_TO_IDX


def get_idx_to_class(task_type: str = 'multi_class') -> dict:
    """根据任务类型获取索引到类别的映射"""
    if task_type == 'binary':
        return BINARY_IDX_TO_CLASS
    return IDX_TO_CLASS