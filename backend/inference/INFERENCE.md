# 推理模块使用说明

本模块提供心脏超声切面分类模型的推理功能，支持单张图像和批量图像推理。

---

1 快速开始
1.1 多分类推理（7类）
python -m inference --checkpoint checkpoints/best_model.pth --input image.jpg

1.2 二分类推理（A2C vs A4C）
python -m inference \
    --checkpoint checkpoints/binary/best_model.pth \
    --task_type binary \
    --input image.jpg

1.3 带置信度过滤的推理 (新增)
# 当模型不确定时 (置信度 < 0.85)，输出 "Unknown/Other" 而非强制分类
python -m inference \
    --checkpoint checkpoints/binary/best_model.pth \
    --task_type binary \
    --threshold 0.85 \
    --input image.jpg
---

## 2 环境配置

### 2.1 依赖安装

```bash
pip install -r requirements.txt
```

### 2.2 依赖列表

| 依赖 | 版本 | 说明 |
|------|------|------|
| torch | ≥1.8.0 | PyTorch 深度学习框架 |
| torchvision | ≥0.9.0 | 图像处理工具 |
| Pillow | ≥8.0.0 | Python 图像库 |
| numpy | ≥1.19.0 | 数值计算 |
| PyYAML | ≥5.4.0 | YAML 配置文件解析 |
| nibabel | ≥3.2.0 | NIfTI 医学图像解析 |

### 2.3 运行环境

- **操作系统**：Windows / Linux / macOS
- **Python 版本**：3.8+
- **GPU**：NVIDIA GPU (CUDA) / CPU

### 2.4 目录结构

```
inference/
├── __init__.py
├── __main__.py           # 入口文件
├── cli.py               # 命令行接口
├── predict.py           # 推理核心
├── classifier.py        # 分类模型
├── transforms.py       # 数据预处理
├── constants.py        # 常量定义
└── inference_config.yaml # 配置文件
```

---

## 3 配置说明

### 3.1 inference_config.yaml

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `checkpoint` | string | - | 模型权重文件路径（必需） |
| `task_type` | string | multi_class | 任务类型：multi_class(7类) 或 binary(二分类) |
| `num_classes` | int | 7/2 | 分类类别数（根据task_type自动确定） |
| `image_size` | int | 224 | 输入图像尺寸 |
| `batch_size` | int | 32 | 批大小 |
| `device` | string | cuda | 推理设备 (cuda/cpu) |
| `input` | string | ~ | 单图路径 |
| `input_dir` | string | ~ | 图像目录路径 |
| `output` | string | inference_results.json | 输出JSON文件 |
| `recursive` | bool | true | 递归搜索子目录 |
threshold	float	~	 置信度阈值 (如 0.85)，低于此值判定为 Unknown/Other
**注意：** YAML 中使用 `~` 表示 null，不要使用 `null`

### 3.2 参数优先级

命令行参数 > 配置文件 > 默认值

### 3.3 任务类型说明

| task_type | num_classes | 类别名称 |
|-----------|-------------|----------|
| multi_class | 7 | A4C, PL, PSAV, PSMV, Random, SC, A2C |
| binary | 2 | A2C, A4C |

---

## 4 使用示例

### 4.1 多分类单张图像推理

```bash
python -m inference \
    --checkpoint checkpoints/best_model.pth \
    --input "CACTUS/Images Dataset/A4C/1_D15_frame_600_v2.jpg"
```

### 4.2 二分类单张图像推理

```bash
python -m inference \
    --checkpoint checkpoints/binary/best_model.pth \
    --task_type binary \
    --input "path/to/image.jpg"
```

### 4.3 多分类批量图像推理

```bash
python -m inference \
    --checkpoint checkpoints/best_model.pth \
    --input-dir "CACTUS/Images Dataset/A4C/"
```

### 4.4 二分类批量图像推理

```bash
python -m inference \
    --checkpoint checkpoints/binary/best_model.pth \
    --task_type binary \
    --input-dir "path/to/camus_images/"
```

### 4.5 使用配置文件

```yaml
# inference_config.yaml 示例 - 二分类
task_type: "binary"
checkpoint: "checkpoints/binary/best_model.pth"
input: "path/to/image.jpg"
device: "cuda"
```

```bash
python -m inference --config inference_config.yaml
```
# 二分类推理，仅当置信度 > 90% 时才输出 A2C 或 A4C
python -m inference \
    --checkpoint checkpoints/binary/best_model.pth \
    --task_type binary \
    --threshold 0.90 \
    --input "path/to/uncertain_image.jpg"

# 批量推理并过滤低置信度结果
python -m inference \
    --checkpoint checkpoints/binary/best_model.pth \
    --task_type binary \
    --threshold 0.85 \
    --input-dir "path/to/images/" \
    --output "filtered_results.json"
### 4.6 输出文件

多分类推理（7类）`inference_results.json` 格式：

```json
{
  "image_path": "path/to/image.jpg",
  "predicted_class": "A4C",
  "confidence": 0.9987,
  "all_probabilities": {
    "A4C": 0.9987,
    "PL": 0.0008,
    "PSAV": 0.0002,
    "PSMV": 0.0001,
    "Random": 0.0001,
    "SC": 0.0001,
    "A2C": 0.0000
  }
}
```

二分类推理（2类）`inference_results.json` 格式：

```json
{
  "image_path": "path/to/image.jpg",
  "predicted_class": "A4C",
  "confidence": 0.9523,
  "all_probabilities": {
    "A2C": 0.0477,
    "A4C": 0.9523
  }
}
```

批量推理时（带目录输入）：

```json
{
  "total_images": 100,
  "inference_time_seconds": 5.23,
  "class_distribution": {
    "A2C": 45,
    "A4C": 55
  },
  "results": [
    { "image_path": "image1.jpg", "predicted_class": "A4C", "confidence": 0.9523 },
    { "image_path": "image2.jpg", "predicted_class": "A2C", "confidence": 0.8876 }
  ]
}
```

---

## 5 Python API

### 5.1 多分类推理（7类）

```python
import torch
from inference.predict import load_model, predict_single
from inference.transforms import get_val_transforms

# 加载模型
model = load_model(
    'checkpoints/best_model.pth',
    num_classes=7,
    device='cuda'
)

# 获取预处理
transform = get_val_transforms(224)

# 推理单张图像
result = predict_single(
    model,
    'path/to/image.jpg',
    transform=transform,
    device='cuda'
)

print(result['predicted_class'])
print(result['confidence'])
```

### 5.2 二分类推理（A2C vs A4C）

```python
from inference.predict import load_model, predict_single
from inference.transforms import get_val_transforms
from inference.constants import get_class_names

# 加载模型（二分类）
model = load_model(
    'checkpoints/binary/best_model.pth',
    num_classes=2,
    device='cuda'
)

# 获取预处理和二分类类别名称
transform = get_val_transforms(224)
class_names = get_class_names('binary')  # ['A2C', 'A4C']

# 推理单张图像
result = predict_single(
    model,
    'path/to/image.jpg',
    transform=transform,
    device='cuda',
    class_names=class_names
)

print(result['predicted_class'])  # 'A2C' 或 'A4C'
print(result['confidence'])
```

### 5.3 批量推理

```python
from inference.predict import predict_directory
from inference.constants import get_class_names

# 多分类批量推理
output = predict_directory(
    model,
    'path/to/image_dir/',
    output_path='results.json',
    recursive=True
)

print(output['class_distribution'])

# 二分类批量推理
class_names = get_class_names('binary')
output = predict_directory(
    model,
    'path/to/image_dir/',
    output_path='binary_results.json',
    recursive=True,
    class_names=class_names
)

print(output['class_distribution'])
```

### 5.4 一键推理函数

```python
from inference.predict import predict_from_path

# 多分类
result = predict_from_path(
    checkpoint_path='checkpoints/best_model.pth',
    image_path='path/to/image.jpg',
    num_classes=7,
    device='cuda',
    task_type='multi_class'
)

# 二分类
result = predict_from_path(
    checkpoint_path='checkpoints/binary/best_model.pth',
    image_path='path/to/image.jpg',
    num_classes=2,
    device='cuda',
    task_type='binary'
)
```

---

## 6 支持的图像格式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tif, .tiff)
- NIfTI (.nii, .nii.gz)

---

## 7 注意事项

1. **任务类型匹配**：确保使用正确的 `--task_type` 参数
   - 多分类模型（7类）：`--task_type multi_class`
   - 二分类模型（2类）：`--task_type binary`
2. **模型类别匹配**：确保模型类别数与任务类型匹配
3. **设备选择**：无 CUDA 时自动使用 CPU
4. **图像尺寸**：自动 resize 到 224×224
5. **批量推理**：大目录建议使用 `--input-dir` 提升效率

---

## 8 常量说明

### 8.1 多分类常量

| 常量 | 说明 |
|------|------|
| `ALL_CLASS_NAMES` | ['A4C', 'PL', 'PSAV', 'PSMV', 'Random', 'SC', 'A2C'] |
| `NUM_CLASSES` | 7 |
| `CLASS_TO_IDX` | 类别名到索引的映射 |
| `IDX_TO_CLASS` | 索引到类别名的映射 |

### 8.2 二分类常量

| 常量 | 说明 |
|------|------|
| `BINARY_CLASS_NAMES` | ['A2C', 'A4C'] |
| `BINARY_CLASS_TO_IDX` | {'A2C': 0, 'A4C': 1} |
| `BINARY_IDX_TO_CLASS` | {0: 'A2C', 1: 'A4C'} |

### 8.3 辅助函数

```python
from inference.constants import get_class_names, get_class_to_idx, get_idx_to_class

# 根据任务类型获取类别名称
get_class_names('multi_class')  # ['A4C', 'PL', 'PSAV', 'PSMV', 'Random', 'SC', 'A2C']
get_class_names('binary')       # ['A2C', 'A4C']
```

---

*文档版本：v1.2*  
*最后更新：2026.4.9*