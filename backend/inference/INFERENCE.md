# 推理模块使用说明

本模块提供心脏超声切面分类模型的推理功能，支持单张图像和批量图像推理。

---

## 1 快速开始

### 1.1 配置文件推理

```bash
python -m inference --config inference_config.yaml --input image.jpg
```

### 1.2 命令行参数推理

```bash
python -m inference --checkpoint checkpoints/best_model.pth --input image.jpg
```

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
| `num_classes` | int | 7 | 分类类别数 |
| `image_size` | int | 224 | 输入图像尺寸 |
| `batch_size` | int | 32 | 批大小 |
| `device` | string | cuda | 推理设备 (cuda/cpu) |
| `input` | string | ~ | 单图路径 |
| `input_dir` | string | ~ | 图像目录路径 |
| `output` | string | inference_results.json | 输出JSON文件 |
| `recursive` | bool | true | 递归搜索子目录 |

**注意：** YAML 中使用 `~` 表示 null，不要使用 `null`

### 3.2 参数优先级

命令行参数 > 配置文件 > 默认值

---

## 4 使用示例

### 4.1 单张图像推理

```bash
python -m inference \
    --config inference_config.yaml \
    --input "CACTUS/Images Dataset/A4C/1_D15_frame_600_v2.jpg"
```

### 4.2 批量图像推理

```bash
python -m inference \
    --config inference_config.yaml \
    --input-dir "CACTUS/Images Dataset/A4C/"
```

### 4.3 输出文件

`inference_results.json` 格式：

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

批量推理时（带目录输入）：

```json
{
  "total_images": 100,
  "inference_time_seconds": 5.23,
  "class_distribution": {
    "A4C": 45,
    "PL": 20,
    "PSAV": 15,
    "PSMV": 10,
    "Random": 5,
    "SC": 3,
    "A2C": 2
  },
  "results": [
    { "image_path": "image1.jpg", "predicted_class": "A4C", "confidence": 0.9987 },
    { "image_path": "image2.jpg", "predicted_class": "PL", "confidence": 0.9876 }
  ]
}
```

---

## 5 Python API

### 5.1 基本用法

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

### 5.2 批量推理

```python
from inference.predict import predict_directory

output = predict_directory(
    model,
    'path/to/image_dir/',
    output_path='results.json',
    recursive=True
)

print(output['class_distribution'])
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

1. **模型类别匹配**：确保模型类别数与配置文件 `num_classes` 匹配
2. **设备选择**：无 CUDA 时自动使用 CPU
3. **图像尺寸**：自动 resize 到 224×224
4. **批量推理**：大目录建议使用 `--input-dir` 提升效率

---

*文档版本：v1.1*  
*最后更新：2026.4.5*