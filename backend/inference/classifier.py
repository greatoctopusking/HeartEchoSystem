"""分类器模型（独立模块）"""
import torch
import torch.nn as nn
from typing import Optional

from .backbone import USFMAEEncoder, create_usfmae_backbone


class CardiacClassifier(nn.Module):
    """
    心脏超声切面分类模型
    基于 USF-MAE Backbone + 分类头
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 7,
        use_cls_token: bool = True,
        use_global_avg_pool: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        self.use_global_avg_pool = use_global_avg_pool
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        feature_dim = backbone.get_feature_dim()
        
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        
        if self.use_cls_token:
            cls_features = features[:, 0]
        elif self.use_global_avg_pool:
            cls_features = features[:, 1:].mean(dim=1)
        else:
            cls_features = features.mean(dim=1)
        
        if self.dropout is not None:
            cls_features = self.dropout(cls_features)
        
        logits = self.classifier(cls_features)
        
        return logits
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        
        if self.use_cls_token:
            embedding = features[:, 0]
        else:
            embedding = features[:, 1:].mean(dim=1)
        
        return embedding


def load_model(
    checkpoint_path: str,
    num_classes: int = 7,
    freeze_backbone: bool = False,
    device: str = 'cuda'
) -> CardiacClassifier:
    """
    加载训练好的分类模型
    
    Args:
        checkpoint_path: 模型权重文件路径
        num_classes: 分类类别数
        freeze_backbone: 是否冻结 backbone
        device: 设备
        
    Returns:
        加载好的 CardiacClassifier 模型
    """
    backbone = create_usfmae_backbone(
        pretrained_path=None,
        freeze=freeze_backbone,
        device=device
    )
    
    model = CardiacClassifier(
        backbone=backbone,
        num_classes=num_classes,
        use_cls_token=True,
        use_global_avg_pool=True
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    # [DEBUG] 打印权重加载信息
    if missing:
        print(f"[WARN] 缺少的权重键 ({len(missing)} 个): {missing[:5]}...")
    if unexpected:
        print(f"[WARN] 意外的权重键 ({len(unexpected)} 个): {unexpected[:5]}...")
    if not missing and not unexpected:
        print(f"[INFO] 模型权重加载成功，所有键匹配")
    
    # [DEBUG] 验证 backbone 是否有有效权重（非零）
    backbone_weight_sum = sum(p.sum().item() for p in backbone.parameters())
    print(f"[DEBUG] Backbone 权重总和: {backbone_weight_sum:.4f}")
    if abs(backbone_weight_sum) < 0.001:
        print(f"[WARN] ⚠️ Backbone 权重可能未正确加载（值接近0）！")
    
    model = model.to(device)
    model.eval()
    
    return model