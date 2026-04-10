import os
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def export_single_fold_onnx():
    # 配置模型所在的根目录 (必须包含 plans.json, dataset.json 和 fold_0 文件夹)
    MODELS = {
        "2ch": {
            "model_folder": "Dataset500_Heart2CH/nnUNetTrainer__nnUNetPlans__2d",
            "onnx_out": "echo_seg_2ch_fast.onnx",
        },
        "4ch": {
            "model_folder": "Dataset501_Heart4CH/nnUNetTrainer__nnUNetPlans__2d",
            "onnx_out": "echo_seg_4ch_fast.onnx",
        }
    }

    for view, cfg in MODELS.items():
        print(f"\n🚀 开始导出 {view.upper()} 极速版 ONNX 模型...")
        
        # 1. 使用 nnUNet 官方 Predictor，让它帮我们自动解析所有配置文件
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=torch.device('cpu'), # 导出过程在 CPU 下进行即可
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )

        # 2. 自动初始化网络并加载 fold_0 的权重
        # 这个方法内部会自动读取 plans.json 和 dataset.json，绝对不会再报 KeyError
        predictor.initialize_from_trained_model_folder(
            cfg["model_folder"],
            use_folds=(0,),  # 只用 fold_0，保证速度
            checkpoint_name='checkpoint_final.pth'
        )

        # 3. 提取出加载好权重的网络
        network = predictor.network
        # 将唯一的 fold_0 权重真正加载进网络
        network.load_state_dict(predictor.list_of_parameters[0])
        network.eval()

        # 4. 自动获取模型训练时的图像尺寸 (Patch Size)
        patch_size = predictor.configuration_manager.patch_size
        H, W = int(patch_size[0]), int(patch_size[1])
        print(f"  -> 自动解析到输入尺寸: {H}x{W}")

        # 5. 导出 ONNX
        dummy_input = torch.randn(1, 1, H, W)
        torch.onnx.export(
            network,
            dummy_input,
            cfg["onnx_out"],
            input_names=["image"],
            output_names=["seg_logits"],
            dynamic_axes={
                "image":      {0: "batch", 2: "height", 3: "width"},
                "seg_logits": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

        size_mb = os.path.getsize(cfg["onnx_out"]) / 1024 / 1024
        print(f"✅ 导出成功: {cfg['onnx_out']} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    export_single_fold_onnx()