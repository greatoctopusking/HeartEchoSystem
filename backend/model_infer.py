import os

from model_infer_nnunet import run_inference as run_inference_nnunet
from model_infer_onnx import run_inference as run_inference_onnx

#可选：默认 CAMUS/普通 NIfTI 是否优先走 ONNX
DEFAULT_PREFER_ONNX=True


def _normalize_source_type(source_type:str) -> str:
    if not source_type:
        return "camus"
    s=str(source_type).strip().lower()
    if s in ("video","avi","mp4","mov"):
        return "video"
    return "camus"


def run_inference(
    image_path,
    case_name,
    dataset_name,
    is_video=False,
    prefer_onnx=DEFAULT_PREFER_ONNX,
    source_type=None,
):
    """
    统一推理入口

    路由规则：
    1. 视频输入 -> 强制 nnUNet
    2. 非视频输入 + prefer_onnx=True -> ONNX
    3. 非视频输入 + prefer_onnx=False -> nnUNet

    返回格式与旧版保持一致：
        output_dir,pred_paths
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入文件不存在:{image_path}")

    src=_normalize_source_type(source_type)

    #is_video 优先级最高
    if is_video or src == "video":
        print("[ROUTER] 检测到视频输入，强制使用 nnUNet")
        return run_inference_nnunet(image_path,case_name,dataset_name)

    if prefer_onnx:
        try:
            print("[ROUTER] 非视频输入，优先使用 ONNX")
            return run_inference_onnx(image_path,case_name,dataset_name)
        except Exception as e:
            print(f"[ROUTER] ONNX 推理失败，自动回退 nnUNet:{e}")
            return run_inference_nnunet(image_path,case_name,dataset_name)

    print("[ROUTER] 非视频输入，按配置使用 nnUNet")
    return run_inference_nnunet(image_path,case_name,dataset_name)