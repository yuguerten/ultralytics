import pytest
import torch

from ultralytics import YOLO
from ultralytics.engine.trainer import DistillationLoss
from ultralytics.models.yolo import detect
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

def _dummy_batch(imgsz=32, batch=2, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.randn(batch, 3, imgsz, imgsz, device=device)

def _build_models():
    # Load lightweight student + larger teacher
    student = YOLO("yolo11n.pt").model.eval()
    teacher = YOLO("yolo11l.pt").model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    student.to(dev)
    teacher.to(dev)
    return student, teacher

def _forward_both(student, teacher, x):
    with torch.no_grad():
        teacher(x)  # teacher first (not required but often nice for caching)
        student(x)

@pytest.mark.parametrize("mode", ["cwd", "mgd"])
def test_feature_distillation_loss_modes(mode):
    student, teacher = _build_models()

    distill = DistillationLoss(student, teacher, distiller=mode)
    try:
        distill.register_hook()
        x = _dummy_batch()
        _forward_both(student, teacher, x)
        loss = distill.get_loss()
        assert torch.is_tensor(loss), "Loss is not a tensor"
        assert loss.ndim == 0, "Loss must be scalar"
        assert torch.isfinite(loss), f"{mode} loss not finite"
    finally:
        distill.remove_handle_()