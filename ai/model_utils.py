import os
import glob
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
import pydicom
from pydicom.uid import ImplicitVRLittleEndian

import torch
import torch.nn.functional as F

from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from model import MultiHeadCTModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH_DEFAULT = BASE_DIR / "best_model.pth"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
IMG_SIZE = (380, 380)
NUM_MULTI = 4

MULTI_LABELS = ['covid', 'cancer_adenocarcinoma', 'cancer_largecell', 'cancer_squamous']

MODEL: Optional[torch.nn.Module] = None
DEVICE: Optional[torch.device] = None
GRADCAM = None

alb_transform = Compose([
    Resize(IMG_SIZE[0], IMG_SIZE[1]),
    Normalize(),
    ToTensorV2()
])

class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        out_bin, out_multi = self.model(input_tensor)
        pred_bin = int(out_bin.argmax(1).cpu().item())
        if pred_bin == 0:
            return None, pred_bin, int(out_multi.argmax(1).cpu().item()) if out_multi is not None else -1
        score = out_bin[:, 1] if class_idx is None else out_bin[:, class_idx]
        score.backward(retain_graph=True)
        gradients = self.gradients
        activations = self.activations
        grads_power_2 = gradients ** 2
        grads_power_3 = gradients ** 3
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        eps = 1e-8
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_activations * grads_power_3 + eps
        alpha = alpha_num / alpha_denom
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + eps)
        return cam.squeeze().cpu().numpy(), pred_bin, int(out_multi.argmax(1).cpu().item()) if out_multi is not None else -1


def init_model(model_path: Optional[str] = None, backbone_name: str = "resnet50", num_multiclass: int = NUM_MULTI):
    global MODEL, DEVICE, GRADCAM
    if model_path is None:
        model_path = MODEL_PATH_DEFAULT
    else:
        model_path = Path(model_path)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = MultiHeadCTModel(backbone_name=backbone_name, num_multiclass=num_multiclass).to(DEVICE)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    state = torch.load(str(model_path), map_location=DEVICE)
    MODEL.load_state_dict(state)
    MODEL.eval()

    try:
        target_layer = MODEL.backbone.layer4[-1]
    except Exception:
        target_layer = list(MODEL.backbone.children())[-1]

    GRADCAM = GradCAMpp(MODEL, target_layer)
    print(f"[model_utils] Model initialized on device: {DEVICE}")


def read_image_file(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def find_sidecar_image_for(path: str):
    p = Path(path)
    folder = p.parent
    stem = p.stem
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        candidate = folder / (stem + ext)
        if candidate.exists():
            return str(candidate)
    return None


def find_any_image_in_folder(folder: str, max_count: int = 50):
    folder = Path(folder)
    imgs = []
    for ext in ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp"):
        imgs.extend(sorted(folder.glob(ext)))
    imgs = [str(p) for p in imgs][:max_count]
    return imgs


def read_dicom_file_flexible(path: str) -> Tuple[Optional[np.ndarray], Optional[pydicom.dataset.FileDataset], Optional[str]]:
    path = str(path)
    try:
        ds = pydicom.dcmread(path, force=True)
    except Exception:
        img = read_image_file(path)
        if img is not None:
            return img, None, None
        return None, None, "Not a DICOM and cannot read as image"

    if not hasattr(ds, "PixelData") or ds.get('PixelData', None) is None:
        sc = find_sidecar_image_for(path)
        if sc:
            img = read_image_file(sc)
            if img is not None:
                return img, ds, None
        imgs = find_any_image_in_folder(Path(path).parent)
        if imgs:
            img = read_image_file(imgs[0])
            if img is not None:
                return img, ds, None
        try:
            if not hasattr(ds, "file_meta"):
                ds.file_meta = pydicom.dataset.FileMetaDataset()
            if not getattr(ds.file_meta, "TransferSyntaxUID", None):
                ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
            arr = ds.pixel_array
        except Exception as e:
            return None, ds, f"Unable to decode pixel data: {e}"
    else:
        try:
            arr = ds.pixel_array
        except Exception as e:
            try:
                if not hasattr(ds, "file_meta"):
                    ds.file_meta = pydicom.dataset.FileMetaDataset()
                if not getattr(ds.file_meta, "TransferSyntaxUID", None):
                    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
                arr = ds.pixel_array
            except Exception as e2:
                sc = find_sidecar_image_for(path)
                if sc:
                    img = read_image_file(sc)
                    if img is not None:
                        return img, ds, None
                imgs = find_any_image_in_folder(Path(path).parent)
                if imgs:
                    img = read_image_file(imgs[0])
                    if img is not None:
                        return img, ds, None
                return None, ds, f"Unable to decode pixel data after fallback: {e2}"

    try:
        arr = np.asarray(arr, dtype=np.float32)
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
        if arr.ndim == 2:
            img_rgb = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3:
            if arr.shape[2] == 1:
                img_rgb = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            elif arr.shape[2] == 3:
                tmp = arr.astype(np.uint8)
                try:
                    img_rgb = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
                except Exception:
                    img_rgb = tmp
            elif arr.shape[2] == 4:
                img_rgb = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGRA2RGB)
            else:
                img_rgb = arr[:, :, :3].astype(np.uint8)
        else:
            arr2 = arr.reshape(arr.shape[0], arr.shape[1])
            img_rgb = cv2.cvtColor(arr2.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return img_rgb, ds, None
    except Exception as e:
        return None, ds, f"Error converting pixel array to RGB: {e}"


def make_placeholder_image() -> np.ndarray:
    return (np.ones((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8) * 128)


def heatmap_rgb_from_cam(cam: np.ndarray) -> np.ndarray:
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("jet")
    heat = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)
    return heat


def save_heatmap_and_mask(img_rgb: np.ndarray, cam: np.ndarray, out_prefix: str) -> Tuple[Optional[str], Optional[str]]:
    if cam is None:
        return None, None
    heat = heatmap_rgb_from_cam(cam)
    overlay = cv2.addWeighted(img_rgb, 0.5, heat, 0.5, 0)
    heatpath = f"{out_prefix}_heatmap.png"
    maskpath = f"{out_prefix}_mask.png"
    cv2.imwrite(heatpath, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    mask = (cam > 0.5).astype(np.uint8) * 255
    cv2.imwrite(maskpath, mask)
    return heatpath, maskpath


def compute_95ci(p_mean: float, n: int) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    se = (p_mean * (1.0 - p_mean) / n) ** 0.5
    lower = max(0.0, p_mean - 1.96 * se)
    upper = min(1.0, p_mean + 1.96 * se)
    return lower, upper


def decide_pathology_from_ci(p_mean: float, lower: float, upper: float) -> Tuple[int, str]:
    if lower >= 0.5:
        return 1, ""
    if upper <= 0.5:
        return 0, ""
    return (1 if p_mean >= 0.5 else 0), "Uncertain CI"


def process_dicom_study(study_dir: str, save_images: bool = False) -> pd.DataFrame:
    global MODEL, DEVICE, GRADCAM
    if MODEL is None or GRADCAM is None:
        raise RuntimeError("Model not initialized. Call init_model() before processing.")

    t0_study = time.time()
    study_dir = Path(study_dir)

    files = sorted(glob.glob(str(study_dir / "**" / "*.dcm"), recursive=True))
    if not files:
        files = sorted(glob.glob(str(study_dir / "**" / "*.*"), recursive=True))

    per_probs = []
    per_multi_probs = []
    per_loc = []
    processing_notes = []
    study_uids = []
    series_uids = []

    for idx, fpath in enumerate(files):
        try:
            img_rgb, ds, err = read_dicom_file_flexible(fpath)
            used_placeholder = False
            if img_rgb is None:
                img_rgb = make_placeholder_image()
                used_placeholder = True

            if ds is not None:
                study_uids.append(getattr(ds, "StudyInstanceUID", "unknown"))
                series_uids.append(getattr(ds, "SeriesInstanceUID", "unknown"))

            augmented = alb_transform(image=img_rgb)
            tensor = augmented['image'].unsqueeze(0).to(DEVICE).float()

            with torch.no_grad():
                out_bin, out_multi = MODEL(tensor)
                probs_bin = torch.softmax(out_bin, dim=1).cpu().numpy()[0]
                prob_path = float(probs_bin[1])
                if out_multi is not None:
                    multi_probs = torch.softmax(out_multi, dim=1).cpu().numpy()[0]
                else:
                    multi_probs = np.zeros(len(MULTI_LABELS), dtype=float)

            cam, pred_bin, pred_multi_idx = GRADCAM(tensor)

            if cam is not None:
                thr = 0.5
                mask = (cam > thr).astype(np.uint8)
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    x_min = float(coords[:,:,0].min()); x_max = float(coords[:,:,0].max())
                    y_min = float(coords[:,:,1].min()); y_max = float(coords[:,:,1].max())
                else:
                    x_min=x_max=y_min=y_max=0.0
            else:
                x_min=x_max=y_min=y_max=0.0
            z_min = float(idx); z_max = float(idx)

            per_probs.append(prob_path)
            per_multi_probs.append(multi_probs)
            per_loc.append([x_min, x_max, y_min, y_max, z_min, z_max])

            note = "Success (placeholder)" if used_placeholder else "Success"
            processing_notes.append(note)

        except Exception as e:
            processing_notes.append(f"Failure: {str(e)}")
            continue

    n_slices = len(per_probs)
    if n_slices == 0:
        total_time = round(time.time() - t0_study, 3)
        study_uid = study_uids[0] if study_uids else "unknown"
        series_uid = series_uids[0] if series_uids else "unknown"
        result = {
            "path_to_study": str(study_dir.resolve()),
            "study_uid": str(study_uid),
            "series_uid": str(series_uid),
            "probability_of_pathology": 0.0,
            "pathology": 0,
            "processing_status": "Failure: no valid image slices",
            "time_of_processing": float(total_time),
            "most_dangerous_pathology_type": None,
            "pathology_localization": [0.0,0.0,0.0,0.0,0.0,0.0]
        }
        df = pd.DataFrame([result])
        cols_order = [
            "path_to_study","study_uid","series_uid","probability_of_pathology","pathology",
            "processing_status","time_of_processing","most_dangerous_pathology_type","pathology_localization"
        ]
        df = df[cols_order]
        return df

    p_mean = float(np.mean(per_probs))
    lower, upper = compute_95ci(p_mean, n_slices)
    pathology_int, ci_note = decide_pathology_from_ci(p_mean, lower, upper)

    if len(per_multi_probs) > 0:
        mean_multi = np.mean(np.stack(per_multi_probs, axis=0), axis=0)
        top_idx = int(np.argmax(mean_multi))
        most_dangerous = MULTI_LABELS[top_idx] if 0 <= top_idx < len(MULTI_LABELS) else str(top_idx)
    else:
        most_dangerous = None

    max_idx = int(np.argmax(per_probs))
    localization = per_loc[max_idx] if 0 <= max_idx < len(per_loc) else [0.0,0.0,0.0,0.0,0.0,0.0]

    any_failure = any(note.startswith("Failure") for note in processing_notes)
    status_main = "Success" if n_slices > 0 else "Failure"
    if any_failure and n_slices > 0:
        failure_count = sum(1 for note in processing_notes if note.startswith("Failure"))
        status = f"Partial Success: {failure_count} slice(s) failed"
    else:
        status = status_main

    if ci_note:
        status = status + f"; {ci_note}" if status else ci_note

    total_time = round(time.time() - t0_study, 3)

    def pick_most_common_or_first(lst: List[str]) -> str:
        filtered = [x for x in lst if x and x != 'unknown']
        if not filtered:
            return lst[0] if lst else "unknown"
        vals, counts = np.unique(filtered, return_counts=True)
        return vals[np.argmax(counts)]

    study_uid_val = pick_most_common_or_first(study_uids)
    series_uid_val = pick_most_common_or_first(series_uids)

    result_row = {
        "path_to_study": str(study_dir.resolve()),
        "study_uid": str(study_uid_val),
        "series_uid": str(series_uid_val),
        "probability_of_pathology": float(round(p_mean, 6)),
        "pathology": int(pathology_int),
        "processing_status": str(status),
        "time_of_processing": float(total_time),
        "most_dangerous_pathology_type": str(most_dangerous) if most_dangerous is not None else None,
        "pathology_localization": [float(x) for x in localization]
    }

    df = pd.DataFrame([result_row])

    cols_order = [
        "path_to_study",
        "study_uid",
        "series_uid",
        "probability_of_pathology",
        "pathology",
        "processing_status",
        "time_of_processing",
        "most_dangerous_pathology_type",
        "pathology_localization"
    ]
    df = df[cols_order]

    return df
