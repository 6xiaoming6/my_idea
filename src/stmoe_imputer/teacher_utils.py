from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TeacherContext:
    model: torch.nn.Module | None
    metadata: dict[str, object]


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        value = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=map_location)
    if not isinstance(value, dict) or "model" not in value or "config" not in value:
        raise ValueError(f"Teacher checkpoint is missing model/config payload: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _teacher_git_metadata(checkpoint: Path) -> tuple[str, str]:
    train_log = checkpoint.parents[1] / "logs" / "train.log"
    try:
        content = train_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "v14-single", "unavailable"
    branch = re.search(r"^\s*git_branch:\s*(\S+)", content, flags=re.MULTILINE)
    commit = re.search(r"^\s*git_commit:\s*(\S+)", content, flags=re.MULTILINE)
    return (
        branch.group(1) if branch else "v14-single",
        commit.group(1) if commit else "unavailable",
    )


def _matches_experiment(payload: dict, cfg: dict, teacher_seed: int) -> bool:
    teacher_cfg = payload.get("config", {})
    teacher_data = teacher_cfg.get("data", {})
    student_data = cfg.get("data", {})
    teacher_mask = teacher_data.get("mask", {})
    student_mask = student_data.get("mask", {})
    teacher_model = teacher_cfg.get("model", {})
    try:
        rate_matches = math.isclose(
            float(teacher_mask.get("missing_rate")),
            float(student_mask.get("missing_rate")),
            abs_tol=1e-9,
        )
    except (TypeError, ValueError):
        return False
    return (
        teacher_model.get("architecture") == "v14_safe_c2f_moe"
        and teacher_data.get("dataset_name") == student_data.get("dataset_name")
        and teacher_mask.get("pattern") == student_mask.get("pattern")
        and rate_matches
        and int(teacher_cfg.get("seed", -1)) == teacher_seed
    )


def resolve_teacher_checkpoint(cfg: dict, project_root: Path = PROJECT_ROOT) -> Path:
    teacher_cfg = cfg.get("teacher", {})
    configured = str(teacher_cfg.get("checkpoint", "AUTO_RESOLVE"))
    if configured != "AUTO_RESOLVE":
        path = Path(configured).expanduser()
        path = path if path.is_absolute() else project_root / path
        if not path.is_file():
            raise FileNotFoundError(f"V16 teacher checkpoint does not exist: {path}")
        payload = _torch_load(path)
        teacher_seed = int(teacher_cfg.get("seed", cfg.get("seed", 42)))
        if not _matches_experiment(payload, cfg, teacher_seed):
            raise ValueError(
                "Explicit V14 teacher does not match dataset/mask/rate/teacher seed: "
                f"{path}"
            )
        return path.resolve()

    data_cfg = cfg["data"]
    mask_cfg = data_cfg["mask"]
    teacher_seed = int(teacher_cfg.get("seed", cfg.get("seed", 42)))
    rate_dir = f"rate{float(mask_cfg['missing_rate']):g}"
    search_root = (
        project_root
        / "outputs"
        / "v14-single"
        / str(data_cfg["dataset_name"])
        / "full"
        / "model"
        / str(mask_cfg["pattern"])
        / rate_dir
    )
    candidates: list[Path] = []
    for path in search_root.glob("*/checkpoints/best.pt"):
        try:
            payload = _torch_load(path)
        except (OSError, RuntimeError, ValueError):
            continue
        if _matches_experiment(payload, cfg, teacher_seed):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No validation-selected V14 teacher matches "
            f"{data_cfg['dataset_name']} {mask_cfg['pattern']}@"
            f"{float(mask_cfg['missing_rate']):g} seed{teacher_seed} under {search_root}"
        )
    return max(candidates, key=lambda item: item.stat().st_mtime).resolve()


def _initialize_student_backbone(
    student: torch.nn.Module,
    teacher_state: dict[str, torch.Tensor],
    strict: bool,
) -> tuple[int, int]:
    student_state = student.state_dict()
    target_prefix = "main_branch.student_backbone."
    source_prefix = "main_branch.main_backbone."
    target_names = [name for name in student_state if name.startswith(target_prefix)]
    copied = 0
    missing: list[str] = []
    for target_name in target_names:
        suffix = target_name[len(target_prefix):]
        source_name = source_prefix + suffix
        source = teacher_state.get(source_name)
        if source is None or source.shape != student_state[target_name].shape:
            missing.append(target_name)
            continue
        student_state[target_name] = source.detach().clone()
        copied += 1
    if strict and missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"V14-to-V16 backbone initialization missed {len(missing)} tensors: {preview}"
        )
    if copied == 0:
        raise RuntimeError("V14-to-V16 backbone initialization copied no tensors")
    student.load_state_dict(student_state, strict=True)
    return copied, len(target_names)


def prepare_v16_teacher(
    cfg: dict,
    student: torch.nn.Module,
    device: torch.device,
) -> TeacherContext:
    teacher_cfg = cfg.get("teacher", {})
    if not bool(teacher_cfg.get("enabled", False)):
        return TeacherContext(None, {"teacher_enabled": False})
    if teacher_cfg.get("architecture", "v14_safe_c2f_moe") != "v14_safe_c2f_moe":
        raise ValueError("V16 currently requires a v14_safe_c2f_moe teacher")

    checkpoint = resolve_teacher_checkpoint(cfg)
    payload = _torch_load(checkpoint)
    source_cfg = payload["config"]
    strict = bool(teacher_cfg.get("strict", True))
    if bool(teacher_cfg.get("initialize_student", True)):
        copied, total = _initialize_student_backbone(student, payload["model"], strict)
    else:
        copied, total = 0, 0

    # Imported lazily to avoid a model-registry import cycle.
    from .models import DualBranchSTImputer

    teacher = DualBranchSTImputer.from_config(source_cfg)
    teacher.load_state_dict(payload["model"], strict=strict)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    teacher_branch, teacher_commit = _teacher_git_metadata(checkpoint)
    metadata: dict[str, object] = {
        "teacher_enabled": True,
        "teacher_architecture": "v14_safe_c2f_moe",
        "teacher_branch": teacher_branch,
        "teacher_commit": teacher_commit,
        "teacher_checkpoint": str(checkpoint),
        "teacher_checkpoint_sha256": _sha256(checkpoint),
        "teacher_best_epoch": int(payload.get("epoch", -1)),
        "teacher_seed": int(source_cfg.get("seed", -1)),
        "student_backbone_tensors_initialized": copied,
        "student_backbone_tensors_total": total,
    }
    return TeacherContext(teacher, metadata)
