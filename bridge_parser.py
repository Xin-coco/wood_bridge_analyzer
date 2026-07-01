from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import math
import numpy as np

try:
    import rhino3dm
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("缺少 rhino3dm。请先运行: pip install -r requirements.txt") from exc


@dataclass
class Rod:
    id: int
    model_start: np.ndarray
    model_end: np.ndarray
    start: np.ndarray
    end: np.ndarray
    model_length_mm: float
    length_mm: float
    direction: np.ndarray
    section_width_mm: float
    section_height_mm: float
    layer: str
    source_type: str
    equivalent_standard_rods: int
    warnings: list[str]

    def to_record(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_start_x_mm"], data["model_start_y_mm"], data["model_start_z_mm"] = self.model_start.tolist()
        data["model_end_x_mm"], data["model_end_y_mm"], data["model_end_z_mm"] = self.model_end.tolist()
        data["start_x_mm"], data["start_y_mm"], data["start_z_mm"] = self.start.tolist()
        data["end_x_mm"], data["end_y_mm"], data["end_z_mm"] = self.end.tolist()
        data["dir_x"], data["dir_y"], data["dir_z"] = self.direction.tolist()
        data["required_standard_rods"] = self.equivalent_standard_rods
        data["real_length_mm"] = self.length_mm
        data.pop("start")
        data.pop("end")
        data.pop("model_start")
        data.pop("model_end")
        data.pop("direction")
        data["warnings"] = "; ".join(self.warnings)
        return data


@dataclass
class ModelMetadata:
    rhino_unit: str
    scale: float
    object_count: int
    model_bbox_min: np.ndarray
    model_bbox_max: np.ndarray
    real_bbox_min: np.ndarray
    real_bbox_max: np.ndarray
    detected_standard_1_to_10: bool
    messages: list[str]

    def dimensions_record(self) -> dict[str, float | str | bool]:
        model_dims = self.model_bbox_max - self.model_bbox_min
        real_dims = self.real_bbox_max - self.real_bbox_min
        return {
            "rhino_unit": self.rhino_unit,
            "scale": self.scale,
            "model_length_x_mm": float(model_dims[0]),
            "model_width_y_mm": float(model_dims[1]),
            "model_height_z_mm": float(model_dims[2]),
            "real_length_x_mm": float(real_dims[0]),
            "real_width_y_mm": float(real_dims[1]),
            "real_height_z_mm": float(real_dims[2]),
            "detected_standard_1_to_10": self.detected_standard_1_to_10,
        }


def _point_to_np(point: Any) -> np.ndarray:
    return np.array([float(point.X), float(point.Y), float(point.Z)], dtype=float)


def _bbox_points(bbox: Any) -> list[np.ndarray]:
    return [
        np.array([bbox.Min.X, bbox.Min.Y, bbox.Min.Z], dtype=float),
        np.array([bbox.Min.X, bbox.Min.Y, bbox.Max.Z], dtype=float),
        np.array([bbox.Min.X, bbox.Max.Y, bbox.Min.Z], dtype=float),
        np.array([bbox.Min.X, bbox.Max.Y, bbox.Max.Z], dtype=float),
        np.array([bbox.Max.X, bbox.Min.Y, bbox.Min.Z], dtype=float),
        np.array([bbox.Max.X, bbox.Min.Y, bbox.Max.Z], dtype=float),
        np.array([bbox.Max.X, bbox.Max.Y, bbox.Min.Z], dtype=float),
        np.array([bbox.Max.X, bbox.Max.Y, bbox.Max.Z], dtype=float),
    ]


def _axis_from_bbox(bbox: Any) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    pts = _bbox_points(bbox)
    spans = np.array(
        [
            bbox.Max.X - bbox.Min.X,
            bbox.Max.Y - bbox.Min.Y,
            bbox.Max.Z - bbox.Min.Z,
        ],
        dtype=float,
    )
    axis = int(np.argmax(spans))
    warnings: list[str] = []
    if spans[axis] <= 0:
        raise ValueError("几何包围盒长度为 0，无法识别为杆件")
    center = np.mean(pts, axis=0)
    start = center.copy()
    end = center.copy()
    start[axis] = [bbox.Min.X, bbox.Min.Y, bbox.Min.Z][axis]
    end[axis] = [bbox.Max.X, bbox.Max.Y, bbox.Max.Z][axis]
    cross_dims = [spans[i] for i in range(3) if i != axis]
    if min(cross_dims) <= 0:
        warnings.append("包围盒截面尺寸异常")
    return start, end, float(spans[axis]), warnings


def _line_curve_endpoints(curve: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if curve is None:
        return None
    if curve.ObjectType == rhino3dm.ObjectType.Curve:
        try:
            start = curve.PointAtStart
            end = curve.PointAtEnd
            return _point_to_np(start), _point_to_np(end)
        except Exception:
            return None
    return None


def _get_layer_name(model: Any, attributes: Any) -> str:
    try:
        index = attributes.LayerIndex
        if index >= 0 and index < len(model.Layers):
            return model.Layers[index].Name or "Default"
    except Exception:
        pass
    return "Default"


def _looks_like_rod(layer: str, keywords: Iterable[str]) -> bool:
    if not keywords:
        return True
    lowered = layer.lower()
    return any(k.lower() in lowered for k in keywords)


def _model_unit_name(model: Any) -> str:
    try:
        return str(model.Settings.ModelUnitSystem).replace("UnitSystem.", "")
    except Exception:
        return "Unknown"


def _standard_1_to_10_detected(rods: list[Rod], config: dict[str, Any]) -> bool:
    if not rods:
        return False
    lengths = np.array([r.model_length_mm for r in rods], dtype=float)
    standard_model_len = float(config["section"]["standard_length_mm"]) / float(config["model"].get("scale", 10.0))
    near_length = np.mean(np.abs(lengths - standard_model_len) <= max(8.0, standard_model_len * 0.12))
    return bool(near_length >= 0.05 or np.any(np.abs(lengths - standard_model_len) <= max(8.0, standard_model_len * 0.12)))


def parse_3dm_model(model_path: str | Path, config: dict[str, Any]) -> tuple[list[Rod], list[str], ModelMetadata]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到模型文件: {path}")
    model = rhino3dm.File3dm.Read(str(path))
    if model is None:
        raise RuntimeError(f"Rhino 文件读取失败: {path}")

    scale = float(config["model"].get("scale", 10.0))
    section = config["section"]
    standard_len = float(section["standard_length_mm"])
    min_len_model = float(config["model"].get("min_rod_length_model_mm", 20.0))
    keywords = config["model"].get("rod_layer_keywords", [])

    rods: list[Rod] = []
    skipped: list[str] = []
    metadata_messages: list[str] = []
    rod_id = 1

    for obj_index, obj in enumerate(model.Objects):
        geom = obj.Geometry
        layer = _get_layer_name(model, obj.Attributes)
        source_type = type(geom).__name__
        if keywords and not _looks_like_rod(layer, keywords):
            skipped.append(f"对象 {obj_index}: 图层 '{layer}' 未匹配木杆关键词，已跳过")
            continue

        start_model = end_model = None
        warnings: list[str] = []
        endpoints = _line_curve_endpoints(geom)
        if endpoints:
            start_model, end_model = endpoints
        else:
            try:
                bbox = geom.GetBoundingBox()
                start_model, end_model, _, warnings = _axis_from_bbox(bbox)
            except Exception as exc:
                skipped.append(f"对象 {obj_index}: {source_type} 无法提取中心线: {exc}")
                continue

        model_vec = end_model - start_model
        model_length = float(np.linalg.norm(model_vec))
        start = start_model * scale
        end = end_model * scale
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length < min_len_model * scale:
            skipped.append(f"对象 {obj_index}: 长度 {length:.1f} mm 过短，已跳过")
            continue
        direction = vec / length
        equivalent = int(math.ceil(length / standard_len))
        rods.append(
            Rod(
                id=rod_id,
                model_start=start_model,
                model_end=end_model,
                start=start,
                end=end,
                model_length_mm=model_length,
                length_mm=length,
                direction=direction,
                section_width_mm=float(section["width_mm"]),
                section_height_mm=float(section["height_mm"]),
                layer=layer,
                source_type=source_type,
                equivalent_standard_rods=equivalent,
                warnings=warnings,
            )
        )
        rod_id += 1

    if not rods:
        raise RuntimeError(
            "未识别到木杆。请检查图层关键词、模型单位，或把木杆放在包含 wood/rod/member/timber/木/杆 的图层。"
        )
    all_model_points = np.vstack([np.vstack([r.model_start, r.model_end]) for r in rods])
    all_real_points = np.vstack([np.vstack([r.start, r.end]) for r in rods])
    detected_standard = _standard_1_to_10_detected(rods, config)
    if detected_standard:
        metadata_messages.append("检测到 1:10 标准木杆模型：杆件长度接近 3×8×130mm 标准小模型杆。")
    metadata = ModelMetadata(
        rhino_unit=_model_unit_name(model),
        scale=scale,
        object_count=len(model.Objects),
        model_bbox_min=np.min(all_model_points, axis=0),
        model_bbox_max=np.max(all_model_points, axis=0),
        real_bbox_min=np.min(all_real_points, axis=0),
        real_bbox_max=np.max(all_real_points, axis=0),
        detected_standard_1_to_10=detected_standard,
        messages=metadata_messages,
    )
    return rods, skipped, metadata
