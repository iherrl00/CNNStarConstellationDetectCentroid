from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BrightAnnotation:
    name: str
    vmag: Optional[float]
    x: Optional[float]
    y: Optional[float]
    kind: str


@dataclass
class AstrometryViewData:
    status: str
    session: str
    subid: Optional[int]
    job_id: Optional[int]
    ra: Optional[float]
    dec: Optional[float]
    pixscale: Optional[float]
    orientation: Optional[float]
    parity: Optional[float]
    radius: Optional[float]
    width_arcsec: Optional[float]
    height_arcsec: Optional[float]
    constellations: List[str]
    objects: List[str]
    tags: List[str]
    stars: List[str]
    bright_annotations: List[BrightAnnotation]
    links: Dict[str, str]


def parse_astrometry_result(result: Dict[str, Any]) -> AstrometryViewData:
    payloads = _as_dict(result.get("api_payloads"))
    info_payload = _as_dict(payloads.get("info"))
    calibration_payload = _as_dict(payloads.get("calibration"))
    if not calibration_payload:
        calibration_payload = _as_dict(info_payload.get("calibration"))

    tags = _unique(_extract_list(payloads.get("tags"), "tags") + _extract_list(info_payload, "tags"))
    machine_tags = _extract_list(payloads.get("machine_tags"), "tags") + _extract_list(info_payload, "machine_tags")
    objects = _unique(_extract_list(payloads.get("objects_in_field"), "objects_in_field") + _extract_list(info_payload, "objects_in_field"))

    merged_tags = _unique(tags + machine_tags)
    constellation_items = _unique([item for item in merged_tags + objects if "constellation" in item.lower()])
    star_items = _unique([item for item in merged_tags + objects if "star" in item.lower()])

    annotations = _extract_annotations(payloads.get("annotations"))
    annotation_star_names = _unique([item.name for item in annotations if item.name])
    star_items = _unique(star_items + annotation_star_names)

    return AstrometryViewData(
        status=str(result.get("status", "unknown")),
        session=str(result.get("session", "")),
        subid=_to_int(result.get("subid")),
        job_id=_to_int(result.get("job_id")),
        ra=_to_float(calibration_payload.get("ra")),
        dec=_to_float(calibration_payload.get("dec")),
        pixscale=_to_float(calibration_payload.get("pixscale")),
        orientation=_to_float(calibration_payload.get("orientation")),
        parity=_to_float(calibration_payload.get("parity")),
        radius=_to_float(calibration_payload.get("radius")),
        width_arcsec=_to_float(calibration_payload.get("width_arcsec")),
        height_arcsec=_to_float(calibration_payload.get("height_arcsec")),
        constellations=constellation_items,
        objects=objects,
        tags=merged_tags,
        stars=star_items,
        bright_annotations=annotations,
        links=_extract_links(result.get("download_urls")),
    )


def _extract_annotations(payload: Any) -> List[BrightAnnotation]:
    content = _as_dict(payload)
    values = content.get("annotations")
    if not isinstance(values, list):
        return []
    items: List[BrightAnnotation] = []
    for row in values:
        row_dict = _as_dict(row)
        names = row_dict.get("names")
        name_list = [str(item).strip() for item in names if str(item).strip()] if isinstance(names, list) else []
        display_name = " / ".join(name_list[:2]) if name_list else "Sin nombre"
        items.append(
            BrightAnnotation(
                name=display_name,
                vmag=_to_float(row_dict.get("vmag")),
                x=_to_float(row_dict.get("pixelx")),
                y=_to_float(row_dict.get("pixely")),
                kind=str(row_dict.get("type", "")),
            )
        )
    items.sort(key=lambda item: item.vmag if item.vmag is not None else 99.0)
    return items


def _extract_links(payload: Any) -> Dict[str, str]:
    data = _as_dict(payload)
    output: Dict[str, str] = {}
    for key, value in data.items():
        text = str(value).strip()
        if text.startswith("http"):
            output[str(key)] = text
    return output


def _extract_list(payload: Any, key: str) -> List[str]:
    data = _as_dict(payload)
    return _flatten_strings(data.get(key))


def _flatten_strings(value: Any) -> List[str]:
    if isinstance(value, list):
        output: List[str] = []
        for item in value:
            output.extend(_flatten_strings(item))
        return output
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, dict):
        output: List[str] = []
        for inner in value.values():
            output.extend(_flatten_strings(inner))
        return output
    return []


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique(values: List[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in values:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output
