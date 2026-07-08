"""
Centroide de constelaciones: solo usa anotaciones reales + conversión WCS.
Sin catálogos hardcodeados.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .result_parser import AstrometryViewData, BrightAnnotation


@dataclass
class ConstellationTarget:
    name: str
    ra_deg: Optional[float]
    dec_deg: Optional[float]
    ra_hms: Optional[str]
    dec_dms: Optional[str]
    source: str   # "annotations+wcs" | "no_data" | "no_wcs"
    n_annotations: int = 0


def deg_to_hms(ra_deg: float) -> str:
    ra_deg = ra_deg % 360.0
    total_sec = ra_deg * 3600.0 / 15.0
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = total_sec % 60
    return f"{h:02d}h {m:02d}m {s:05.2f}s"


def deg_to_dms(dec_deg: float) -> str:
    sign = "+" if dec_deg >= 0 else "-"
    dec_deg = abs(dec_deg)
    d = int(dec_deg)
    m = int((dec_deg - d) * 60)
    s = ((dec_deg - d) * 60 - m) * 60
    return f"{sign}{d:02d}° {m:02d}' {s:05.2f}\""


def pixel_to_radec(
    px: float, py: float,
    img_w: float, img_h: float,
    ra_center: float, dec_center: float,
    pixscale_arcsec: float,
    orientation_deg: float,
    parity: float,
) -> Tuple[float, float]:
    """Convierte coordenadas de píxel a RA/Dec usando calibración WCS."""
    cx, cy = img_w / 2.0, img_h / 2.0
    dx = px - cx
    dy = cy - py  # flip Y

    scale = pixscale_arcsec / 3600.0  # grados/pixel
    theta = math.radians(orientation_deg)

    dra_deg  = scale * parity * (dx * math.cos(theta) + dy * math.sin(theta))
    ddec_deg = scale * (-dx * math.sin(theta) + dy * math.cos(theta))

    cos_dec = math.cos(math.radians(dec_center))
    ra_out  = ra_center + (dra_deg / cos_dec if cos_dec > 1e-8 else 0.0)
    dec_out = dec_center + ddec_deg

    ra_out = ra_out % 360.0
    dec_out = max(-90.0, min(90.0, dec_out))
    return ra_out, dec_out


def _normalize(name: str) -> str:
    return name.lower().strip()


def _annotation_matches(annotation_name: str, constellation_name: str) -> bool:
    """True si la anotación pertenece a la constelación seleccionada."""
    ann = _normalize(annotation_name)
    con = _normalize(constellation_name)
    # Extraer palabras significativas del nombre de constelación
    words = [w for w in con.split() if len(w) > 2 and w not in ("the", "con", "constellation")]
    return any(w in ann for w in words)


def centroid_from_annotations(
    annotations: List[BrightAnnotation],
    constellation_name: str,
) -> Tuple[Optional[Tuple[float, float]], int]:
    """
    Promedia los píxeles de anotaciones que correspondan a esa constelación.
    Devuelve (punto_px, n_coincidencias).
    """
    matching = [
        a for a in annotations
        if a.x is not None and a.y is not None
        and _annotation_matches(a.name, constellation_name)
    ]
    if not matching:
        return None, 0
    avg_x = sum(a.x for a in matching) / len(matching)
    avg_y = sum(a.y for a in matching) / len(matching)
    return (avg_x, avg_y), len(matching)


def get_target(
    constellation_name: str,
    data: AstrometryViewData,
    img_w: Optional[float] = None,
    img_h: Optional[float] = None,
) -> ConstellationTarget:
    """
    Devuelve el centroide RA/Dec calculado SOLO desde anotaciones reales + WCS.
    Si no hay datos suficientes, lo indica claramente sin inventar nada.
    """
    # Verificar que tenemos WCS completo
    wcs_ok = (
        data.ra is not None and
        data.dec is not None and
        data.pixscale is not None and
        data.orientation is not None and
        data.parity is not None and
        img_w is not None and
        img_h is not None
    )

    if not wcs_ok:
        return ConstellationTarget(
            name=constellation_name,
            ra_deg=None, dec_deg=None,
            ra_hms=None, dec_dms=None,
            source="no_wcs",
        )

    pix, n = centroid_from_annotations(data.bright_annotations, constellation_name)

    if pix is None:
    # Fallback: usar el centro de campo de la calibración WCS
        return ConstellationTarget(
            name=constellation_name,
            ra_deg=data.ra,
            dec_deg=data.dec,
            ra_hms=deg_to_hms(data.ra),
            dec_dms=deg_to_dms(data.dec),
            source="wcs_center_fallback",   # nuevo source
            n_annotations=0,
        )

    ra, dec = pixel_to_radec(
        pix[0], pix[1], img_w, img_h,
        data.ra, data.dec,
        data.pixscale, data.orientation, data.parity,
    )

    return ConstellationTarget(
        name=constellation_name,
        ra_deg=ra,
        dec_deg=dec,
        ra_hms=deg_to_hms(ra),
        dec_dms=deg_to_dms(dec),
        source="annotations+wcs",
        n_annotations=n,
    )

def get_star_target(
    star_name: str,
    data: AstrometryViewData,
    img_w: Optional[float] = None,
    img_h: Optional[float] = None,
) -> ConstellationTarget:
    wcs_ok = (
        data.ra is not None and data.dec is not None and
        data.pixscale is not None and data.orientation is not None and
        data.parity is not None and img_w is not None and img_h is not None
    )
    if not wcs_ok:
        return ConstellationTarget(name=star_name, ra_deg=None, dec_deg=None,
                                   ra_hms=None, dec_dms=None, source="no_wcs")
    match = next(
        (a for a in data.bright_annotations
         if a.x is not None and a.y is not None
         and (
            _normalize(star_name) in _normalize(a.name)
            or _normalize(a.name) in _normalize(star_name)
            or any(tok in _normalize(a.name) for tok in _normalize(star_name).split() if len(tok) > 2)
         )),
        None
    )
    
    if match is None:
        return ConstellationTarget(
            name=star_name,
            ra_deg=data.ra,
            dec_deg=data.dec,
            ra_hms=deg_to_hms(data.ra),
            dec_dms=deg_to_dms(data.dec),
            source="wcs_center_fallback",
            n_annotations=0,
        )
    
    ra, dec = pixel_to_radec(
        match.x, match.y, img_w, img_h,
        data.ra, data.dec, data.pixscale, data.orientation, data.parity,
    )
    
    return ConstellationTarget(name=star_name, ra_deg=ra, dec_deg=dec,
                               ra_hms=deg_to_hms(ra), dec_dms=deg_to_dms(dec),
                               source="annotations+wcs", n_annotations=1)