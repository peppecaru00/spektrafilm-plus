from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import colour
import exiv2
import lensfunpy
import numpy as np
import rawpy
from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)

_TUNGSTEN_TEMPERATURE = 2850.0
_DAYLIGHT_REFERENCE_TEMPERATURE = 6504.0
_ACES_COLOURSPACE = colour.RGB_COLOURSPACES["ACES2065-1"]


@dataclass(frozen=True, slots=True)
class ExifData:
    make: str
    model: str
    lens_make: str
    lens_model: str
    focal_length: float
    f_number: float


def _whitepoint_xyz_from_temperature(temperature: float) -> np.ndarray:
    """Convert a colour temperature to an XYZ whitepoint.

    Daylight whitepoints above 4000 K are better approximated by the CIE
    daylight locus than by a pure Planckian radiator. Warmer illuminants such as
    tungsten are modelled with the Kang 2002 Planckian approximation.
    """

    method = 'CIE Illuminant D Series' if temperature >= 4000.0 else 'Kang 2002'
    xy = colour.CCT_to_xy(np.float64(temperature), method=method)
    return np.asarray(colour.xy_to_XYZ(xy), dtype=np.float64)


def _apply_white_balance_adaptation(
    rgb: np.ndarray,
    source_white_xyz: np.ndarray,
    target_white_xyz: np.ndarray,
) -> np.ndarray:
    """Apply a colour-science chromatic adaptation in linear ACES RGB."""

    source_white_xyz = np.asarray(source_white_xyz, dtype=np.float64)
    target_white_xyz = np.asarray(target_white_xyz, dtype=np.float64)
    source_white_xyz = source_white_xyz / source_white_xyz[1]
    target_white_xyz = target_white_xyz / target_white_xyz[1]

    xyz = colour.RGB_to_XYZ(
        rgb,
        colourspace=_ACES_COLOURSPACE,
        chromatic_adaptation_transform=None,
        apply_cctf_decoding=False,
    )
    xyz = colour.chromatic_adaptation(
        xyz,
        source_white_xyz,
        target_white_xyz,
        method='Von Kries',
    )
    return colour.XYZ_to_RGB(
        xyz,
        colourspace=_ACES_COLOURSPACE,
        chromatic_adaptation_transform=None,
        apply_cctf_encoding=False,
    ).astype(np.float32)


def _apply_tint_adjustment(rgb: np.ndarray, tint: float | None) -> np.ndarray:
    """Apply a simple green-magenta tint adjustment in linear ACES RGB."""

    if tint is None or np.isclose(tint, 1.0):
        return rgb

    tint_scale = np.array([1.0, float(tint), 1.0], dtype=np.float32)
    return (rgb * tint_scale).astype(np.float32)


def _map_highlight_mode(mode: str | int | object) -> object:
    """Map string or int highlight mode representation to rawpy.HighlightMode."""
    hl_mode = getattr(rawpy, 'HighlightMode', None)
    if hl_mode is not None:
        if isinstance(mode, hl_mode):
            return mode
        m = str(mode).lower().strip()
        if m in ('clip', '0'):
            return hl_mode.Clip
        if m in ('ignore', '1', 'unclip'):
            return hl_mode.Ignore
        if m in ('blend', '2'):
            return hl_mode.Blend
        if m in ('reconstruct', '5', 'reconstructdefault'):
            return hl_mode.ReconstructDefault
    if isinstance(mode, int):
        return mode
    return 5 if hl_mode is None else getattr(hl_mode, 'ReconstructDefault', 5)


def _postprocess_params(
    white_balance: str | tuple[float, float],
    temperature: float | None,
    tint: float | None,
    chroma_denoise: bool = False,
    chroma_denoise_strength: float = 1.0,
    raw_exposure_ev: float = 0.0,
    highlight_recovery: float = 0.0,
    highlight_mode: str | int = "reconstruct",
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray] | None, float | None]:
    """Build the ``rawpy.postprocess`` parameters for the requested settings.

    The output is always configured as linear 16-bit ACES RGB. White balance is
    handled in one of two ways:

    - ``'as_shot'`` uses LibRaw camera white balance directly during demosaic.
    - Other modes use LibRaw's daylight-balanced default output as the base and
      apply colour-science chromatic adaptation in linear ACES RGB.
    """

    params: dict[str, object] = {
        'output_color': getattr(rawpy, 'ColorSpace').ACES,
        'output_bps': 16,
        'no_auto_bright': True,
        'gamma': (1, 1),
    }

    hl_val = _map_highlight_mode(highlight_mode)
    if hl_val is not None:
        params['highlight_mode'] = hl_val

    if raw_exposure_ev != 0.0:
        params['exp_shift'] = float(2.0 ** raw_exposure_ev)

    if highlight_recovery > 0.0:
        params['exp_preserve_highlights'] = float(np.clip(highlight_recovery, 0.0, 1.0))
        if 'exp_shift' not in params:
            params['exp_shift'] = 1.0

    if chroma_denoise and chroma_denoise_strength > 0.0:
        passes = min(max(int(round(float(chroma_denoise_strength) * 2.0)), 1), 3)
        params['median_filter_passes'] = passes
        fbdd_mode = getattr(rawpy, 'FBDDNoiseReductionMode', None)
        if fbdd_mode is not None:
            if chroma_denoise_strength >= 1.2:
                params['fbdd_noise_reduction'] = fbdd_mode.Full
            elif chroma_denoise_strength >= 0.5:
                params['fbdd_noise_reduction'] = fbdd_mode.Light
    postprocess_adaptation: tuple[np.ndarray, np.ndarray] | None = None
    tint_multiplier: float | None = None
    reference_white_xyz = _whitepoint_xyz_from_temperature(_DAYLIGHT_REFERENCE_TEMPERATURE)

    def set_colour_science_adjustment(target_temperature: float, target_tint: float | None) -> None:
        nonlocal postprocess_adaptation, tint_multiplier

        scene_white_xyz = _whitepoint_xyz_from_temperature(target_temperature)
        if not np.allclose(reference_white_xyz, scene_white_xyz):
            postprocess_adaptation = (scene_white_xyz, reference_white_xyz)
        tint_multiplier = target_tint

    if white_balance == 'as_shot':
        params['use_camera_wb'] = True
    elif white_balance == 'daylight':
        pass
    elif white_balance == 'tungsten':
        set_colour_science_adjustment(_TUNGSTEN_TEMPERATURE, 1.0)
    elif white_balance == 'custom':
        if temperature is None:
            raise ValueError('A custom raw white balance requires a temperature value.')
        set_colour_science_adjustment(temperature, tint)
    else:
        custom_temperature, custom_tint = white_balance
        set_colour_science_adjustment(custom_temperature, custom_tint)

    return params, postprocess_adaptation, tint_multiplier


def _read_exif_with_tifffile(raw_path: str | PathLike[str]) -> ExifData | None:
    """Fallback EXIF extraction using tifffile for TIFF-based RAW containers."""
    try:
        import tifffile
        with tifffile.TiffFile(str(raw_path)) as tif:
            for page in tif.pages:
                tags = page.tags
                make = str(tags.get(271).value if 271 in tags else "").strip()
                model = str(tags.get(272).value if 272 in tags else "").strip()
                lens_make = str(tags.get(42035).value if 42035 in tags else "").strip()
                lens_model = str(tags.get(42036).value if 42036 in tags else "").strip()

                focal_length = 0.0
                if 37386 in tags:
                    val = tags[37386].value
                    try:
                        focal_length = float(val[0] / val[1]) if isinstance(val, (tuple, list)) else float(val)
                    except Exception:
                        pass

                f_number = 0.0
                if 33437 in tags:
                    val = tags[33437].value
                    try:
                        f_number = float(val[0] / val[1]) if isinstance(val, (tuple, list)) else float(val)
                    except Exception:
                        pass

                if model or lens_model:
                    return ExifData(
                        make=make,
                        model=model,
                        lens_make=lens_make,
                        lens_model=lens_model,
                        focal_length=focal_length,
                        f_number=f_number,
                    )
    except Exception:
        pass
    return None


_COLOR_TEMP_EXIF_KEYS = (
    "Exif.Photo.ColorTemperature",
    "Exif.CanonSi.ColorTemperature",
    "Exif.CanonFi.ColorTemperature",
    "Exif.NikonWt.ColorTemperature",
    "Exif.Nikon3.ColorTemperature",
    "Exif.Sony1.ColorTemperature",
    "Exif.Sony2.ColorTemperature",
    "Exif.Panasonic.ColorTempKelvin",
    "Exif.Olympus.ColorTemperature",
    "Exif.Fuji.ColorTemperature",
    "Exif.Minolta.ColorTemperature",
)

_COLOR_TEMP_XMP_KEYS = (
    "Xmp.crs.Temperature",
    "Xmp.photoshop.ColorTemperature",
)


def _parse_exif_numeric(value: Any) -> float | None:
    try:
        s = str(value).strip()
        if "/" in s:
            num, denom = s.split("/", 1)
            d = float(denom)
            return float(num) / d if d != 0 else None
        return float(s)
    except Exception:
        return None


def _cct_tint_from_wb_multipliers(
    cam_wb: list[float] | tuple[float, ...] | np.ndarray,
    day_wb: list[float] | tuple[float, ...] | np.ndarray | None = None,
    rgb_xyz: np.ndarray | None = None,
) -> tuple[float, float]:
    """Estimate correlated colour temperature and tint from camera WB multipliers."""
    if not cam_wb or len(cam_wb) < 3:
        return 5500.0, 1.0

    r_cam, g_cam, b_cam = float(cam_wb[0]), float(cam_wb[1]), float(cam_wb[2])
    if r_cam <= 0.0 or g_cam <= 0.0 or b_cam <= 0.0:
        return 5500.0, 1.0

    # Camera sensor response to neutral illuminant:
    # Under the scene illuminant, a neutral surface satisfies:
    # R_sensor * r_cam = G_sensor * g_cam = B_sensor * b_cam
    # => R_sensor / G_sensor = g_cam / r_cam, B_sensor / G_sensor = g_cam / b_cam
    sensor_rgb = np.array([g_cam / r_cam, 1.0, g_cam / b_cam], dtype=np.float64)

    # Convert Camera RGB sensor response to CIE XYZ:
    # In LibRaw, rgb_xyz_matrix is (3,3) or (4,3) mapping CIE XYZ under D65 to Camera RGB.
    # Therefore, Camera RGB to XYZ is the inverse of rgb_xyz[:3, :3].
    xyz = None
    if rgb_xyz is not None:
        try:
            arr = np.asarray(rgb_xyz, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 3:
                sub = arr[:3, :3]
                # If sub is not a dummy identity matrix and is invertible:
                if not np.allclose(sub, np.eye(3)) and abs(np.linalg.det(sub)) > 1e-5:
                    cam_to_xyz = np.linalg.inv(sub)
                    xyz = cam_to_xyz @ sensor_rgb
        except Exception:
            pass

    if xyz is None or np.any(np.isnan(xyz)) or np.all(xyz <= 0):
        # Fallback when no valid camera matrix is available:
        if day_wb is not None and len(day_wb) >= 3 and day_wb[0] > 0 and day_wb[1] > 0 and day_wb[2] > 0:
            # Multipliers relative to daylight D65:
            r_rel = (float(day_wb[0]) / float(day_wb[1])) / (r_cam / g_cam)
            b_rel = (float(day_wb[2]) / float(day_wb[1])) / (b_cam / g_cam)
            xyz = np.array([r_rel * 0.95047, 1.0, b_rel * 1.08883], dtype=np.float64)
        else:
            mat_srgb = np.array(
                [
                    [0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041],
                ],
                dtype=np.float64,
            )
            xyz = mat_srgb @ sensor_rgb

    xyz = np.maximum(xyz, 1e-6)
    if xyz[1] <= 0:
        return 5500.0, 1.0
    xyz_norm = xyz / xyz[1]
    xy = colour.XYZ_to_xy(xyz_norm)

    cct = None
    for method in ("Hernandez 1999", "Kang 2002", "McCamy 1992"):
        try:
            val = float(colour.xy_to_CCT(xy, method=method))
            if 1500.0 <= val <= 25000.0 and not np.isnan(val):
                cct = val
                break
        except Exception:
            pass

    if cct is None:
        cct = 5500.0

    locus_xyz = _whitepoint_xyz_from_temperature(cct)
    locus_norm = locus_xyz / locus_xyz[1]
    aces_scene = colour.XYZ_to_RGB(
        xyz_norm,
        colourspace=_ACES_COLOURSPACE,
        apply_cctf_encoding=False,
    )
    aces_locus = colour.XYZ_to_RGB(
        locus_norm,
        colourspace=_ACES_COLOURSPACE,
        apply_cctf_encoding=False,
    )

    tint = 1.0
    if aces_locus[1] > 0 and aces_scene[1] > 0:
        tint = float(aces_scene[1] / aces_locus[1])
    if not (0.2 <= tint <= 5.0) or np.isnan(tint):
        tint = 1.0

    return round(float(cct), 0), round(float(tint), 2)


def _open_exiv2_image(raw_path: str | PathLike[str]) -> Any:
    """Open an image using exiv2 with fallback to Windows 8.3 short paths if needed."""
    p_str = str(raw_path)
    try:
        return exiv2.ImageFactory.open(p_str)
    except Exception:
        if sys.platform == "win32":
            try:
                import ctypes
                buf = ctypes.create_unicode_buffer(1024)
                if ctypes.windll.kernel32.GetShortPathNameW(p_str, buf, 1024):
                    short_path = buf.value
                    if short_path:
                        return exiv2.ImageFactory.open(short_path)
            except Exception:
                pass
        raise


def _safe_get_exif_dict(exif: Any) -> dict[str, Any]:
    """Safely extract tags from exiv2 ExifData or mock dict into a pure Python dict.

    Avoids querying non-existent keys directly via `key in exif`, which triggers
    an uncatchable C++ abort/assertion in python-exiv2's findKey binding.
    """
    exif_dict: dict[str, Any] = {}
    if exif is None:
        return exif_dict
    if isinstance(exif, dict):
        return exif
    try:
        for datum in exif:
            try:
                if hasattr(datum, "key"):
                    exif_dict[str(datum.key())] = datum
                elif isinstance(datum, (tuple, list)) and len(datum) == 2:
                    exif_dict[str(datum[0])] = datum[1]
                else:
                    exif_dict[str(datum)] = datum
            except Exception:
                pass
    except Exception:
        pass
    return exif_dict


def _safe_get_xmp_dict(xmp: Any) -> dict[str, Any]:
    """Safely extract tags from exiv2 XmpData or mock dict into a pure Python dict."""
    xmp_dict: dict[str, Any] = {}
    if xmp is None:
        return xmp_dict
    if isinstance(xmp, dict):
        return xmp
    try:
        for datum in xmp:
            try:
                if hasattr(datum, "key"):
                    xmp_dict[str(datum.key())] = datum
                elif isinstance(datum, (tuple, list)) and len(datum) == 2:
                    xmp_dict[str(datum[0])] = datum[1]
                else:
                    xmp_dict[str(datum)] = datum
            except Exception:
                pass
    except Exception:
        pass
    return xmp_dict


def read_raw_white_balance(raw_path: str | PathLike[str]) -> tuple[float, float]:
    """Read the actual as-shot white balance (temperature in Kelvin, tint) from a RAW file.

    Extracts the recorded Correlated Color Temperature (CCT) and green-magenta tint
    using EXIF/XMP metadata tags (Canon, Nikon, Sony, Fuji, Panasonic, Olympus, DNG)
    or calculating from LibRaw camera white-balance multipliers and color matrix.

    Returns
    -------
    tuple[float, float]
        (temperature_kelvin, tint)
        Defaults to (5500.0, 1.0) if no metadata could be determined.
    """
    raw_cct: float | None = None
    raw_tint: float | None = None
    exif_cct: float | None = None
    exif_tint: float | None = None

    target_path = str(raw_path)
    p = Path(raw_path)
    if p.suffix.lower() != ".dng":
        for cand in (p.parent / f"{p.stem}.dng", p.parent / f"{p.stem}.DNG"):
            if cand.is_file() and cand.stat().st_size > 0:
                target_path = str(cand)
                break

    try:
        with rawpy.imread(target_path) as raw:
            cam_wb = getattr(raw, "camera_whitebalance", None)
            day_wb = getattr(raw, "daylight_whitebalance", None)
            rgb_xyz = getattr(raw, "rgb_xyz_matrix", None)
            if cam_wb and len(cam_wb) >= 3:
                cct, tint = _cct_tint_from_wb_multipliers(cam_wb, day_wb, rgb_xyz)
                if 1500.0 <= cct <= 25000.0:
                    raw_cct, raw_tint = cct, tint
    except Exception:
        pass

    try:
        image = _open_exiv2_image(raw_path)
        image.readMetadata()
        exif = image.exifData()
        xmp = image.xmpData() if hasattr(image, "xmpData") else None

        exif_dict = _safe_get_exif_dict(exif)
        xmp_dict = _safe_get_xmp_dict(xmp)

        if exif_dict:
            for tag in _COLOR_TEMP_EXIF_KEYS:
                if tag in exif_dict:
                    datum = exif_dict[tag]
                    val = _parse_exif_numeric(datum.value() if hasattr(datum, "value") else datum)
                    if val is not None and 1500.0 <= val <= 25000.0:
                        exif_cct = val
                        break

        if exif_cct is None and xmp_dict:
            for tag in _COLOR_TEMP_XMP_KEYS:
                if tag in xmp_dict:
                    datum = xmp_dict[tag]
                    val = _parse_exif_numeric(datum.value() if hasattr(datum, "value") else datum)
                    if val is not None and 1500.0 <= val <= 25000.0:
                        exif_cct = val
                        break

        if exif_cct is None and exif_dict:
            for tag in (
                "Exif.Nikon3.WB_RBLevels",
                "Exif.Nikon1.WB_RBLevels",
                "Exif.Nikon2.WB_RBLevels",
                "Exif.Sony1.WB_RBLevels",
                "Exif.Panasonic.WB_RBLevels",
                "Exif.Olympus.WB_RBLevels",
            ):
                if tag in exif_dict:
                    val_str = str(exif_dict[tag].value() if hasattr(exif_dict[tag], "value") else exif_dict[tag])
                    parts = [_parse_exif_numeric(p) for p in val_str.split()]
                    if len(parts) >= 3 and all(p is not None and p > 0 for p in parts[:3]):
                        r, b, g = parts[0], parts[1], parts[2]
                        cct_est, tint_est = _cct_tint_from_wb_multipliers([r / g, 1.0, b / g])
                        if 1500.0 <= cct_est <= 25000.0:
                            exif_cct = cct_est
                            if exif_tint is None:
                                exif_tint = tint_est
                            break

            if exif_cct is None:
                for tag in ("Exif.Image.AsShotNeutral", "Exif.Photo.AsShotNeutral", "Xmp.crs.AsShotNeutral"):
                    if tag in exif_dict or (xmp_dict and tag in xmp_dict):
                        datum = exif_dict.get(tag) or (xmp_dict.get(tag) if xmp_dict else None)
                        val_str = str(datum.value() if hasattr(datum, "value") else datum)
                        parts = [_parse_exif_numeric(p) for p in val_str.split()]
                        if len(parts) >= 3 and all(p is not None and p > 0 for p in parts[:3]):
                            r_neut, g_neut, b_neut = parts[0], parts[1], parts[2]
                            cam_wb = [1.0 / r_neut, 1.0 / g_neut, 1.0 / b_neut]
                            cct_est, tint_est = _cct_tint_from_wb_multipliers(cam_wb)
                            if 1500.0 <= cct_est <= 25000.0:
                                exif_cct = cct_est
                                if exif_tint is None:
                                    exif_tint = tint_est
                                break

        if "Xmp.crs.Tint" in xmp_dict:
            datum = xmp_dict["Xmp.crs.Tint"]
            val = _parse_exif_numeric(datum.value() if hasattr(datum, "value") else datum)
            if val is not None and -150.0 <= val <= 150.0:
                exif_tint = round(1.0 + (val / 100.0), 2)
        elif exif_dict:
            if "Exif.Sony2.WBShiftABGM" in exif_dict:
                v_str = str(exif_dict["Exif.Sony2.WBShiftABGM"].value() if hasattr(exif_dict["Exif.Sony2.WBShiftABGM"], "value") else exif_dict["Exif.Sony2.WBShiftABGM"])
                parts = [_parse_exif_numeric(p) for p in v_str.split()]
                if len(parts) >= 2 and parts[1] is not None and parts[1] != 0:
                    exif_tint = round(1.0 + (parts[1] * 0.025), 2)
            elif "Exif.Panasonic.WBAdjustGM" in exif_dict:
                v = _parse_exif_numeric(exif_dict["Exif.Panasonic.WBAdjustGM"].value() if hasattr(exif_dict["Exif.Panasonic.WBAdjustGM"], "value") else exif_dict["Exif.Panasonic.WBAdjustGM"])
                if v is not None and v != 0:
                    exif_tint = round(1.0 + (v * 0.025), 2)
    except Exception:
        pass

    temp = exif_cct if exif_cct is not None else (raw_cct if raw_cct is not None else 5500.0)
    tint = exif_tint if exif_tint is not None else (raw_tint if raw_tint is not None else 1.0)

    return float(round(temp, 1)), float(round(tint, 2))


def _read_exif_metadata(raw_path: str | PathLike[str]) -> ExifData:
    """Read basic EXIF data from a RAW file for lens correction.

    The function uses ``exiv2`` to extract standard EXIF tags for camera make
    and model, lens make and model, as well as focal length and aperture.

    Parameters
    ----------
    raw_path
        Path to the RAW image file.

    Returns:
    -------
    ExifData
        Frozen dataclass with camera and lens metadata. Fields that could not be
        read default to ``""`` for strings and ``0.0`` for numeric values.

    """
    exif_dict: dict[str, Any] = {}
    xmp_dict: dict[str, Any] = {}
    exif = None
    try:
        image = _open_exiv2_image(raw_path)
        image.readMetadata()
        exif = image.exifData()
        xmp = image.xmpData() if hasattr(image, "xmpData") else None
        exif_dict = _safe_get_exif_dict(exif)
        xmp_dict = _safe_get_xmp_dict(xmp)
    except Exception:
        fallback = _read_exif_with_tifffile(raw_path)
        if fallback is not None:
            return fallback
        return ExifData(
            make="",
            model="",
            lens_make="",
            lens_model="",
            focal_length=0.0,
            f_number=0.0,
        )

    def _str_exif(key: str) -> str:
        datum = exif_dict.get(key)
        if datum is not None:
            try:
                if hasattr(datum, "value"):
                    return str(datum.value()).strip()
                return str(datum).strip()
            except Exception:
                try:
                    return str(datum).strip()
                except Exception:
                    pass
        return ""

    def _str_xmp(key: str) -> str:
        datum = xmp_dict.get(key)
        if datum is not None:
            try:
                if hasattr(datum, "value"):
                    return str(datum.value()).strip()
                return str(datum).strip()
            except Exception:
                try:
                    return str(datum).strip()
                except Exception:
                    pass
        return ""

    def _get_first(exif_keys: tuple[str, ...], xmp_keys: tuple[str, ...] = ()) -> str:
        for k in exif_keys:
            val = _str_exif(k)
            if val:
                return val
        for k in xmp_keys:
            val = _str_xmp(k)
            if val:
                return val
        return ""

    def _float_exif(key: str) -> float:
        datum = exif_dict.get(key)
        if datum is not None:
            try:
                if hasattr(datum, "toFloat"):
                    return float(datum.toFloat())
                val = _parse_exif_numeric(datum.value() if hasattr(datum, "value") else datum)
                if val is not None:
                    return float(val)
            except Exception:
                pass
        return 0.0

    def _float_xmp(key: str) -> float:
        datum = xmp_dict.get(key)
        if datum is not None:
            try:
                val = _parse_exif_numeric(datum.value() if hasattr(datum, "value") else datum)
                if val is not None:
                    return float(val)
            except Exception:
                pass
        return 0.0

    # Camera Make
    make = ""
    try:
        if exif is not None and hasattr(exiv2, "make"):
            make_datum = exiv2.make(exif)
            if make_datum is not None:
                make = str(make_datum.value() if hasattr(make_datum, "value") else make_datum).strip()
    except Exception:
        pass
    if not make:
        make = _get_first(
            ("Exif.Image.Make", "Exif.Photo.Make", "Exif.Panasonic.Make"),
            ("Xmp.tiff.Make",),
        )

    # Camera Model
    model = ""
    try:
        if exif is not None and hasattr(exiv2, "model"):
            model_datum = exiv2.model(exif)
            if model_datum is not None:
                model = str(model_datum.value() if hasattr(model_datum, "value") else model_datum).strip()
    except Exception:
        pass
    if not model:
        model = _get_first(
            ("Exif.Image.Model", "Exif.Photo.Model", "Exif.Panasonic.Model"),
            ("Xmp.tiff.Model",),
        )

    def _is_valid_lens_name(name: str) -> bool:
        s = name.strip()
        if not s or len(s) < 2:
            return False
        if s.isdigit():
            return False
        if not any(c.isalpha() for c in s):
            return False
        if s.lower() in {"manual", "unknown", "n/a", "none", "null"}:
            return False
        return True

    # Lens Model: try exiv2's built-in vendor-aware easyaccess helper first
    lens_model = ""
    try:
        if exif is not None and hasattr(exiv2, "lensName"):
            lens_datum = exiv2.lensName(exif)
            if lens_datum is not None:
                val = str(lens_datum.value() if hasattr(lens_datum, "value") else lens_datum).strip()
                if _is_valid_lens_name(val):
                    lens_model = val
    except Exception:
        pass

    # If easyaccess helper did not find a lens, probe vendor maker notes & XMP
    if not lens_model:
        lens_model = _get_first(
            (
                "Exif.Photo.LensModel",
                "Exif.Canon.LensModel",
                "Exif.Nikon3.Lens",
                "Exif.NikonLd3.Lens",
                "Exif.Sony2.LensModel",
                "Exif.Sony1.LensSpec",
                "Exif.Fujifilm.LensModel",
                "Exif.Fujifilm.LensType",
                "Exif.Panasonic.LensModel",
                "Exif.Panasonic.LensType",
                "Exif.OlympusEq.LensModel",
                "Exif.OlympusEq.LensType",
                "Exif.Pentax.LensType",
                "Exif.Photo.LensSpecification",
            ),
            (
                "Xmp.aux.Lens",
                "Xmp.exif.LensModel",
                "Xmp.aux.LensInfo",
            ),
        )
        if not _is_valid_lens_name(lens_model):
            lens_model = ""

    # Lens Make
    lens_make = _get_first(
        (
            "Exif.Photo.LensMake",
            "Exif.Image.Make",
        ),
        (
            "Xmp.aux.LensMake",
        ),
    )
    if not lens_make:
        lens_make = make

    # Focal Length
    focal_length = 0.0
    try:
        if exif is not None and hasattr(exiv2, "focalLength"):
            fl_datum = exiv2.focalLength(exif)
            if fl_datum is not None:
                if hasattr(fl_datum, "toFloat"):
                    focal_length = float(fl_datum.toFloat())
                else:
                    parsed = _parse_exif_numeric(fl_datum.value() if hasattr(fl_datum, "value") else fl_datum)
                    if parsed is not None:
                        focal_length = float(parsed)
    except Exception:
        pass
    if focal_length <= 0.0:
        for k in ("Exif.Photo.FocalLength", "Exif.Photo.FocalLengthIn35mmFilm"):
            fl = _float_exif(k)
            if fl > 0.0:
                focal_length = fl
                break
    if focal_length <= 0.0:
        focal_length = _float_xmp("Xmp.exif.FocalLength")

    # Aperture / F-Number
    f_number = 0.0
    try:
        if exif is not None and hasattr(exiv2, "fNumber"):
            fn_datum = exiv2.fNumber(exif)
            if fn_datum is not None:
                if hasattr(fn_datum, "toFloat"):
                    f_number = float(fn_datum.toFloat())
                else:
                    parsed = _parse_exif_numeric(fn_datum.value() if hasattr(fn_datum, "value") else fn_datum)
                    if parsed is not None:
                        f_number = float(parsed)
    except Exception:
        pass
    if f_number <= 0.0:
        for k in ("Exif.Photo.FNumber", "Exif.Photo.ApertureValue", "Exif.Photo.MaxApertureValue"):
            fn = _float_exif(k)
            if fn > 0.0:
                f_number = fn
                break
    if f_number <= 0.0:
        for k in ("Xmp.exif.FNumber", "Xmp.exif.ApertureValue"):
            fn = _float_xmp(k)
            if fn > 0.0:
                f_number = fn
                break

    if not model or not lens_model:
        fallback = _read_exif_with_tifffile(raw_path)
        if fallback is not None:
            if not make and fallback.make:
                make = fallback.make
            if not model and fallback.model:
                model = fallback.model
            if not lens_make and fallback.lens_make:
                lens_make = fallback.lens_make
            if not lens_model and fallback.lens_model:
                lens_model = fallback.lens_model
            if focal_length <= 0.0 and fallback.focal_length > 0.0:
                focal_length = fallback.focal_length
            if f_number <= 0.0 and fallback.f_number > 0.0:
                f_number = fallback.f_number

    return ExifData(
        make=make,
        model=model,
        lens_make=lens_make,
        lens_model=lens_model,
        focal_length=focal_length,
        f_number=f_number,
    )


def _clean_maker_name(maker: object) -> str:
    """Clean company suffixes and normalize maker names for lensfun matching."""
    text = str(maker or "").strip()
    if not text:
        return ""
    for suffix in [
        " corporation", " corp.", " corp",
        " co., ltd.", " co.,ltd.", " co. ltd", " co. ltd.",
        " inc.", " inc", " llc", " ltd.", " ltd", " ag", " gmbh",
    ]:
        if text.lower().endswith(suffix):
            text = text[:-len(suffix)].strip()
            break
    aliases = {
        "nikon corporation": "Nikon",
        "nikon": "Nikon",
        "canon": "Canon",
        "sony": "Sony",
        "panasonic": "Panasonic",
        "fujifilm": "Fujifilm",
        "fuji": "Fujifilm",
        "olympus": "Olympus",
        "leica camera": "Leica",
        "leica camera ag": "Leica",
        "leica": "Leica",
        "pentax": "Pentax",
        "sigma": "Sigma",
        "tamron": "Tamron",
        "tokina": "Tokina",
        "samyang": "Samyang",
        "zeiss": "Zeiss",
        "carl zeiss": "Carl Zeiss",
        "om digital solutions": "OM Digital Solutions",
    }
    return aliases.get(text.lower(), text)


def _normalize_lens_text(value: object) -> str:
    """Normalize lens metadata strings for case-insensitive comparisons."""

    return ' '.join(str(value).strip().lower().split())


def _compact_lens_text(value: object) -> str:
    """Collapse lens metadata to alphanumeric characters for loose matching."""

    return ''.join(character.lower() for character in str(value) if character.isalnum())


def _lens_model_score(lens_model: object, exif_lens_model: str) -> tuple[int, int, int]:
    """Score how closely a database lens model matches the EXIF lens model."""

    normalized_model = _normalize_lens_text(exif_lens_model)
    compact_model = _compact_lens_text(exif_lens_model)
    normalized_lens_model = _normalize_lens_text(lens_model)
    compact_lens_model = _compact_lens_text(lens_model)

    compact_match = int(
        bool(compact_model)
        and bool(compact_lens_model)
        and (compact_model in compact_lens_model or compact_lens_model in compact_model)
    )

    return (
        int(normalized_lens_model == normalized_model),
        compact_match,
        len(set(normalized_model.split()) & set(normalized_lens_model.split())),
    )


def _lens_matches_focal(lens: object, focal_length: float) -> bool:
    """Return whether the focal length falls within the lens calibration range."""

    min_focal = getattr(lens, "min_focal", None)
    max_focal = getattr(lens, "max_focal", None)
    if min_focal is None or max_focal is None or focal_length <= 0:
        return False
    return float(min_focal) <= focal_length <= float(max_focal)


def _lens_matches_aperture(lens: object, f_number: float) -> bool:
    """Return whether the aperture falls within the lens calibration range."""

    min_aperture = getattr(lens, "min_aperture", None)
    max_aperture = getattr(lens, "max_aperture", None)
    if min_aperture is None or f_number <= 0:
        return False
    upper_bound = float(max_aperture) if max_aperture is not None else float('inf')
    return float(min_aperture) <= f_number <= upper_bound


def _find_lens_candidates(db: object, camera: object, exif_metadata: ExifData) -> list[object]:
    """Find lensfun candidates with direct queries before a broad fallback search."""

    clean_make = _clean_maker_name(exif_metadata.lens_make)
    camera_make = _clean_maker_name(getattr(camera, "maker", ""))

    queries = [
        (exif_metadata.lens_make or None, exif_metadata.lens_model, False),
        (clean_make or None, exif_metadata.lens_model, False),
        (camera_make or None, exif_metadata.lens_model, False),
        (exif_metadata.lens_make or None, exif_metadata.lens_model, True),
        (clean_make or None, exif_metadata.lens_model, True),
        (camera_make or None, exif_metadata.lens_model, True),
        (None, exif_metadata.lens_model, True),
    ]

    def dedupe(lenses: list[object]) -> list[object]:
        candidates: list[object] = []
        seen: set[tuple[object, ...]] = set()
        for lens in lenses:
            identity = (
                _normalize_lens_text(getattr(lens, 'maker', '')),
                _normalize_lens_text(getattr(lens, 'model', '')),
                getattr(lens, 'min_focal', None),
                getattr(lens, 'max_focal', None),
                getattr(lens, 'min_aperture', None),
                getattr(lens, 'max_aperture', None),
            )
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(lens)
        return candidates

    for lens_make, lens_model, loose_search in queries:
        try:
            direct_candidates = dedupe(db.find_lenses(camera, lens_make, lens_model, loose_search=loose_search))
            if direct_candidates:
                return direct_candidates
        except Exception:
            continue

    try:
        broader_candidates = dedupe(db.find_lenses(camera, None, None, loose_search=True))
        matched = [
            lens for lens in broader_candidates
            if _lens_model_score(getattr(lens, 'model', ''), exif_metadata.lens_model) > (0, 0, 0)
        ]
        if matched:
            return matched
    except Exception:
        pass

    # Global database search across all lenses if camera-specific mounts failed
    all_lenses = getattr(db, "lenses", [])
    global_matched = [
        lens for lens in all_lenses
        if _lens_model_score(getattr(lens, 'model', ''), exif_metadata.lens_model) > (0, 0, 0)
    ]
    if global_matched:
        return dedupe(global_matched)

    return []


def _parse_focal_range(text: str) -> tuple[float, float] | None:
    """Extract (min_focal, max_focal) from a lens model name or lens specification."""
    if not text:
        return None
    m_zoom = re.search(r'(\d+(?:\.\d+)?)\s*[-–—/]\s*(\d+(?:\.\d+)?)\s*(?:mm)?', text, re.IGNORECASE)
    if m_zoom:
        try:
            return float(m_zoom.group(1)), float(m_zoom.group(2))
        except (ValueError, TypeError):
            pass
    m_prime = re.search(r'(\d+(?:\.\d+)?)\s*mm\b', text, re.IGNORECASE)
    if m_prime:
        try:
            f = float(m_prime.group(1))
            return f, f
        except (ValueError, TypeError):
            pass
    return None


def _select_lens_candidate(lenses: list[object], exif_metadata: ExifData, camera: object = None) -> object:
    """Pick the best lens candidate using camera mount, maker, focal range, and model cues."""

    if not lenses:
        raise ValueError("Cannot select from empty lens candidate list")

    cam_mount = getattr(camera, 'mount', '') if camera is not None else ''
    cam_maker = _clean_maker_name(getattr(camera, 'maker', '')) if camera is not None else ''
    exif_make = _clean_maker_name(exif_metadata.make)
    exif_lens_make = _clean_maker_name(exif_metadata.lens_make)
    exif_focal_range = _parse_focal_range(exif_metadata.lens_model)

    def lens_score(lens: object) -> tuple[int, int, int, tuple[int, int, int], int, int, float]:
        lens_maker = _clean_maker_name(getattr(lens, 'maker', ''))
        lens_mounts = getattr(lens, 'mounts', []) or []

        # 1. Mount compatibility
        mount_score = 0
        if cam_mount:
            if lens_mounts:
                mount_score = 10 if cam_mount in lens_mounts else -50
            else:
                mount_score = 0

        # 2. Maker compatibility
        maker_score = 0
        expected_maker = exif_lens_make or exif_make or cam_maker
        if expected_maker:
            if lens_maker.lower() == expected_maker.lower():
                maker_score = 5
            elif exif_lens_make and lens_maker.lower() != exif_lens_make.lower():
                maker_score = -20
            elif cam_maker and lens_maker.lower() != cam_maker.lower():
                # Major camera brands are mutually exclusive
                major_brands = {'canon', 'nikon', 'sony', 'fujifilm', 'panasonic', 'olympus', 'pentax', 'leica'}
                if lens_maker.lower() in major_brands and cam_maker.lower() in major_brands:
                    maker_score = -30

        # 3. Focal range match from model name
        range_score = 0
        if exif_focal_range is not None:
            exif_min, exif_max = exif_focal_range
            lens_min = float(getattr(lens, 'min_focal', 0.0) or 0.0)
            lens_max = float(getattr(lens, 'max_focal', 0.0) or 0.0)
            if abs(lens_min - exif_min) < 1.0 and abs(lens_max - exif_max) < 1.0:
                range_score = 10
            elif lens_min <= exif_min + 0.5 and exif_max <= lens_max + 0.5:
                range_score = 3
            else:
                range_score = -10

        # 4. Model text match
        m_score = _lens_model_score(getattr(lens, 'model', ''), exif_metadata.lens_model)

        # 5. Shot focal and aperture matches
        shot_focal = int(_lens_matches_focal(lens, exif_metadata.focal_length))
        shot_aperture = int(_lens_matches_aperture(lens, exif_metadata.f_number))

        # 6. Database score
        db_score = float(getattr(lens, 'score', 0.0) or 0.0)

        return (
            mount_score,
            maker_score,
            range_score,
            m_score,
            shot_focal,
            shot_aperture,
            db_score,
        )

    return max(lenses, key=lens_score)


def _apply_lens_correction(
    rgb: np.ndarray,
    exif_metadata: ExifData,
) -> tuple[np.ndarray, str]:
    """Apply ``lensfun`` lens corrections to the image.

    Camera and lens are looked up in the ``lensfun`` database using the
    supplied EXIF metadata. If either the camera or lens is not found, no
    correction is applied and the original image is returned.

    The following corrections are applied in sequence, if supported by the
    identified lens: vignetting, chromatic aberration, distortion,
    geometry and scale.

    Parameters
    ----------
    rgb
        Linear ``float32`` RGB image array with shape ``(H, W, 3)``.
    exif_metadata
        Camera and lens metadata read from the RAW file.

    Returns:
    -------
    numpy.ndarray
        Corrected (or original) ``float32`` RGB image.
    str
        Human readable summary of the camera, lens, focal length and aperture
        used for the correction. Empty when the camera or lens was not found in
        the ``lensfun`` database.

    """
    if not exif_metadata.lens_model.strip():
        return rgb, ""

    db = lensfunpy.Database()

    clean_make = _clean_maker_name(exif_metadata.make)
    clean_lens_make = _clean_maker_name(exif_metadata.lens_make)

    # 1. Exact camera search first
    exact_cameras: list[object] = []
    for make in (exif_metadata.make, clean_make):
        if make and exif_metadata.model:
            try:
                found = db.find_cameras(make, exif_metadata.model, loose_search=False)
                if found:
                    exact_cameras = found
                    break
            except Exception:
                continue

    camera = exact_cameras[0] if exact_cameras else None

    # 2. Collect candidate lenses from camera (if exact) and global database
    camera_lenses: list[object] = []
    if camera is not None:
        camera_lenses = _find_lens_candidates(db, camera, exif_metadata)

    all_lenses = getattr(db, "lenses", [])
    global_lenses = [
        lens for lens in all_lenses
        if _lens_model_score(getattr(lens, 'model', ''), exif_metadata.lens_model) > (0, 0, 0)
    ]

    # Combine and deduplicate lenses preserving order
    all_candidates: list[object] = []
    seen: set[tuple[object, ...]] = set()
    for l in camera_lenses + global_lenses:
        ident = (
            _normalize_lens_text(getattr(l, 'maker', '')),
            _normalize_lens_text(getattr(l, 'model', '')),
            getattr(l, 'min_focal', None),
            getattr(l, 'max_focal', None),
            getattr(l, 'min_aperture', None),
            getattr(l, 'max_aperture', None),
        )
        if ident not in seen:
            seen.add(ident)
            all_candidates.append(l)

    if not all_candidates:
        return rgb, ""

    lens = _select_lens_candidate(all_candidates, exif_metadata, camera=camera)

    # 3. If no exact camera was found, find best loose camera matching lens mount & crop
    if camera is None:
        loose_cameras: list[object] = []
        for make in (exif_metadata.make, clean_make, None):
            if exif_metadata.model:
                try:
                    found = db.find_cameras(make, exif_metadata.model, loose_search=True)
                    if found:
                        loose_cameras = found
                        break
                except Exception:
                    continue

        if loose_cameras:
            lens_mounts = getattr(lens, "mounts", [])
            lens_crop = getattr(lens, "crop_factor", None)

            def cam_score(cam: object) -> tuple[int, int, int, float]:
                cam_mount = getattr(cam, "mount", "")
                mount_match = int(bool(lens_mounts and cam_mount in lens_mounts))
                cam_crop = getattr(cam, "crop_factor", None)
                crop_match = int(bool(lens_crop and cam_crop and abs(cam_crop - lens_crop) < 0.1))
                cam_model = getattr(cam, "model", "").lower()
                req_model = exif_metadata.model.lower()
                prefix_len = 0
                for c1, c2 in zip(cam_model, req_model):
                    if c1 == c2:
                        prefix_len += 1
                    else:
                        break
                return (mount_match, crop_match, prefix_len, float(getattr(cam, "score", 0.0) or 0.0))

            loose_cameras.sort(key=cam_score, reverse=True)
            camera = loose_cameras[0]
            lens = _select_lens_candidate(all_candidates, exif_metadata, camera=camera)

    crop_factor = getattr(camera, "crop_factor", None) or getattr(lens, "crop_factor", 1.0) or 1.0

    focal = exif_metadata.focal_length
    if focal <= 0.0:
        focal = float(getattr(lens, "min_focal", 50.0) or 50.0)

    aperture = exif_metadata.f_number
    if aperture <= 0.0:
        aperture = float(getattr(lens, "min_aperture", 2.8) or 2.8)

    lens_label = getattr(lens, 'model', None) or exif_metadata.lens_model or str(lens)
    lens_info = f"{lens_label} @ {focal}mm f/{aperture}"

    height, width = rgb.shape[:2]

    try:
        mod = lensfunpy.Modifier(lens, crop_factor, width, height)
        mod.initialize(
            focal,
            aperture,
            pixel_format=np.float32,
            flags=lensfunpy.ModifyFlags.ALL,
        )

        mod.apply_color_modification(rgb)
        undist_coords = mod.apply_subpixel_geometry_distortion()

        if undist_coords is None:
            geom_coords = mod.apply_geometry_distortion()
            if geom_coords is not None:
                undist_coords = np.repeat(geom_coords[:, :, np.newaxis, :], 3, axis=2)

        if undist_coords is not None:
            from spektrafilm.cuda.device import is_cuda_available
            if is_cuda_available():
                import cupy as cp
                import cupyx.scipy.ndimage as cnd

                rgb_gpu = cp.asarray(rgb)
                coords_gpu = cp.asarray(undist_coords)
                corrected_gpu = cp.empty_like(rgb_gpu)
                for c in range(rgb.shape[2]):
                    coord_stack = cp.stack([coords_gpu[:, :, c, 1], coords_gpu[:, :, c, 0]], axis=0)
                    corrected_gpu[:, :, c] = cnd.map_coordinates(rgb_gpu[:, :, c], coord_stack, order=1, mode="nearest")
                rgb = cp.asnumpy(corrected_gpu)
                del rgb_gpu, coords_gpu, corrected_gpu
            else:
                corrected = np.empty_like(rgb)

                for c in range(rgb.shape[2]):
                    corrected[:, :, c] = map_coordinates(
                        rgb[:, :, c],
                        [undist_coords[:, :, c, 1], undist_coords[:, :, c, 0]],
                        order=1,
                        mode="nearest",
                    )

                rgb = corrected
    except Exception as exc:
        logger.warning("Lens correction failed: %s", exc)
        return rgb, ""

    return rgb, lens_info


def find_dng_converter() -> str | None:
    """Find the path to the Adobe DNG Converter executable on the system."""
    env_path = os.environ.get("SPEKTRAFILM_DNG_CONVERTER")
    if env_path and os.path.isfile(env_path):
        return env_path

    which_path = shutil.which("Adobe DNG Converter") or shutil.which("dngconverter")
    if which_path:
        return which_path

    win_paths = [
        Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "Adobe" / "Adobe DNG Converter" / "Adobe DNG Converter.exe",
        Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Adobe" / "Adobe DNG Converter" / "Adobe DNG Converter.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Adobe DNG Converter" / "Adobe DNG Converter.exe",
        Path("C:/Program Files/Adobe/Adobe DNG Converter/Adobe DNG Converter.exe"),
        Path("C:/Program Files (x86)/Adobe/Adobe DNG Converter/Adobe DNG Converter.exe"),
    ]
    for p in win_paths:
        if p.is_file():
            return str(p)

    mac_paths = [
        Path("/Applications/Adobe DNG Converter.app/Contents/MacOS/Adobe DNG Converter"),
    ]
    for p in mac_paths:
        if p.is_file():
            return str(p)

    return None


def is_dng_converter_available() -> bool:
    """Return True if Adobe DNG Converter is installed and accessible."""
    return find_dng_converter() is not None


def convert_raw_to_dng(
    raw_path: str | PathLike[str],
    output_dir: str | PathLike[str] | None = None,
    timeout: float = 120.0,
) -> Path:
    """Convert an unsupported camera RAW file into a standard DNG RAW file.

    Uses Adobe DNG Converter to perform lossless raw compression into DNG.
    """
    p = Path(raw_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"RAW file not found: {raw_path}")

    if p.suffix.lower() == ".dng":
        return p

    if output_dir is not None:
        dest_dir = Path(output_dir).resolve()
    else:
        dest_dir = p.parent

    # Test if dest_dir is writable; fall back to cache if read-only
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        test_file = dest_dir / f".sf_test_write_{os.getpid()}"
        test_file.touch()
        test_file.unlink()
    except (OSError, PermissionError):
        dest_dir = Path(tempfile.gettempdir()) / "spektrafilm_dng"
        dest_dir.mkdir(parents=True, exist_ok=True)

    target_dng = dest_dir / f"{p.stem}.dng"
    target_dng_upper = dest_dir / f"{p.stem}.DNG"

    # Reuse existing DNG if already converted and up to date
    for cand in (target_dng, target_dng_upper):
        if cand.is_file() and cand.stat().st_size > 0:
            try:
                if cand.stat().st_mtime >= p.stat().st_mtime:
                    return cand
            except OSError:
                return cand

    converter_exe = find_dng_converter()
    if converter_exe is None:
        raise RuntimeError(
            f"Cannot decode RAW file '{p.name}' because its sensor compression is unsupported by LibRaw, "
            f"and Adobe DNG Converter was not found.\n\n"
            f"Please install Adobe DNG Converter or set the SPEKTRAFILM_DNG_CONVERTER environment variable."
        )

    logger.info("Converting unsupported RAW '%s' to DNG in '%s'...", p.name, dest_dir)
    cmd = [converter_exe, "-c", "-d", str(dest_dir), str(p)]

    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Adobe DNG Converter timed out converting '{p.name}'.") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to execute Adobe DNG Converter: {exc}") from exc

    for cand in (target_dng, target_dng_upper):
        if cand.is_file() and cand.stat().st_size > 0:
            logger.info("Successfully converted '%s' to DNG-RAW '%s'.", p.name, cand.name)
            return cand

    stderr_text = proc.stderr.decode("utf-8", errors="replace").strip() if proc.stderr else ""
    stdout_text = proc.stdout.decode("utf-8", errors="replace").strip() if proc.stdout else ""
    msg = f"Adobe DNG Converter failed to convert '{p.name}' (exit code {proc.returncode})."
    if stderr_text or stdout_text:
        msg += f"\nOutput: {stderr_text or stdout_text}"
    raise RuntimeError(msg)


def resolve_raw_path(raw_path: str | PathLike[str]) -> Path | str:
    """Resolve a RAW file path, converting it to DNG-RAW if unsupported by LibRaw.

    If the RAW file is already supported or is a DNG, returns the original path.
    If the RAW file is unsupported (e.g. Nikon HE / HE* NEF), checks for an
    existing converted DNG in the same directory or converts it using Adobe DNG Converter.
    """
    p = Path(raw_path)
    if not p.is_file():
        return raw_path

    if p.suffix.lower() == ".dng":
        return raw_path

    # Check if a converted DNG already exists alongside the file
    for cand_name in (f"{p.stem}.dng", f"{p.stem}.DNG"):
        cand = p.parent / cand_name
        if cand.is_file() and cand.stat().st_size > 0:
            try:
                if cand.stat().st_mtime >= p.stat().st_mtime:
                    return str(cand) if isinstance(raw_path, str) else cand
            except OSError:
                return str(cand) if isinstance(raw_path, str) else cand

    # Check if LibRaw can decode sensor data directly
    try:
        with rawpy.imread(str(raw_path)) as raw:
            raw.unpack()
            return raw_path
    except Exception:
        pass

    logger.info("RAW file '%s' is unsupported by LibRaw; converting to DNG-RAW...", p.name)
    dng = convert_raw_to_dng(p)
    return str(dng) if isinstance(raw_path, str) else dng


def load_and_process_raw_file(
    raw_path: str | PathLike[str],
    white_balance='as_shot',
    temperature: float | None = None,
    tint: float | None = None,
    lens_correction: bool = False,
    chroma_denoise: bool = False,
    chroma_denoise_strength: float = 1.0,
    raw_exposure_ev: float = 0.0,
    highlight_recovery: float = 0.0,
    highlight_mode: str | int = "reconstruct",
    output_colorspace: str = "ACES2065-1",
    output_cctf_encoding: bool = False,
    lens_info_out: dict[str, str] | None = None,
) -> np.ndarray:
    """Load a RAW file into linear RGB and optionally convert its colourspace.

    The RAW is demosaiced by ``rawpy`` into linear 16-bit ACES RGB with auto
    brightening disabled. If the RAW file uses a compression format unsupported
    by LibRaw (such as Nikon High Efficiency / HE* compression), it is
    automatically converted to standard DNG-RAW before demosaicing.

    Parameters
    ----------
    raw_path
        Path to the RAW image.
    white_balance
        White-balance mode. Supported string values are ``'as_shot'``,
        ``'daylight'``, ``'tungsten'``, and ``'custom'``. A
        ``(temperature, tint)`` tuple is also accepted.
    temperature
        Correlated colour temperature in kelvin for ``'custom'`` mode.
    tint
        Multiplicative adjustment applied to both green channels for
        temperature-derived white balance.
    lens_correction
        When ``True``, apply all lensfun lens corrections (vignetting,
        TCA, distortion, geometry and scale). Camera and lens are identified
        from the EXIF metadata.
    chroma_denoise
        When ``True``, apply edge-preserving chrominance denoising to reduce color noise.
    chroma_denoise_strength
        Strength of chrominance denoising (0.0 to 2.0).
    raw_exposure_ev
        Pre-demosaic raw exposure shift in EV stops (-4.0 to +4.0). Negative values
        recover unclipped sensor highlights prior to clipping.
    highlight_recovery
        Highlight preservation and recovery factor (0.0 = linear, 1.0 = maximum preservation).
    highlight_mode
        Highlight reconstruction mode (``'reconstruct'``, ``'blend'``, ``'clip'``, ``'ignore'``).
    output_colorspace
        Output RGB colourspace name understood by ``colour.RGB_COLOURSPACES``.
    output_cctf_encoding
        Whether to apply the output colourspace transfer function when a
        colourspace conversion is requested.
    lens_info_out
        Optional dictionary to collect lens identification info.

    Returns
    -------
    numpy.ndarray
        RGB image as ``float32`` in the requested output colourspace.
    """
    actual_path = resolve_raw_path(raw_path)

    params, postprocess_adaptation, tint_multiplier = _postprocess_params(
        white_balance,
        temperature,
        tint,
        chroma_denoise=chroma_denoise,
        chroma_denoise_strength=chroma_denoise_strength,
        raw_exposure_ev=raw_exposure_ev,
        highlight_recovery=highlight_recovery,
        highlight_mode=highlight_mode,
    )

    with rawpy.imread(str(actual_path)) as raw:
        rgb = raw.postprocess(**params).astype(np.float32) / np.float32(65535.0)

    if lens_correction:
        exif_metadata = _read_exif_metadata(actual_path)
        rgb, lens_info = _apply_lens_correction(rgb, exif_metadata)

        if lens_info_out is not None and lens_info:
            lens_info_out["summary"] = lens_info

    if postprocess_adaptation is not None:
        rgb = _apply_white_balance_adaptation(rgb, *postprocess_adaptation)

    rgb = _apply_tint_adjustment(rgb, tint_multiplier)

    if output_colorspace != 'ACES2065-1':
        rgb = colour.RGB_to_RGB(
            rgb,
            input_colourspace=_ACES_COLOURSPACE,
            output_colourspace=colour.RGB_COLOURSPACES[output_colorspace],
            apply_cctf_decoding=False,
            apply_cctf_encoding=output_cctf_encoding,
        )

    if chroma_denoise:
        from spektrafilm.utils.chroma_denoise import chroma_denoise as apply_chroma_denoise
        from spektrafilm.cuda.device import is_cuda_available
        if is_cuda_available():
            import cupy as cp
            rgb_gpu = cp.asarray(rgb)
            rgb_gpu = apply_chroma_denoise(rgb_gpu, strength=chroma_denoise_strength)
            rgb = cp.asnumpy(rgb_gpu)
            del rgb_gpu
        else:
            rgb = apply_chroma_denoise(rgb, strength=chroma_denoise_strength)

    return np.ascontiguousarray(rgb, dtype=np.float32)


__all__ = [
    'load_and_process_raw_file',
    'read_raw_white_balance',
    'convert_raw_to_dng',
    'find_dng_converter',
    'is_dng_converter_available',
    'resolve_raw_path',
]