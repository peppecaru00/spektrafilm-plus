from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from spektrafilm_gui.gallery_sidecar import load_sidecar, save_sidecar
from spektrafilm_gui.persistence import PROJECT_DEFAULT_GUI_STATE
from spektrafilm_gui.state import (
    GuiState,
    clone_gui_state,
    clone_state_section,
)

logger = logging.getLogger(__name__)

RAW_AFFECTING_PARAM_IDS = frozenset({
    "white_balance",
    "white_balance_all",
    "temperature",
    "tint",
    "white_balance_mode",
    "raw_exposure_ev",
    "highlight_recovery",
    "highlight_mode",
    "lens_correction",
    "chroma_denoise",
    "input_color_space",
    "input_cctf_decoding",
})


@dataclass(frozen=True)
class SingleParamDef:
    """Definition for a parameter that can be copied and pasted across gallery images."""

    id: str
    name: str
    category: str
    description: str
    display_getter: Callable[[GuiState], str]
    getter: Callable[[GuiState], Any]
    setter: Callable[[GuiState, Any], None]

    def get_display(self, state: GuiState) -> str:
        try:
            return self.display_getter(state)
        except Exception:
            return "(unavailable)"

    def extract(self, state: GuiState) -> Any:
        return self.getter(state)

    def apply(self, target_state: GuiState, value: Any) -> None:
        self.setter(target_state, value)


# Helper setters
def _set_wb_all(target: GuiState, val: tuple[str, float, float]) -> None:
    target.gui_only.load_raw.white_balance = str(val[0])
    target.gui_only.load_raw.temperature = float(val[1])
    target.gui_only.load_raw.tint = float(val[2])


def _set_chroma_denoise(target: GuiState, val: tuple[bool, float]) -> None:
    target.gui_only.load_raw.chroma_denoise = bool(val[0])
    target.gui_only.load_raw.chroma_denoise_strength = float(val[1])


def _set_enlarger_all(target: GuiState, val: Any) -> None:
    target.simulation.enlarger = clone_state_section(val)


def _set_output_cs(target: GuiState, val: tuple[str, bool, str]) -> None:
    target.simulation.workflow.saving_color_space = str(val[0])
    target.simulation.workflow.saving_cctf_encoding = bool(val[1])
    target.simulation.io.output_color_space = str(val[2])


def _set_grain_all(target: GuiState, val: Any) -> None:
    target.grain = clone_state_section(val)


def _set_halation_all(target: GuiState, val: Any) -> None:
    target.halation = clone_state_section(val)


def _set_couplers_all(target: GuiState, val: Any) -> None:
    target.couplers = clone_state_section(val)


def _set_chemistry_all(target: GuiState, val: Any) -> None:
    target.chemistry = clone_state_section(val)


def _set_preflashing_all(target: GuiState, val: tuple[float, float, float]) -> None:
    target.preflashing.preflash_exposure = float(val[0])
    target.preflashing.preflash_y_filter_shift = float(val[1])
    target.preflashing.preflash_m_filter_shift = float(val[2])


def _set_camera_diff(target: GuiState, val: Any) -> None:
    target.camera_diffusion = clone_state_section(val)


def _set_enlarger_diff(target: GuiState, val: Any) -> None:
    target.enlarger_diffusion = clone_state_section(val)


def _set_glare(target: GuiState, val: Any) -> None:
    target.glare = clone_state_section(val)


def _set_scanner(target: GuiState, val: Any) -> None:
    target.scanner = clone_state_section(val)


def _set_crop(target: GuiState, val: tuple[bool, tuple[float, float], tuple[float, float]]) -> None:
    target.input_image.io.crop = bool(val[0])
    target.input_image.io.crop_center = (float(val[1][0]), float(val[1][1]))
    target.input_image.io.crop_size = (float(val[2][0]), float(val[2][1]))


def _set_channel_swaps(target: GuiState, val: tuple[tuple[int, int, int], tuple[int, int, int]]) -> None:
    target.special.film_channel_swap = (int(val[0][0]), int(val[0][1]), int(val[0][2]))
    target.special.print_channel_swap = (int(val[1][0]), int(val[1][1]), int(val[1][2]))


SINGLE_PARAMS: list[SingleParamDef] = [
    # ---------------------------------------------------------
    # 1. Film Simulation
    # ---------------------------------------------------------
    SingleParamDef(
        id="film_stock",
        name="Film Stock",
        category="Film Simulation",
        description="Selected film emulsion profile",
        display_getter=lambda s: s.simulation.selection.film_stock or "(Default)",
        getter=lambda s: s.simulation.selection.film_stock,
        setter=lambda s, v: setattr(s.simulation.selection, "film_stock", str(v)),
    ),
    SingleParamDef(
        id="print_paper",
        name="Print Paper",
        category="Film Simulation",
        description="Selected darkroom photographic print paper",
        display_getter=lambda s: s.simulation.selection.print_paper or "(Default)",
        getter=lambda s: s.simulation.selection.print_paper,
        setter=lambda s, v: setattr(s.simulation.selection, "print_paper", str(v)),
    ),
    SingleParamDef(
        id="scan_film",
        name="Scan Film Mode (Negative vs Print)",
        category="Film Simulation",
        description="Toggle between direct film negative scan and optical print simulation",
        display_getter=lambda s: "Direct Negative Scan" if s.simulation.io.scan_film else "Optical Print Simulation",
        getter=lambda s: bool(s.simulation.io.scan_film),
        setter=lambda s, v: setattr(s.simulation.io, "scan_film", bool(v)),
    ),

    # ---------------------------------------------------------
    # 2. Camera & Exposure Control
    # ---------------------------------------------------------
    SingleParamDef(
        id="exposure_compensation",
        name="Exposure Compensation (EV)",
        category="Camera & Exposure",
        description="Camera exposure compensation in stops",
        display_getter=lambda s: f"{s.camera.exposure_compensation_ev:+.2f} EV",
        getter=lambda s: float(s.camera.exposure_compensation_ev),
        setter=lambda s, v: setattr(s.camera, "exposure_compensation_ev", float(v)),
    ),
    SingleParamDef(
        id="auto_exposure_method",
        name="Auto Exposure Method",
        category="Camera & Exposure",
        description="Method used for automated exposure metering",
        display_getter=lambda s: str(s.camera.auto_exposure_method),
        getter=lambda s: str(s.camera.auto_exposure_method),
        setter=lambda s, v: setattr(s.camera, "auto_exposure_method", str(v)),
    ),
    SingleParamDef(
        id="auto_exposure_active",
        name="Auto Exposure Enable",
        category="Camera & Exposure",
        description="Enable automated scene exposure calculation",
        display_getter=lambda s: "Enabled" if s.camera.auto_exposure else "Disabled",
        getter=lambda s: bool(s.camera.auto_exposure),
        setter=lambda s, v: setattr(s.camera, "auto_exposure", bool(v)),
    ),

    # ---------------------------------------------------------
    # 3. RAW Processing & Optics
    # ---------------------------------------------------------
    SingleParamDef(
        id="white_balance",
        name="White Balance (Full Preset, Temp & Tint)",
        category="RAW Processing",
        description="RAW color temperature, tint, and WB preset mode",
        display_getter=lambda s: f"{s.gui_only.load_raw.white_balance} ({s.gui_only.load_raw.temperature:.0f} K, {s.gui_only.load_raw.tint:+.1f})",
        getter=lambda s: (
            s.gui_only.load_raw.white_balance,
            s.gui_only.load_raw.temperature,
            s.gui_only.load_raw.tint,
        ),
        setter=_set_wb_all,
    ),
    SingleParamDef(
        id="temperature",
        name="Color Temperature (Kelvin)",
        category="RAW Processing",
        description="Color temperature in Kelvin for custom white balance",
        display_getter=lambda s: f"{s.gui_only.load_raw.temperature:.0f} K",
        getter=lambda s: float(s.gui_only.load_raw.temperature),
        setter=lambda s, v: setattr(s.gui_only.load_raw, "temperature", float(v)),
    ),
    SingleParamDef(
        id="tint",
        name="Color Tint (Green / Magenta)",
        category="RAW Processing",
        description="Tint offset for custom white balance",
        display_getter=lambda s: f"{s.gui_only.load_raw.tint:+.1f}",
        getter=lambda s: float(s.gui_only.load_raw.tint),
        setter=lambda s, v: setattr(s.gui_only.load_raw, "tint", float(v)),
    ),
    SingleParamDef(
        id="white_balance_mode",
        name="White Balance Preset Mode",
        category="RAW Processing",
        description="WB mode preset (as_shot, daylight, tungsten, custom)",
        display_getter=lambda s: str(s.gui_only.load_raw.white_balance),
        getter=lambda s: str(s.gui_only.load_raw.white_balance),
        setter=lambda s, v: setattr(s.gui_only.load_raw, "white_balance", str(v)),
    ),
    SingleParamDef(
        id="lens_correction",
        name="Lens Distortion Correction",
        category="RAW Processing",
        description="Geometric lens distortion and vignette correction",
        display_getter=lambda s: "Enabled" if s.gui_only.load_raw.lens_correction else "Disabled",
        getter=lambda s: bool(s.gui_only.load_raw.lens_correction),
        setter=lambda s, v: setattr(s.gui_only.load_raw, "lens_correction", bool(v)),
    ),
    SingleParamDef(
        id="raw_exposure_ev",
        name="RAW Exposure Shift",
        category="RAW Processing",
        description="Pre-demosaic raw exposure shift in EV stops",
        display_getter=lambda s: f"{getattr(s.gui_only.load_raw, 'raw_exposure_ev', 0.0):+.1f} EV",
        getter=lambda s: float(getattr(s.gui_only.load_raw, "raw_exposure_ev", 0.0)),
        setter=lambda s, v: setattr(s.gui_only.load_raw, "raw_exposure_ev", float(v)),
    ),
    SingleParamDef(
        id="highlight_recovery",
        name="Highlight Recovery",
        category="RAW Processing",
        description="Highlight preservation and recovery strength (0.0 to 1.0)",
        display_getter=lambda s: f"{getattr(s.gui_only.load_raw, 'highlight_recovery', 0.5):.0%}",
        getter=lambda s: float(getattr(s.gui_only.load_raw, "highlight_recovery", 0.5)),
        setter=lambda s, v: setattr(s.gui_only.load_raw, "highlight_recovery", float(v)),
    ),
    SingleParamDef(
        id="highlight_mode",
        name="Highlight Reconstruction Mode",
        category="RAW Processing",
        description="Algorithm for reconstructing clipped sensor channels",
        display_getter=lambda s: str(getattr(s.gui_only.load_raw, "highlight_mode", "reconstruct")),
        getter=lambda s: str(getattr(s.gui_only.load_raw, "highlight_mode", "reconstruct")),
        setter=lambda s, v: setattr(s.gui_only.load_raw, "highlight_mode", str(v)),
    ),
    SingleParamDef(
        id="chroma_denoise",
        name="Chroma Denoise",
        category="RAW Processing",
        description="Bayer chrominance noise suppression and strength",
        display_getter=lambda s: f"{'Enabled' if s.gui_only.load_raw.chroma_denoise else 'Disabled'} ({s.gui_only.load_raw.chroma_denoise_strength:.1f}x)",
        getter=lambda s: (
            bool(s.gui_only.load_raw.chroma_denoise),
            float(s.gui_only.load_raw.chroma_denoise_strength),
        ),
        setter=_set_chroma_denoise,
    ),

    # ---------------------------------------------------------
    # 4. Enlarger & Darkroom Printing
    # ---------------------------------------------------------
    SingleParamDef(
        id="enlarger",
        name="Enlarger All (Exposure & Filtration)",
        category="Enlarger & Print",
        description="Enlarger print exposure time and Y/M color filtration shifts",
        display_getter=lambda s: f"Exp {s.simulation.enlarger.print_exposure:.2f}, Y {s.simulation.enlarger.y_filter_shift:+.1f}, M {s.simulation.enlarger.m_filter_shift:+.1f}",
        getter=lambda s: clone_state_section(s.simulation.enlarger),
        setter=_set_enlarger_all,
    ),
    SingleParamDef(
        id="print_exposure",
        name="Print Exposure Time",
        category="Enlarger & Print",
        description="Darkroom enlarger base print exposure duration in seconds",
        display_getter=lambda s: f"{s.simulation.enlarger.print_exposure:.2f} s",
        getter=lambda s: float(s.simulation.enlarger.print_exposure),
        setter=lambda s, v: setattr(s.simulation.enlarger, "print_exposure", float(v)),
    ),
    SingleParamDef(
        id="print_exposure_compensation",
        name="Print Auto Compensation",
        category="Enlarger & Print",
        description="Automatically adjust enlarger exposure to compensate for camera EV shifts",
        display_getter=lambda s: "Enabled" if s.simulation.enlarger.print_exposure_compensation else "Disabled",
        getter=lambda s: bool(s.simulation.enlarger.print_exposure_compensation),
        setter=lambda s, v: setattr(s.simulation.enlarger, "print_exposure_compensation", bool(v)),
    ),
    SingleParamDef(
        id="y_filter_shift",
        name="Yellow Filter Shift (Y)",
        category="Enlarger & Print",
        description="Enlarger yellow dichroic filter shift",
        display_getter=lambda s: f"{s.simulation.enlarger.y_filter_shift:+.1f}",
        getter=lambda s: float(s.simulation.enlarger.y_filter_shift),
        setter=lambda s, v: setattr(s.simulation.enlarger, "y_filter_shift", float(v)),
    ),
    SingleParamDef(
        id="m_filter_shift",
        name="Magenta Filter Shift (M)",
        category="Enlarger & Print",
        description="Enlarger magenta dichroic filter shift",
        display_getter=lambda s: f"{s.simulation.enlarger.m_filter_shift:+.1f}",
        getter=lambda s: float(s.simulation.enlarger.m_filter_shift),
        setter=lambda s, v: setattr(s.simulation.enlarger, "m_filter_shift", float(v)),
    ),
    SingleParamDef(
        id="print_illuminant",
        name="Print Illuminant",
        category="Enlarger & Print",
        description="Darkroom enlarger lamp spectral illuminant",
        display_getter=lambda s: str(s.simulation.enlarger.illuminant),
        getter=lambda s: str(s.simulation.enlarger.illuminant),
        setter=lambda s, v: setattr(s.simulation.enlarger, "illuminant", str(v)),
    ),

    # ---------------------------------------------------------
    # 5. Film Characteristics
    # ---------------------------------------------------------
    SingleParamDef(
        id="grain",
        name="Film Grain (All Settings)",
        category="Film Characteristics",
        description="Grain scale, particle area, blur, and uniformity",
        display_getter=lambda s: f"{'Active' if s.grain.active else 'Inactive'} (Area {s.grain.particle_area_um2:.2f}, Blur {s.grain.blur:.2f})",
        getter=lambda s: clone_state_section(s.grain),
        setter=_set_grain_all,
    ),
    SingleParamDef(
        id="grain_active",
        name="Film Grain Enable",
        category="Film Characteristics",
        description="Toggle silver halide grain simulation on or off",
        display_getter=lambda s: "Enabled" if s.grain.active else "Disabled",
        getter=lambda s: bool(s.grain.active),
        setter=lambda s, v: setattr(s.grain, "active", bool(v)),
    ),
    SingleParamDef(
        id="halation",
        name="Halation",
        category="Film Characteristics",
        description="Red halation glow amount, scatter, and threshold",
        display_getter=lambda s: f"{'Active' if s.halation.active else 'Off'} (Amount {s.halation.halation_amount:.2f})",
        getter=lambda s: clone_state_section(s.halation),
        setter=_set_halation_all,
    ),
    SingleParamDef(
        id="couplers",
        name="DIR Couplers",
        category="Film Characteristics",
        description="Interlayer inhibition couplers and edge sharpness",
        display_getter=lambda s: f"{'Active' if s.couplers.active else 'Inactive'} (Amount {s.couplers.amount:.2f})",
        getter=lambda s: clone_state_section(s.couplers),
        setter=_set_couplers_all,
    ),
    SingleParamDef(
        id="film_density_gamma",
        name="Film Density Curve Gamma",
        category="Film Characteristics",
        description="Negative density curve contrast gamma",
        display_getter=lambda s: f"Gamma {s.special.film_render.density_curve_gamma:.2f}",
        getter=lambda s: float(s.special.film_render.density_curve_gamma),
        setter=lambda s, v: setattr(s.special.film_render, "density_curve_gamma", float(v)),
    ),

    # ---------------------------------------------------------
    # 6. Darkroom & Chemistry
    # ---------------------------------------------------------
    SingleParamDef(
        id="chemistry",
        name="Print Curves / Chemistry Morph",
        category="Darkroom & Chemistry",
        description="Developer chemistry time and print density curve adjustments",
        display_getter=lambda s: "Chemistry curve morph settings",
        getter=lambda s: clone_state_section(s.chemistry),
        setter=_set_chemistry_all,
    ),
    SingleParamDef(
        id="glare",
        name="Print Glare",
        category="Darkroom & Chemistry",
        description="Darkroom print optical glare and flare",
        display_getter=lambda s: f"{'Active' if s.glare.active else 'Off'} ({s.glare.percent:.1f}%)",
        getter=lambda s: clone_state_section(s.glare),
        setter=_set_glare,
    ),
    SingleParamDef(
        id="preflashing",
        name="Preflashing",
        category="Darkroom & Chemistry",
        description="Paper preflash exposure and color filtration shifts",
        display_getter=lambda s: f"Exp {s.preflashing.preflash_exposure:.2f}, Y {s.preflashing.preflash_y_filter_shift:+.1f}, M {s.preflashing.preflash_m_filter_shift:+.1f}",
        getter=lambda s: (
            float(s.preflashing.preflash_exposure),
            float(s.preflashing.preflash_y_filter_shift),
            float(s.preflashing.preflash_m_filter_shift),
        ),
        setter=_set_preflashing_all,
    ),

    # ---------------------------------------------------------
    # 7. Optics & Diffusion
    # ---------------------------------------------------------
    SingleParamDef(
        id="camera_diffusion",
        name="Camera Diffusion Filter",
        category="Optics & Diffusion",
        description="Taking lens optical diffusion filter simulation",
        display_getter=lambda s: f"{'Active' if s.camera_diffusion.active else 'Off'} ({s.camera_diffusion.filter_family}, Str {s.camera_diffusion.strength:.2f})",
        getter=lambda s: clone_state_section(s.camera_diffusion),
        setter=_set_camera_diff,
    ),
    SingleParamDef(
        id="enlarger_diffusion",
        name="Enlarger Diffusion Filter",
        category="Optics & Diffusion",
        description="Darkroom enlarger diffusion glass filter simulation",
        display_getter=lambda s: f"{'Active' if s.enlarger_diffusion.active else 'Off'} ({s.enlarger_diffusion.filter_family}, Str {s.enlarger_diffusion.strength:.2f})",
        getter=lambda s: clone_state_section(s.enlarger_diffusion),
        setter=_set_enlarger_diff,
    ),

    # ---------------------------------------------------------
    # 8. Scanner & Output
    # ---------------------------------------------------------
    SingleParamDef(
        id="scanner",
        name="Scanner Simulation",
        category="Scanner & Output",
        description="Film scanner CCD noise and sensor transfer",
        display_getter=lambda s: f"Blur {s.scanner.lens_blur:.2f}, White {s.scanner.white_correction:.2f}",
        getter=lambda s: clone_state_section(s.scanner),
        setter=_set_scanner,
    ),
    SingleParamDef(
        id="output_color_space",
        name="Output Color Space & CCTF",
        category="Scanner & Output",
        description="Saving color space, transfer encoding, and gamut",
        display_getter=lambda s: f"{s.simulation.workflow.saving_color_space} (CCTF: {s.simulation.workflow.saving_cctf_encoding})",
        getter=lambda s: (
            s.simulation.workflow.saving_color_space,
            bool(s.simulation.workflow.saving_cctf_encoding),
            s.simulation.io.output_color_space,
        ),
        setter=_set_output_cs,
    ),

    # ---------------------------------------------------------
    # 9. Input Image & Advanced
    # ---------------------------------------------------------
    SingleParamDef(
        id="upscale_factor",
        name="Upscale Factor",
        category="Input Image & Advanced",
        description="Spatial upscaling multiplier before simulation",
        display_getter=lambda s: f"{s.input_image.io.upscale_factor:.2f}x",
        getter=lambda s: float(s.input_image.io.upscale_factor),
        setter=lambda s, v: setattr(s.input_image.io, "upscale_factor", float(v)),
    ),
    SingleParamDef(
        id="rgb_to_raw_method",
        name="RGB to RAW Spectral Method",
        category="Input Image & Advanced",
        description="Algorithm for estimating spectral reflectance from RGB",
        display_getter=lambda s: str(s.input_image.settings.rgb_to_raw_method),
        getter=lambda s: str(s.input_image.settings.rgb_to_raw_method),
        setter=lambda s, v: setattr(s.input_image.settings, "rgb_to_raw_method", str(v)),
    ),
    SingleParamDef(
        id="input_gamut_compress",
        name="Input Gamut Compression",
        category="Input Image & Advanced",
        description="Input gamut mapping and knee compression",
        display_getter=lambda s: f"{'Active' if s.input_gamut_compress.active else 'Inactive'} ({s.input_gamut_compress.algorithm})",
        getter=lambda s: clone_state_section(s.input_gamut_compress),
        setter=lambda s, v: setattr(s, "input_gamut_compress", clone_state_section(v)),
    ),
    SingleParamDef(
        id="output_gamut_compress",
        name="Output Gamut Compression",
        category="Input Image & Advanced",
        description="Output gamut compression algorithm",
        display_getter=lambda s: f"{s.output_gamut_compress.algorithm}",
        getter=lambda s: clone_state_section(s.output_gamut_compress),
        setter=lambda s, v: setattr(s, "output_gamut_compress", clone_state_section(v)),
    ),
    SingleParamDef(
        id="crop",
        name="Crop Area",
        category="Input Image & Advanced",
        description="Active crop center and window size",
        display_getter=lambda s: "Active" if s.input_image.io.crop else "Full Image",
        getter=lambda s: (
            bool(s.input_image.io.crop),
            tuple(s.input_image.io.crop_center),
            tuple(s.input_image.io.crop_size),
        ),
        setter=_set_crop,
    ),
    SingleParamDef(
        id="channel_swaps",
        name="Channel Swaps (Film & Print)",
        category="Input Image & Advanced",
        description="Color channel permutations for infrared / special effects",
        display_getter=lambda s: f"Film {s.special.film_channel_swap}, Print {s.special.print_channel_swap}",
        getter=lambda s: (
            tuple(s.special.film_channel_swap),
            tuple(s.special.print_channel_swap),
        ),
        setter=_set_channel_swaps,
    ),
]

_PARAMS_BY_ID = {param.id: param for param in SINGLE_PARAMS}


def get_param_def(param_id: str) -> SingleParamDef | None:
    """Return the parameter definition for a given parameter ID, or None."""
    return _PARAMS_BY_ID.get(param_id)


def all_param_defs() -> list[SingleParamDef]:
    """Return all available single parameter definitions in order."""
    return list(SINGLE_PARAMS)


def apply_single_param_to_file(
    image_path: str | Path,
    param_id: str,
    value: Any,
    fallback_state: GuiState | None = None,
) -> bool:
    """Apply a single parameter to an image's .spktrflm sidecar file without modifying other settings."""
    return apply_params_dict_to_file(image_path, {param_id: value}, fallback_state=fallback_state)


def apply_params_dict_to_file(
    image_path: str | Path,
    params_dict: dict[str, Any],
    fallback_state: GuiState | None = None,
) -> bool:
    """Apply multiple parameters to an image's .spktrflm sidecar file without modifying other settings."""
    if not params_dict:
        return True

    p = Path(image_path)
    state = load_sidecar(p)
    if state is None:
        state = clone_gui_state(fallback_state) if fallback_state is not None else clone_gui_state(PROJECT_DEFAULT_GUI_STATE)

    for param_id, value in params_dict.items():
        param_def = get_param_def(param_id)
        if param_def is not None:
            param_def.apply(state, value)

    save_sidecar(p, state)
    logger.debug("Applied %d parameter(s) to sidecar for %s", len(params_dict), p.name)
    return True


def apply_params_dict_to_state(target_state: GuiState, params_dict: dict[str, Any]) -> GuiState:
    """Apply a dictionary of {param_id: value} to a GuiState object and return it."""
    for param_id, value in params_dict.items():
        param_def = get_param_def(param_id)
        if param_def is not None:
            param_def.apply(target_state, value)
    return target_state
