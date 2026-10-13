import numpy as np
import struct
import colour
import scipy
import functools
import importlib.resources
from opt_einsum import contract
import scipy.interpolate
import scipy.special

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from spektrafilm.profiles.io import Hanatos2025SensitivityAdaptation
from spektrafilm.utils.fast_interp_lut import apply_lut_cubic_2d
from spektrafilm.config import SPECTRAL_SHAPE, STANDARD_OBSERVER_CMFS
from spektrafilm.model.illuminants import standard_illuminant

_SQRT2PI = float(np.sqrt(2.0 * np.pi))
_HANATOS2025_MAX_CORRECTION_STOPS = 2.0  # matches max_correction_stops used during the fit

################################################################################
# LUT generatation of irradiance spectra for any xy chromaticity
# Thanks to hanatos for providing luts and sample code to develop this. I am grateful.

def _load_coeffs_lut(filename='hanatos_irradiance_xy_coeffs_250304.lut'):
    # load lut of coefficients for efficient computations of irradiance spectra
    # formatting
    header_fmt = '=4i'
    header_len = struct.calcsize(header_fmt)
    pixel_fmt = '=4f'
    pixel_len = struct.calcsize(pixel_fmt)

    package = importlib.resources.files('spektrafilm.data.luts.spectral_upsampling')
    resource = package / filename
    with resource.open("rb") as file:
        header = file.read(header_len)
        _, _, width, height = struct.Struct(header_fmt).unpack_from(header)
        px = [[0] * height for _ in range(width)]
        for j in range(height):
            for i in range(width):
                data = file.read(pixel_len)
                px[i][j] = struct.Struct(pixel_fmt).unpack_from(data)
    return np.array(px)

def _tri2quad(tc):
    # converts triangular coordinates into square coordinates.
    # for better sampling of the visible locus of xy chromaticities.
    # the lut is represented in triangular coordinates
    tc = np.array(tc)
    tx = tc[...,0]
    ty = tc[...,1]
    y = ty / np.fmax(1.0 - tx, 1e-10)
    x = (1.0 - tx)*(1.0 - tx)
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    return np.stack((x,y), axis=-1)

def _quad2tri(xy):
    # converts square coordinates into triangular coordinates
    x = xy[...,0]
    y = xy[...,1]
    tx = 1 - np.sqrt(x)
    ty = y * np.sqrt(x)
    return np.stack((tx,ty), axis=-1)

def _fetch_coeffs(tc, lut_coeffs):
    # find the coefficients for spectral upsampling of given rgb coordinates
    # if color_space!='ITU-R BT.2020' or apply_cctf_decoding:
    #     rgb = colour.RGB_to_RGB(rgb, input_colourspace=color_space, apply_cctf_decoding=apply_cctf_decoding,
    #                                     output_colourspace='ITU-R BT.2020', apply_cctf_encoding=False)
    #     rgb = np.clip(rgb,0,1)
    # xyz = colour.RGB_to_XYZ(rgb, colourspace='ITU-R BT.2020', apply_cctf_decoding=False)
    # b = np.sum(xyz, axis=-1)
    # xy = xyz[...,0:2] / b[...,None]
    # tc = _tri2quad(xy)
    coeffs = np.zeros(np.concatenate((tc.shape[:-1],[4])))
    x = np.linspace(0,1,lut_coeffs.shape[0])
    for i in np.arange(4):
        coeffs[...,i] = scipy.interpolate.RegularGridInterpolator((x,x), lut_coeffs[:,:,i], method='cubic')(tc)
    return coeffs

def _compute_spectra_from_coeffs(coeffs, smooth_steps=1):
    wl = SPECTRAL_SHAPE.wavelengths
    wl_up = np.linspace(360,800,441) # upsampled wl for finer initial calculation 0.5 nm
    x = (coeffs[...,0,None] * wl_up + coeffs[...,1,None])*  wl_up  + coeffs[...,2,None]
    y = 1.0 / np.sqrt(x * x + 1.0)
    spectra = 0.5 * x * y +  0.5
    spectra /= coeffs[...,3][...,None]
    
    # gaussian smooth with smooth_step*sigmas and downsample
    step = np.mean(np.diff(wl))
    spectra = scipy.ndimage.gaussian_filter(spectra, step*smooth_steps, axes=-1)
    def interp_slice(a, wl, wl_up):
        return np.interp(wl, wl_up, a)
    spectra = np.apply_along_axis(interp_slice, axis=-1, wl=wl, wl_up=wl_up, arr=spectra)
    return spectra

def compute_lut_spectra(lut_size=128, smooth_steps=1, lut_coeffs_filename='hanatos_irradiance_xy_coeffs_250304.lut'):
    v = np.linspace(0,1,lut_size)
    tx,ty = np.meshgrid(v,v, indexing='ij')
    tc = np.stack((tx,ty), axis=-1)
    lut_coeffs = _load_coeffs_lut(lut_coeffs_filename)
    coeffs = _fetch_coeffs(tc, lut_coeffs)
    lut_spectra = _compute_spectra_from_coeffs(coeffs, smooth_steps=smooth_steps)
    lut_spectra = np.array(lut_spectra, dtype=np.half)
    return lut_spectra

def _load_hanatos2025_spectra_lut(filename='irradiance_xy_tc.npy'):
    data_path = importlib.resources.files('spektrafilm.data.luts.spectral_upsampling').joinpath(filename)
    with data_path.open('rb') as file:
        spectra_lut = np.double(np.load(file))
    return spectra_lut

def _illuminant_to_xy(illuminant_label):
    illu = standard_illuminant(illuminant_label)
    xyz = np.zeros((3))
    for i in np.arange(3):
        xyz[i] = np.sum(illu * STANDARD_OBSERVER_CMFS[:][:,i])
    xy = xyz[0:2] / np.sum(xyz)
    return xy

def _rgb_to_tc_b(rgb, color_space='ITU-R BT.2020', apply_cctf_decoding=False, reference_illuminant='D55'):
    # source_cs = colour.RGB_COLOURSPACES[color_space]
    # target_cs = source_cs.copy()
    # target_cs.whitepoint = ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    # adapted_rgb = colour.RGB_to_RGB(rgb, input_colourspace=source_cs,
    #                                 output_colourspace=target_cs,
    #                                 adaptation_transform='Bradford')    
    # illu_xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][reference_illuminant]
    illu_xy = _illuminant_to_xy(reference_illuminant)
    # CAT16 (Li et al. 2017) replaces the cone-primary instabilities
    # of CAT02 around blue and violet; it is the adaptation matrix
    # CIECAM16 and the spektrafilm output-side CAM16-UCS algorithm use
    # internally, so the input chromaticity projection stays in the
    # same adaptation family as the output gamut compression.
    from spektrafilm.cuda.device import is_cupy_array
    if is_cupy_array(rgb):
        from spektrafilm.cuda.color import cuda_rgb_to_tc_b
        return cuda_rgb_to_tc_b(
            rgb,
            colourspace=color_space,
            apply_cctf_decoding=apply_cctf_decoding,
            reference_illuminant_xy=tuple(illu_xy),
        )

    xyz = colour.RGB_to_XYZ(rgb, colourspace=color_space,
                            apply_cctf_decoding=apply_cctf_decoding,
                            illuminant=illu_xy,
                            chromatic_adaptation_transform='CAT16')
    b = np.sum(xyz, axis=-1)
    xy = xyz[...,0:2] / np.fmax(b[...,None], 1e-10)
    tc = _tri2quad(xy)
    b = np.nan_to_num(b)
    return tc, b

################################################################################
# hanatos2025 sensitivity adaptation


def _radial_mobius_warp_xy(
    xy: np.ndarray,
    center: tuple[float, float],
    alpha: float,
) -> np.ndarray:
    cx, cy = center
    dx = xy[..., 0] - cx
    dy = xy[..., 1] - cy
    d = np.sqrt(dx * dx + dy * dy)
    scale = 1.0 / (1.0 + alpha * d)
    wxy = np.zeros_like(xy)
    wxy[..., 0] = cx + dx * scale
    wxy[..., 1] = cy + dy * scale
    return wxy
    
def hanika_sigmoid(z: np.ndarray, max_val: float) -> np.ndarray:
    """Algebraic sigmoid matching Jakob & Hanika 2019, bounded to [-max_val, max_val]."""
    return z / np.sqrt(1.0 + (z / max_val) ** 2)


def poly2d_deg4(tc: np.ndarray,
                params: np.ndarray,
                center_tc: tuple[float, float] = (0.0, 0.0),) -> np.ndarray:
    x = tc[..., 0] - center_tc[0]
    y = tc[..., 1] - center_tc[1]
    x2 = x * x
    y2 = y * y
    xy_term = x * y
    x3 = x2 * x
    y3 = y2 * y
    _, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14 = params
    # note that we want center_tc to be zero correction -> not adding c0
    return (
        c1 * x + c2 * y + c3 * x2 + c4 * y2 + c5 * xy_term
        + c6 * x3 + c7 * y3 + c8 * (x2 * y) + c9 * (x * y2)
        + c10 * (x2 * x2) + c11 * (y2 * y2) + c12 * (x3 * y) + c13 * (x2 * y2) + c14 * (x * y3)
    )

def locked_logistic_rising(
    x: np.ndarray, mu: float, sigma: float, nu: float,
) -> np.ndarray:
    """Rising flex-and-sigma-locked logistic.

    Locked through (μ, ½) with maximum slope `S = 1/(σ·√(2π))`. ν > 0
    bends the tails (ν=1 ≈ erf in shape; ν<1 fattens the lower tail,
    ν>1 thins it). Inputs may broadcast freely.
    """
    nu = np.maximum(np.asarray(nu, dtype=float), 1e-5)
    S = 1.0 / (sigma * _SQRT2PI)
    k = (2.0 * nu * S) / (1.0 - 2.0 ** (-nu))
    exponent = np.clip(-k * (x - mu), -500.0, 500.0)
    nu_log2 = nu * np.log(2.0)
    log_q = np.where(
        nu_log2 > 50.0,
        nu_log2 + np.log1p(-np.exp(-nu_log2)),
        np.log(np.expm1(nu_log2)),
    )
    log_denom = np.logaddexp(0.0, log_q + exponent)
    return np.exp(-(1.0 / nu) * log_denom)


def eval_poly4_warp_log_exposure_surface(params, illuminant_xy) -> np.ndarray:
    """Evaluate a degree-4 polynomial surface after warping in xy, then mapping back to tc."""
    # Warp xy coordinate with the mobius warp, which is a radial compression towards the illuminant chromaticity.
    # This is to better sample the spectra of chromaticities close to the illuminant
    # and stabilize the surface for large chromaticity shifts.
    surface_size = HANATOS2025_SPECTRA_LUT.shape[0]
    tc_base = np.linspace(0,1,surface_size)
    tc = np.stack(np.meshgrid(tc_base, tc_base, indexing='ij'), axis=-1)
    xy = _quad2tri(tc)
    tc_center = _tri2quad(illuminant_xy)
    surface_log_exposure_correction = np.zeros((surface_size, surface_size,3))
    for i in range(3):
        poly_params = params[i, :-1]
        alpha = float(params[i, -1])
        xy_warped = _radial_mobius_warp_xy(xy, illuminant_xy, alpha=alpha)
        tc_warped = _tri2quad(xy_warped)
        raw = poly2d_deg4(tc_warped, poly_params, center_tc=tc_center)
        surface_log_exposure_correction[...,i] = hanika_sigmoid(raw, _HANATOS2025_MAX_CORRECTION_STOPS)
    return surface_log_exposure_correction

def eval_poly4_log_exposure_surface(params, illuminant_xy) -> np.ndarray:
    surface_size = HANATOS2025_SPECTRA_LUT.shape[0]
    tc_base = np.linspace(0,1,surface_size)
    tc = np.stack(np.meshgrid(tc_base, tc_base, indexing='ij'), axis=-1)
    tc_center = _tri2quad(illuminant_xy)
    surface_log_exposure_correction = np.zeros((surface_size, surface_size,3))
    for i in range(3):
        poly_params = params[i]
        raw = poly2d_deg4(tc, poly_params, center_tc=tc_center)
        surface_log_exposure_correction[...,i] = hanika_sigmoid(raw, _HANATOS2025_MAX_CORRECTION_STOPS)
    return surface_log_exposure_correction

def eval_logiflex8_spectral_bandpass(params: np.ndarray) -> np.ndarray:
    """8-param logiflex window: erf6 skeleton + (nu_uv, nu_ir) shared across channels.

    params: (c_uv_base, sigma_uv, c_ir_base, sigma_ir, c_uv_b, c_ir_r, nu_uv, nu_ir).
    """
    wavelengths = SPECTRAL_SHAPE.wavelengths
    (c_uv_base, sigma_uv, c_ir_base, sigma_ir,
     c_uv_b, c_ir_r, nu_uv, nu_ir) = params
    cuv = np.array([c_uv_base, c_uv_base, c_uv_b], dtype=float)
    cir = np.array([c_ir_r, c_ir_base, c_ir_base], dtype=float)
    w = wavelengths[:, None]
    edge_uv = locked_logistic_rising(w, cuv[None, :], sigma_uv, nu_uv)
    edge_ir = 1.0 - locked_logistic_rising(w, cir[None, :], sigma_ir, nu_ir)
    window = edge_uv * edge_ir
    return window

def eval_erf4_spectral_bandpass(params: np.ndarray) -> np.ndarray:
    """4-param erf window: common bandpass replicated across all channels.

    params: (c_uv, sigma_uv, c_ir, sigma_ir).
    """
    _SQRT2 = float(np.sqrt(2.0))
    wavelengths = SPECTRAL_SHAPE.wavelengths
    c_uv, sigma_uv, c_ir, sigma_ir = (float(v) for v in params)
    edge_uv = 0.5 * (1.0 + scipy.special.erf((wavelengths - c_uv) / (sigma_uv * _SQRT2)))
    edge_ir = 0.5 * (1.0 - scipy.special.erf((wavelengths - c_ir) / (sigma_ir * _SQRT2)))
    common = edge_uv * edge_ir
    return np.repeat(common[:, None], 3, axis=1)

def eval_spectral_bandpass_window(params: np.ndarray, model: str = 'erf4') -> np.ndarray:
    if model == 'logiflex8':
        return eval_logiflex8_spectral_bandpass(params)
    if model == 'erf4':
        return eval_erf4_spectral_bandpass(params)
    raise ValueError(f"Unknown spectral bandpass model: {model}")
    
def eval_log_exposure_correction_surface(params, illuminant_xy, model='poly4') -> np.ndarray:
    if model == 'poly4_warp_xy':
        return eval_poly4_warp_log_exposure_surface(params, illuminant_xy)
    if model == 'poly4':
        return eval_poly4_log_exposure_surface(params, illuminant_xy)
    raise ValueError(f"Unknown log exposure correction surface model: {model}")

################################################################################
# From [Mallett2019]

MALLETT2019_BASIS = colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019.copy().align(SPECTRAL_SHAPE)
def rgb_to_raw_mallett2019(RGB, sensitivity,
                           color_space='sRGB', apply_cctf_decoding=True,
                           reference_illuminant='D65'):
    """
    Converts an RGB color to a raw sensor response using the method described in Mallett et al. (2019).

    Parameters
    ----------
    RGB : array_like
        RGB color values.
    illuminant : array_like
        Illuminant spectral distribution.
    sensitivity : array_like
        Camera sensor spectral sensitivities.
    color_space : str, optional
        The color space of the input RGB values. Default is 'sRGB'.
    apply_cctf_decoding : bool, optional
        Whether to apply the color component transfer function (CCTF) decoding. Default is True.

    Returns
    -------
    raw : ndarray
        Raw sensor response.
    """
    from spektrafilm.cuda.device import is_cupy_array, to_cpu, to_device
    is_gpu = is_cupy_array(RGB)
    cpu_rgb = to_cpu(RGB) if is_gpu else RGB

    illuminant = standard_illuminant(reference_illuminant)[:]
    basis_set_with_illuminant = np.array(MALLETT2019_BASIS[:])*np.array(illuminant)[:, None]
    lrgb = colour.RGB_to_RGB(cpu_rgb, color_space, 'sRGB',
                    apply_cctf_decoding=apply_cctf_decoding,
                    apply_cctf_encoding=False)
    raw  = contract('ijk,lk,lm->ijm', lrgb, basis_set_with_illuminant, sensitivity)
    raw = np.nan_to_num(raw)
    raw = np.ascontiguousarray(raw)
    
    raw_midgray  = np.einsum('k,km->m', illuminant*0.184, sensitivity) # use 0.184 as midgray reference
    res = raw / raw_midgray[1] # normalize with green channel
    if is_gpu:
        return to_device(res)
    return res

################################################################################
# Using hanatos irradiance spectra generation

HANATOS2025_SPECTRA_LUT = _load_hanatos2025_spectra_lut()
HANATOS2025_NO_ADAPTATION = Hanatos2025SensitivityAdaptation(apply_surface=False, apply_window=False)

def compute_hanatos2025_tc_lut(sensitivity, hanatos2025_adaptation, gamut_compress=None):
    """Compute the LUT of spectra for given sensitivities and hanatos2025 adaptation parameters.
    Both window and surface preserve white balance.

    Parameters
    ----------
    sensitivity, hanatos2025_adaptation :
        As before.
    gamut_compress :
        Optional ``spektrafilm.utils.gamut_compression.InputGamutCompressSpec``.
        When provided and ``gamut_compress.active``, the returned
        LUT has the input gamut compression baked into it via remap-resample
        (see n100 §3.1): a runtime lookup ``new_lut[xy]`` returns the same
        raw value the uncompressed LUT would have returned for
        ``compress(xy)``. The per-pixel hot path therefore stays
        compression-agnostic.
    """

    spectra_lut = HANATOS2025_SPECTRA_LUT

    if hanatos2025_adaptation.spectral_gaussian_blur > 0:
        spectra_lut = scipy.ndimage.gaussian_filter(spectra_lut,
                        (0, 0, hanatos2025_adaptation.spectral_gaussian_blur))

    if hanatos2025_adaptation.apply_window:
        window = eval_spectral_bandpass_window(hanatos2025_adaptation.window_params)
        illuminant = standard_illuminant(hanatos2025_adaptation.reference_illuminant)
        normalization = (
            np.sum(sensitivity * illuminant[:, None] * window, axis=0)
            / np.sum(sensitivity * illuminant[:, None], axis=0)
        )
        window = window / normalization
        raw_lut = contract('ijl,lm->ijm', spectra_lut, sensitivity * window)
    else:
        raw_lut = contract('ijl,lm->ijm', spectra_lut, sensitivity)

    if hanatos2025_adaptation.apply_surface:
        xy_illu = _illuminant_to_xy(hanatos2025_adaptation.reference_illuminant)
        surface = eval_log_exposure_correction_surface(hanatos2025_adaptation.surface_params,
                                                       illuminant_xy=xy_illu)
        raw_lut *= 2**surface

    if gamut_compress is not None and gamut_compress.active:
        # Bake compression into the LUT at build time. The reference
        # illuminant matches what _rgb_to_tc_b uses at runtime, so the
        # compression's achromatic axis is the film's white.
        from spektrafilm.utils.gamut_compression import remap_tc_lut_for_compression
        ref_xy = _illuminant_to_xy(hanatos2025_adaptation.reference_illuminant)
        raw_lut = remap_tc_lut_for_compression(raw_lut, ref_xy, gamut_compress)

    return raw_lut

def rgb_to_raw_hanatos2025(rgb, sensitivity,
                           color_space,
                           apply_cctf_decoding,
                           reference_illuminant,
                           tc_lut=None
                           ):
    tc_raw, b = _rgb_to_tc_b(
        rgb,
        color_space=color_space,
        apply_cctf_decoding=apply_cctf_decoding,
        reference_illuminant=reference_illuminant,
    )
    if tc_lut is None: # fallback to on-the-fly computation if tc_lut not provided
        tc_lut = compute_hanatos2025_tc_lut(sensitivity, HANATOS2025_NO_ADAPTATION)
    raw = apply_lut_cubic_2d(tc_lut, tc_raw)
    raw *= b[...,None] # scale the raw back with the scale factor
    # note that sensitivities are already normalized in balancing such that raw_midgray is 1, so no need to normalize here
    return raw

def rgb_to_smooth_spectrum(rgb, color_space, apply_cctf_decoding, reference_illuminant):
    # direct interpolation of the spectra lut, to be used only for smooth spectra close to white
    tc_w, b_w = _rgb_to_tc_b(
        rgb,
        color_space=color_space,
        apply_cctf_decoding=apply_cctf_decoding,
        reference_illuminant=reference_illuminant,
    )
    v = np.linspace(0, 1, HANATOS2025_SPECTRA_LUT.shape[0])
    spectrum_w = scipy.interpolate.RegularGridInterpolator((v,v), HANATOS2025_SPECTRA_LUT)(tc_w)
    spectrum_w *= b_w
    return spectrum_w.flatten()


################################################################################
# Generic reflectance-LUT registry – identifier-driven loading + usage

_LUT_DATA_PACKAGE = 'spektrafilm.data.luts.spectral_upsampling'


@functools.lru_cache(maxsize=1)
def _lut_registry():
    """{identifier: descriptor} parsed from every `*.toml` sidecar in the LUT data
    folder. Cached once per process (shipped data does not change at runtime)."""
    registry = {}
    for entry in importlib.resources.files(_LUT_DATA_PACKAGE).iterdir():
        if entry.name.endswith('.toml'):
            with entry.open('rb') as file:
                descriptor = tomllib.load(file)
            registry[descriptor['identifier']] = descriptor
    # Alias hanatos2026 to arctic2026beta04 if present
    if 'arctic2026beta04' in registry and 'hanatos2026' not in registry:
        registry['hanatos2026'] = registry['arctic2026beta04']
    return registry


def available_lut_identifiers(kind=None):
    """Sorted identifiers of shipped LUTs, optionally filtered to a `kind`
    ('reflectance' | 'irradiance')."""
    registry = _lut_registry()
    return sorted(i for i, d in registry.items() if kind is None or d.get('kind') == kind)


def lut_descriptor(identifier):
    """Descriptor dict (parsed `.toml` sidecar) for `identifier`. Raises KeyError if
    no LUT with that identifier is shipped."""
    registry = _lut_registry()
    if identifier not in registry:
        raise KeyError(f"no spectral-upsampling LUT {identifier!r}; "
                       f"available: {sorted(registry)}")
    return registry[identifier]


_LUT_SPECTRA_CACHE = {}


def get_lut_spectra(identifier):
    """Lazy-load + cache the (S, S, bands) spectra array for `identifier` from the `.npy`
    its descriptor points at. Lazy so a fresh checkout missing the `.npy` does
    not fail at import; the array loads on first use."""
    if identifier not in _LUT_SPECTRA_CACHE:
        descriptor = lut_descriptor(identifier)
        path = importlib.resources.files(_LUT_DATA_PACKAGE).joinpath(descriptor['file'])
        with path.open('rb') as file:
            _LUT_SPECTRA_CACHE[identifier] = np.double(np.load(file))
    return _LUT_SPECTRA_CACHE[identifier]


def compute_reflectance_tc_lut(identifier, sensitivity, reference_illuminant,
                               spectra_lut=None, gamut_compress=None):
    """(S, S, C) raw LUT for any REFLECTANCE-family method, selected by `identifier`
    (C = 3 for colour film, 1 for a black-and-white emulsion).

    Relight the reflectance by the film reference illuminant, integrate against the film
    sensitivity, self-normalize PER CHANNEL on the method's own emitted neutral (so a neutral
    surface -> gray (1,1,1) raw, matching the white-balance convention the hanatos2025 window
    preserves), and optionally bake input gamut compression."""
    descriptor = lut_descriptor(identifier)
    if descriptor.get('kind') != 'reflectance':
        raise ValueError(f"{identifier!r} is kind={descriptor.get('kind')!r}, not "
                         f"'reflectance' (cannot relight a non-reflectance LUT)")
    if spectra_lut is None:
        spectra_lut = get_lut_spectra(identifier)
    E_ref = standard_illuminant(reference_illuminant)[:]
    relit = spectra_lut * E_ref[None, None, :]
    raw_lut = contract('ijl,lm->ijm', relit, sensitivity)

    scene_illuminant = descriptor['reflectance']['scene_illuminant']
    white_tc = _tri2quad(_illuminant_to_xy(scene_illuminant))
    neutral_refl = apply_lut_cubic_2d(spectra_lut, np.asarray(white_tc).reshape(1, 1, 2))[0, 0]
    raw_neutral = np.einsum('l,lm->m', neutral_refl * E_ref, sensitivity)
    raw_lut = raw_lut / raw_neutral

    if gamut_compress is not None and gamut_compress.active:
        from spektrafilm.utils.gamut_compression import remap_tc_lut_for_compression
        scene_xy = _illuminant_to_xy(scene_illuminant)
        raw_lut = remap_tc_lut_for_compression(raw_lut, scene_xy, gamut_compress)

    return raw_lut


def rgb_to_raw_reflectance(identifier, rgb, sensitivity, color_space,
                           apply_cctf_decoding, reference_illuminant,
                           tc_lut=None, scene_illuminant=None):
    """Runtime path for any REFLECTANCE-family method, selected by `identifier`."""
    descriptor = lut_descriptor(identifier)
    if scene_illuminant is None:
        scene_illuminant = descriptor['reflectance']['scene_illuminant']
    tc_raw, b = _rgb_to_tc_b(
        rgb,
        color_space=color_space,
        apply_cctf_decoding=apply_cctf_decoding,
        reference_illuminant=scene_illuminant,
    )
    if tc_lut is None:
        tc_lut = compute_reflectance_tc_lut(identifier, sensitivity, reference_illuminant)
    raw = apply_lut_cubic_2d(tc_lut, tc_raw)
    raw *= b[..., None]
    return raw


def compute_tc_lut(method, sensitivity, reference_illuminant=None,
                   spectra_lut=None, gamut_compress=None, hanatos2025_adaptation=None):
    """Build the (S, S, 3) raw tc_lut for any shipped `method`, dispatched on its
    descriptor `kind`. REFLECTANCE methods relight + integrate (need reference_illuminant);
    the IRRADIANCE method integrates directly with optional sensitivity adaptation."""
    kind = lut_descriptor(method).get('kind')
    if kind == 'reflectance':
        return compute_reflectance_tc_lut(method, sensitivity, reference_illuminant,
                                          spectra_lut=spectra_lut, gamut_compress=gamut_compress)
    if kind == 'irradiance':
        adaptation = (hanatos2025_adaptation if hanatos2025_adaptation is not None
                      else HANATOS2025_NO_ADAPTATION)
        return compute_hanatos2025_tc_lut(sensitivity, adaptation, gamut_compress=gamut_compress)
    raise ValueError(f"method {method!r} has unsupported LUT kind {kind!r}")


def rgb_to_raw(method, rgb, sensitivity, color_space, apply_cctf_decoding,
               reference_illuminant, tc_lut=None, scene_illuminant=None):
    """Convert RGB to film raw for any shipped `method`, dispatched on its descriptor
    `kind`. REFLECTANCE methods project under the scene illuminant and sample the relit
    tc_lut; the IRRADIANCE method (hanatos2025) samples its irradiance tc_lut directly."""
    kind = lut_descriptor(method).get('kind')
    if kind == 'reflectance':
        return rgb_to_raw_reflectance(method, rgb, sensitivity, color_space,
                                      apply_cctf_decoding, reference_illuminant,
                                      tc_lut=tc_lut, scene_illuminant=scene_illuminant)
    if kind == 'irradiance':
        return rgb_to_raw_hanatos2025(rgb, sensitivity, color_space=color_space,
                                      apply_cctf_decoding=apply_cctf_decoding,
                                      reference_illuminant=reference_illuminant, tc_lut=tc_lut)
    raise ValueError(f"method {method!r} has unsupported LUT kind {kind!r}")


if __name__=='__main__':
    lut_coeffs = _load_coeffs_lut()
    coeffs = _fetch_coeffs(np.array([[1,1]]) ,lut_coeffs)
    spectra = _compute_spectra_from_coeffs(coeffs)
    lut_spectra = compute_lut_spectra(lut_size=128)

