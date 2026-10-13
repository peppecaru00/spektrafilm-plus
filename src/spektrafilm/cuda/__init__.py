from __future__ import annotations

import warnings

# Suppress benign CuPy warning on Windows wheels when CUDA 12 packages use splayed layout
warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*", category=UserWarning)

from spektrafilm.cuda.device import (
    get_device_info,
    is_cuda_available,
    is_cupy_array,
    resolve_device,
    synchronize,
    to_cpu,
    to_device,
)
from spektrafilm.cuda.interp import (
    cuda_apply_lut_3d,
    cuda_apply_lut_cubic_2d,
    cuda_fast_interp,
)
from spektrafilm.cuda.filters import (
    cuda_apply_diffusion_filter_um,
    cuda_exponential_filter,
    cuda_fftconvolve_2d,
    cuda_gaussian_filter,
    cuda_unsharp_mask,
)
from spektrafilm.cuda.spectral import (
    cuda_compute_density_spectral,
    cuda_density_to_light,
    cuda_fused_spectral_integrate,
    cuda_spectral_integrate,
)
from spektrafilm.cuda.grain import (
    cuda_apply_grain,
    cuda_apply_grain_to_density_layers,
    cuda_interp_density_cmy_layers,
    cuda_layer_particle_model,
)
from spektrafilm.cuda.gamut_compression import (
    cuda_compress_rgb,
    cuda_compress_rgb_aces_rgc,
    cuda_reinhard_knee,
)
from spektrafilm.cuda.color import (
    cuda_apply_matrix,
    cuda_cctf_decoding,
    cuda_cctf_encoding,
    cuda_rgb_to_tc_b,
    cuda_tri2quad,
    get_rgb_to_xyz_matrix,
    get_xyz_to_rgb_matrix,
)

__all__ = [
    "is_cuda_available",
    "get_device_info",
    "resolve_device",
    "to_device",
    "to_cpu",
    "is_cupy_array",
    "synchronize",
    "cuda_fast_interp",
    "cuda_apply_lut_cubic_2d",
    "cuda_apply_lut_3d",
    "cuda_gaussian_filter",
    "cuda_exponential_filter",
    "cuda_fftconvolve_2d",
    "cuda_unsharp_mask",
    "cuda_apply_diffusion_filter_um",
    "cuda_compute_density_spectral",
    "cuda_density_to_light",
    "cuda_spectral_integrate",
    "cuda_fused_spectral_integrate",
    "cuda_layer_particle_model",
    "cuda_interp_density_cmy_layers",
    "cuda_apply_grain_to_density_layers",
    "cuda_apply_grain",
    "cuda_reinhard_knee",
    "cuda_compress_rgb_aces_rgc",
    "cuda_compress_rgb",
    "get_xyz_to_rgb_matrix",
    "get_rgb_to_xyz_matrix",
    "cuda_apply_matrix",
    "cuda_cctf_encoding",
    "cuda_cctf_decoding",
    "cuda_tri2quad",
    "cuda_rgb_to_tc_b",
]
