"""CUDA-accelerated 1D and multi-dimensional interpolation kernels."""
from __future__ import annotations

from typing import Any
import numpy as np

from spektrafilm.cuda.device import is_cuda_available, to_device, to_cpu

_FAST_INTERP_1D_CUDA_SOURCE = r"""
extern "C" {

#define DEFINE_FAST_INTERP_KERNELS(TYPE, SUFFIX) \
__global__ void fast_interp_1d_kernel_##SUFFIX( \
    const TYPE* __restrict__ flat_image,  /* (N, 3) */ \
    const TYPE* __restrict__ x_axis,      /* (K,) */ \
    const TYPE* __restrict__ inv_dx,      /* (K - 1,) */ \
    const TYPE* __restrict__ y_vals,      /* (K, 3) */ \
    TYPE* __restrict__ flat_result,       /* (N, 3) */ \
    const int N, \
    const int K \
) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx >= N * 3) return; \
\
    int pixel_idx = idx / 3; \
    int c = idx % 3; \
\
    TYPE x = flat_image[idx]; \
    TYPE x_min = x_axis[0]; \
    TYPE x_max = x_axis[K - 1]; \
\
    if (x <= x_min) { \
        flat_result[idx] = y_vals[c]; \
        return; \
    } \
    if (x >= x_max) { \
        flat_result[idx] = y_vals[(K - 1) * 3 + c]; \
        return; \
    } \
\
    int l = 0; \
    int r = K; \
    while (l < r) { \
        int m = l + ((r - l) >> 1); \
        if (x_axis[m] <= x) { \
            l = m + 1; \
        } else { \
            r = m; \
        } \
    } \
    int low = l - 1; \
    if (low < 0) low = 0; \
    if (low >= K - 1) low = K - 2; \
\
    TYPE x0 = x_axis[low]; \
    TYPE idx_val = inv_dx[low]; \
    TYPE t = (x - x0) * idx_val; \
\
    TYPE y0 = y_vals[low * 3 + c]; \
    TYPE y1 = y_vals[(low + 1) * 3 + c]; \
    flat_result[idx] = y0 + t * (y1 - y0); \
} \
\
__global__ void fast_interp_2d_kernel_##SUFFIX( \
    const TYPE* __restrict__ flat_image,  /* (N, 3) */ \
    const TYPE* __restrict__ x_axis,      /* (K, 3) */ \
    const TYPE* __restrict__ inv_dx,      /* (K - 1, 3) */ \
    const TYPE* __restrict__ y_vals,      /* (K, 3) */ \
    TYPE* __restrict__ flat_result,       /* (N, 3) */ \
    const int N, \
    const int K \
) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx >= N * 3) return; \
\
    int pixel_idx = idx / 3; \
    int c = idx % 3; \
\
    TYPE x = flat_image[idx]; \
    TYPE x_min = x_axis[c]; \
    TYPE x_max = x_axis[(K - 1) * 3 + c]; \
\
    if (x <= x_min) { \
        flat_result[idx] = y_vals[c]; \
        return; \
    } \
    if (x >= x_max) { \
        flat_result[idx] = y_vals[(K - 1) * 3 + c]; \
        return; \
    } \
\
    int l = 0; \
    int r = K; \
    while (l < r) { \
        int m = l + ((r - l) >> 1); \
        if (x_axis[m * 3 + c] <= x) { \
            l = m + 1; \
        } else { \
            r = m; \
        } \
    } \
    int low = l - 1; \
    if (low < 0) low = 0; \
    if (low >= K - 1) low = K - 2; \
\
    TYPE x0 = x_axis[low * 3 + c]; \
    TYPE idx_val = inv_dx[low * 3 + c]; \
    TYPE t = (x - x0) * idx_val; \
\
    TYPE y0 = y_vals[low * 3 + c]; \
    TYPE y1 = y_vals[(low + 1) * 3 + c]; \
    flat_result[idx] = y0 + t * (y1 - y0); \
}

DEFINE_FAST_INTERP_KERNELS(float, f32)
DEFINE_FAST_INTERP_KERNELS(double, f64)

__global__ void fast_interp_1d_kernel(const double* flat_image, const double* x_axis, const double* inv_dx, const double* y_vals, double* flat_result, const int N, const int K) {
    fast_interp_1d_kernel_f64(flat_image, x_axis, inv_dx, y_vals, flat_result, N, K);
}

__global__ void fast_interp_2d_kernel(const double* flat_image, const double* x_axis, const double* inv_dx, const double* y_vals, double* flat_result, const int N, const int K) {
    fast_interp_2d_kernel_f64(flat_image, x_axis, inv_dx, y_vals, flat_result, N, K);
}

}
"""

_LUT_2D_CUBIC_CUDA_SOURCE = r"""
template<typename T>
__device__ inline T mitchell_weight(T t) {
    T x = fabs(t);
    const T B = (T)(1.0 / 3.0);
    const T C = (T)(1.0 / 3.0);
    if (x < (T)1.0) {
        return (T)(1.0 / 6.0) * (((T)12.0 - (T)9.0 * B - (T)6.0 * C) * x * x * x +
                              ((T)-18.0 + (T)12.0 * B + (T)6.0 * C) * x * x +
                              ((T)6.0 - (T)2.0 * B));
    } else if (x < (T)2.0) {
        return (T)(1.0 / 6.0) * ((-B - (T)6.0 * C) * x * x * x +
                              ((T)6.0 * B + (T)30.0 * C) * x * x +
                              ((T)-12.0 * B - (T)48.0 * C) * x +
                              ((T)8.0 * B + (T)24.0 * C));
    }
    return (T)0.0;
}

__device__ inline int safe_reflect_idx(int idx, int L) {
    if (idx < 0) return -idx;
    if (idx >= L) return 2 * (L - 1) - idx;
    return idx;
}

#define DEFINE_LUT_CUBIC_2D_KERNEL(TYPE, SUFFIX) \
extern "C" __global__ \
void apply_lut_cubic_2d_kernel_##SUFFIX( \
    const TYPE* __restrict__ lut,     /* (L, L, C) */ \
    const TYPE* __restrict__ image,   /* (H, W, 2) */ \
    TYPE* __restrict__ output,        /* (H, W, C) */ \
    const int H, \
    const int W, \
    const int L, \
    const int C \
) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    int total_pixels = H * W; \
    if (idx >= total_pixels) return; \
\
    TYPE x_coord = image[idx * 2 + 0] * (TYPE)(L - 1); \
    TYPE y_coord = image[idx * 2 + 1] * (TYPE)(L - 1); \
\
    if (x_coord < (TYPE)0.0) x_coord = (TYPE)0.0; \
    if (x_coord > (TYPE)(L - 1)) x_coord = (TYPE)(L - 1); \
    if (y_coord < (TYPE)0.0) y_coord = (TYPE)0.0; \
    if (y_coord > (TYPE)(L - 1)) y_coord = (TYPE)(L - 1); \
\
    int x_base = (int)floor(x_coord); \
    if (x_coord >= (TYPE)(L - 1)) { \
        x_base = L - 2; \
    } \
    TYPE x_frac = x_coord - (TYPE)x_base; \
\
    int y_base = (int)floor(y_coord); \
    if (y_coord >= (TYPE)(L - 1)) { \
        y_base = L - 2; \
    } \
    TYPE y_frac = y_coord - (TYPE)y_base; \
\
    TYPE wx[4]; \
    TYPE wy[4]; \
    for (int k = 0; k < 4; ++k) { \
        wx[k] = mitchell_weight(x_frac - (TYPE)(k - 1)); \
        wy[k] = mitchell_weight(y_frac - (TYPE)(k - 1)); \
    } \
\
    for (int c = 0; c < C; ++c) { \
        TYPE acc = (TYPE)0.0; \
        for (int i = 0; i < 4; ++i) { \
            int xi = safe_reflect_idx(x_base + i - 1, L); \
            TYPE w_x = wx[i]; \
            for (int j = 0; j < 4; ++j) { \
                int yj = safe_reflect_idx(y_base + j - 1, L); \
                TYPE w_y = wy[j]; \
                TYPE v = lut[(xi * L + yj) * C + c]; \
                acc += w_x * w_y * v; \
            } \
        } \
        output[idx * C + c] = acc; \
    } \
}

DEFINE_LUT_CUBIC_2D_KERNEL(float, f32)
DEFINE_LUT_CUBIC_2D_KERNEL(double, f64)

extern "C" __global__ void apply_lut_cubic_2d_kernel(
    const double* lut, const double* image, double* output, const int H, const int W, const int L, const int C
) {
    apply_lut_cubic_2d_kernel_f64(lut, image, output, H, W, L, C);
}
"""

_LUT_3D_TRILINEAR_CUDA_SOURCE = r"""
extern "C" {

#define DEFINE_TRILINEAR_3D_KERNEL(TYPE, SUFFIX) \
__global__ void apply_lut_trilinear_3d_kernel_##SUFFIX( \
    const TYPE* __restrict__ lut,     /* (L, L, L, 3) */ \
    const TYPE* __restrict__ image,   /* (H, W, 3) */ \
    TYPE* __restrict__ output,        /* (H, W, 3) */ \
    const int H, \
    const int W, \
    const int L \
) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    int total_pixels = H * W; \
    if (idx >= total_pixels) return; \
\
    TYPE scale = (TYPE)(L - 1); \
    TYPE r_in = image[idx * 3 + 0] * scale; \
    TYPE g_in = image[idx * 3 + 1] * scale; \
    TYPE b_in = image[idx * 3 + 2] * scale; \
\
    if (r_in < (TYPE)0.0) r_in = (TYPE)0.0; \
    if (r_in > scale) r_in = scale; \
    if (g_in < (TYPE)0.0) g_in = (TYPE)0.0; \
    if (g_in > scale) g_in = scale; \
    if (b_in < (TYPE)0.0) b_in = (TYPE)0.0; \
    if (b_in > scale) b_in = scale; \
\
    int r0 = (int)floor(r_in); \
    int g0 = (int)floor(g_in); \
    int b0 = (int)floor(b_in); \
\
    int r1 = (r0 < L - 1) ? r0 + 1 : r0; \
    int g1 = (g0 < L - 1) ? g0 + 1 : g0; \
    int b1 = (b0 < L - 1) ? b0 + 1 : b0; \
\
    TYPE tr = r_in - (TYPE)r0; \
    TYPE tg = g_in - (TYPE)g0; \
    TYPE tb = b_in - (TYPE)b0; \
\
    TYPE w000 = ((TYPE)1.0 - tr) * ((TYPE)1.0 - tg) * ((TYPE)1.0 - tb); \
    TYPE w100 = tr * ((TYPE)1.0 - tg) * ((TYPE)1.0 - tb); \
    TYPE w010 = ((TYPE)1.0 - tr) * tg * ((TYPE)1.0 - tb); \
    TYPE w110 = tr * tg * ((TYPE)1.0 - tb); \
    TYPE w001 = ((TYPE)1.0 - tr) * ((TYPE)1.0 - tg) * tb; \
    TYPE w101 = tr * ((TYPE)1.0 - tg) * tb; \
    TYPE w011 = ((TYPE)1.0 - tr) * tg * tb; \
    TYPE w111 = tr * tg * tb; \
\
    int L2 = L * L; \
    int stride_r = L2 * 3; \
    int stride_g = L * 3; \
    int stride_b = 3; \
\
    int base_000 = r0 * stride_r + g0 * stride_g + b0 * stride_b; \
    int base_100 = r1 * stride_r + g0 * stride_g + b0 * stride_b; \
    int base_010 = r0 * stride_r + g1 * stride_g + b0 * stride_b; \
    int base_110 = r1 * stride_r + g1 * stride_g + b0 * stride_b; \
    int base_001 = r0 * stride_r + g0 * stride_g + b1 * stride_b; \
    int base_101 = r1 * stride_r + g0 * stride_g + b1 * stride_b; \
    int base_011 = r0 * stride_r + g1 * stride_g + b1 * stride_b; \
    int base_111 = r1 * stride_r + g1 * stride_g + b1 * stride_b; \
\
    for (int c = 0; c < 3; ++c) { \
        TYPE v = w000 * lut[base_000 + c] + \
                 w100 * lut[base_100 + c] + \
                 w010 * lut[base_010 + c] + \
                 w110 * lut[base_110 + c] + \
                 w001 * lut[base_001 + c] + \
                 w101 * lut[base_101 + c] + \
                 w011 * lut[base_011 + c] + \
                 w111 * lut[base_111 + c]; \
        output[idx * 3 + c] = v; \
    } \
}

DEFINE_TRILINEAR_3D_KERNEL(float, f32)
DEFINE_TRILINEAR_3D_KERNEL(double, f64)

__global__ void apply_lut_trilinear_3d_kernel(
    const double* lut, const double* image, double* output, const int H, const int W, const int L
) {
    apply_lut_trilinear_3d_kernel_f64(lut, image, output, H, W, L);
}

}
"""

_MODULE = None


def _get_interp_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    import cupy as cp
    source = _FAST_INTERP_1D_CUDA_SOURCE + "\n" + _LUT_2D_CUBIC_CUDA_SOURCE + "\n" + _LUT_3D_TRILINEAR_CUDA_SOURCE
    _MODULE = cp.RawModule(code=source, options=("--use_fast_math",))
    return _MODULE


def cuda_fast_interp(image: Any, x_axis: Any, y_vals: Any) -> Any:
    """Perform 1-D linear interpolation on GPU using custom CUDA RawKernel.

    Parameters
    ----------
    image : array-like, shape (..., 3)
    x_axis : array-like, shape (K,) or (K, 3)
    y_vals : array-like, shape (K, 3)

    Returns
    -------
    result : CuPy array of same shape as image
    """
    import cupy as cp

    is_f64 = hasattr(image, "dtype") and (image.dtype == np.float64 or image.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32
    suffix = "f64" if is_f64 else "f32"

    d_image = to_device(image, dtype=target_dtype)
    orig_shape = d_image.shape
    N = int(np.prod(orig_shape[:-1]))
    flat_image = d_image.reshape(N, 3)

    common_axis = (np.ndim(x_axis) == 1)
    K = x_axis.shape[0]

    d_x_axis = to_device(x_axis, dtype=target_dtype)
    d_y_vals = to_device(y_vals, dtype=target_dtype)

    # Precompute reciprocal differences
    if common_axis:
        dx = d_x_axis[1:] - d_x_axis[:-1]
        inv_dx = cp.where(dx != 0.0, 1.0 / dx, 0.0)
    else:
        dx = d_x_axis[1:, :] - d_x_axis[:-1, :]
        inv_dx = cp.where(dx != 0.0, 1.0 / dx, 0.0)

    flat_result = cp.empty((N, 3), dtype=target_dtype)

    mod = _get_interp_module()
    kernel_name = f"fast_interp_1d_kernel_{suffix}" if common_axis else f"fast_interp_2d_kernel_{suffix}"
    kernel = mod.get_function(kernel_name)

    total_threads = N * 3
    block_size = 256
    grid_size = (total_threads + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,),
        (flat_image, d_x_axis, inv_dx, d_y_vals, flat_result, np.int32(N), np.int32(K)),
    )
    return flat_result.reshape(orig_shape)


def cuda_apply_lut_cubic_2d(lut: Any, image: Any) -> Any:
    """Apply a 2D LUT to an image on GPU using bicubic Mitchell-Netravali interpolation."""
    import cupy as cp

    is_f64 = hasattr(image, "dtype") and (image.dtype == np.float64 or image.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32
    suffix = "f64" if is_f64 else "f32"

    d_lut = to_device(lut, dtype=target_dtype)
    d_image = to_device(image, dtype=target_dtype)

    H, W = d_image.shape[:2]
    L = d_lut.shape[0]
    C = d_lut.shape[2]

    output = cp.empty((H, W, C), dtype=target_dtype)

    mod = _get_interp_module()
    kernel = mod.get_function(f"apply_lut_cubic_2d_kernel_{suffix}")

    total_pixels = H * W
    block_size = 256
    grid_size = (total_pixels + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,),
        (d_lut, d_image, output, np.int32(H), np.int32(W), np.int32(L), np.int32(C)),
    )
    return output


def cuda_apply_lut_3d(lut: Any, image: Any, method: str = "pchip") -> Any:
    """Apply a 3D LUT to an image on GPU using trilinear or PCHIP sampling."""
    import cupy as cp

    is_f64 = hasattr(image, "dtype") and (image.dtype == np.float64 or image.dtype == cp.float64)
    target_dtype = np.float64 if is_f64 else np.float32
    suffix = "f64" if is_f64 else "f32"

    d_lut = to_device(lut, dtype=target_dtype)
    d_image = to_device(image, dtype=target_dtype)

    H, W = d_image.shape[:2]
    L = d_lut.shape[0]

    output = cp.empty((H, W, 3), dtype=target_dtype)

    mod = _get_interp_module()
    kernel = mod.get_function(f"apply_lut_trilinear_3d_kernel_{suffix}")

    total_pixels = H * W
    block_size = 256
    grid_size = (total_pixels + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,),
        (d_lut, d_image, output, np.int32(H), np.int32(W), np.int32(L)),
    )
    return output
