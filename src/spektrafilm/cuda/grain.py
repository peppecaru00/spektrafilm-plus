"""CUDA-accelerated stochastic film grain simulation.

Adapted from the proven high-performance CUDA implementation in spektrafilm-ofx
with physical film coordinate spatial anchoring and variance damping.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from spektrafilm.cuda.device import is_cuda_available, to_device, to_cpu
from spektrafilm.cuda.filters import cuda_gaussian_filter
from spektrafilm.cuda.interp import cuda_fast_interp


CUDA_GRAIN_SOURCE = r"""
typedef unsigned int uint32_t;
typedef int int32_t;

__device__ __forceinline__ uint32_t grainHash(uint32_t x) {
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

__device__ __forceinline__ float grainRand01(uint32_t seed) {
  return (static_cast<float>(grainHash(seed) & 0x00ffffffu) + 0.5f) / 16777216.0f;
}

__device__ __forceinline__ float grainGaussian(uint32_t seed) {
  const float u1 = fmaxf(grainRand01(seed), 1.0e-6f);
  const float u2 = grainRand01(seed ^ 0x9e3779b9u);
  constexpr float twoPi = 6.28318530718f;
  return sqrtf(-2.0f * logf(u1)) * cosf(twoPi * u2);
}

__device__ __forceinline__ uint32_t grainFilmCellSeed(float filmUmX, float filmUmY, float cellSizeUm, uint32_t seed) {
  const float safeCellSize = fmaxf(cellSizeUm, 1.0e-4f);
  const int32_t cellX = static_cast<int32_t>(floorf(filmUmX / safeCellSize));
  const int32_t cellY = static_cast<int32_t>(floorf(filmUmY / safeCellSize));
  return grainHash(seed ^ (static_cast<uint32_t>(cellX) * 0x1f123bb5u) ^ (static_cast<uint32_t>(cellY) * 0x5f356495u));
}

__device__ __forceinline__ float grainParticleDevelopedDensity(
  float density,
  float densityMax,
  float particles,
  float uniformity,
  float blurDamping,
  uint32_t seed
) {
  const float safeDensityMax = fmaxf(densityMax, 1.0e-6f);
  const float safeParticles = fmaxf(particles, 1.0e-3f);
  const float probability = fminf(fmaxf(density / safeDensityMax, 1.0e-6f), 1.0f - 1.0e-6f);
  const float saturation = fmaxf(1.0f - probability * uniformity * (1.0f - 1.0e-6f), 1.0e-6f);
  const float expectedSeeds = safeParticles / saturation;
  const float expectedDeveloped = expectedSeeds * probability;
  const float variance = fmaxf(expectedSeeds * probability, 1.0e-6f);
  const float developed = fminf(fmaxf(
    expectedDeveloped + sqrtf(variance) * grainGaussian(seed) * blurDamping,
    0.0f),
    expectedSeeds);
  return developed * (safeDensityMax / safeParticles) * saturation;
}

template <typename T>
__device__ __forceinline__ void singleLayerGrain_impl(
    const T* __restrict__ densityIn,
    const T* __restrict__ densityMax,
    const T* __restrict__ particlesPerPixel,
    const T* __restrict__ particleArea,
    const T* __restrict__ uniformity,
    const T* __restrict__ densityMin,
    T* __restrict__ output,
    int width,
    int height,
    int nSubLayers,
    float pixelSizeUm,
    float blurDamping,
    float originXUm,
    float originYUm,
    uint32_t baseSeed
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float filmUmX = originXUm + (static_cast<float>(x) + 0.5f) * pixelSizeUm;
    float filmUmY = originYUm + (static_cast<float>(y) + 0.5f) * pixelSizeUm;

    size_t pixelIdx = static_cast<size_t>(y) * width + x;
    float subInv = 1.0f / fmaxf(static_cast<float>(nSubLayers), 1.0f);

    for (int ch = 0; ch < 3; ++ch) {
        float chMin = static_cast<float>(densityMin[ch]);
        float srcDensity = fmaxf(static_cast<float>(densityIn[pixelIdx * 3 + ch]) + chMin, 0.0f);
        float dMax = static_cast<float>(densityMax[ch]);
        float nParticles = static_cast<float>(particlesPerPixel[ch]);
        float pArea = static_cast<float>(particleArea[ch]);
        float chUniformity = static_cast<float>(uniformity[ch]);

        float accum = 0.0f;
        for (int sl = 0; sl < nSubLayers; ++sl) {
            uint32_t cellSeed = grainFilmCellSeed(
                filmUmX, filmUmY,
                sqrtf(fmaxf(pArea, 1.0e-4f)),
                baseSeed ^ (static_cast<uint32_t>(ch) * 0x85ebca6bu) ^ (static_cast<uint32_t>(sl) * 0x9e3779b9u)
            );
            accum += grainParticleDevelopedDensity(
                srcDensity, dMax, nParticles, chUniformity,
                blurDamping, cellSeed
            );
        }
        accum *= subInv;
        accum -= chMin;
        output[pixelIdx * 3 + ch] = static_cast<T>(fmaxf(accum, 0.0f));
    }
}

template <typename T>
__device__ __forceinline__ void multilayerGrain_impl(
    const T* __restrict__ layers,
    const T* __restrict__ densityMaxLayers,
    const T* __restrict__ particlesPerPixel,
    const T* __restrict__ particleAreaLayers,
    const T* __restrict__ uniformity,
    const T* __restrict__ densityMin,
    T* __restrict__ output,
    int width,
    int height,
    float pixelSizeUm,
    float blurDamping,
    float microScale,
    float microSigma,
    float originXUm,
    float originYUm,
    uint32_t baseSeed
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float filmUmX = originXUm + (static_cast<float>(x) + 0.5f) * pixelSizeUm;
    float filmUmY = originYUm + (static_cast<float>(y) + 0.5f) * pixelSizeUm;

    float microFactor[3] = {1.0f, 1.0f, 1.0f};
    if (microSigma > 0.05f && microScale > 0.0f) {
        float fSigma = microSigma;
        uint32_t microSeed = grainFilmCellSeed(
            filmUmX, filmUmY,
            fmaxf(microScale, 1.0e-4f),
            baseSeed ^ 0x23d3c1f1u ^ 0xa349b329u
        );
        float logSigma = sqrtf(logf(1.0f + fSigma * fSigma));
        float logMean = -0.5f * logSigma * logSigma;
        microFactor[0] = expf(logMean + logSigma * grainGaussian(microSeed ^ 0x165667b1u));
        microFactor[1] = expf(logMean + logSigma * grainGaussian(microSeed ^ 0x27d4eb2du));
        microFactor[2] = expf(logMean + logSigma * grainGaussian(microSeed ^ 0x85ebca6bu));
    }

    size_t pixelIdx = static_cast<size_t>(y) * width + x;

    for (int ch = 0; ch < 3; ++ch) {
        float chAccum = 0.0f;
        float chUniformity = static_cast<float>(uniformity[ch]);

        for (int sl = 0; sl < 3; ++sl) {
            size_t layerIdx = (pixelIdx * 3 + sl) * 3 + ch;
            float layerDensity = static_cast<float>(layers[layerIdx]);
            float dMax = static_cast<float>(densityMaxLayers[sl * 3 + ch]);
            float nParticles = static_cast<float>(particlesPerPixel[sl * 3 + ch]);
            float pArea = static_cast<float>(particleAreaLayers[sl * 3 + ch]);

            uint32_t cellSeed = grainFilmCellSeed(
                filmUmX, filmUmY,
                sqrtf(fmaxf(pArea, 1.0e-4f)),
                baseSeed ^ (static_cast<uint32_t>(ch) * 0x85ebca6bu) ^ (static_cast<uint32_t>(sl) * 0xc2b2ae35u)
            );

            float developed = grainParticleDevelopedDensity(
                layerDensity,
                dMax,
                nParticles,
                chUniformity,
                blurDamping,
                cellSeed
            );
            chAccum += developed;
        }

        chAccum *= microFactor[ch];
        chAccum -= static_cast<float>(densityMin[ch]);

        output[pixelIdx * 3 + ch] = static_cast<T>(fmaxf(chAccum, 0.0f));
    }
}

template <typename T>
__device__ __forceinline__ float interp1d(
    float x,
    const T* __restrict__ x_axis,
    const T* __restrict__ y_vals,
    int K
) {
    if (x <= static_cast<float>(x_axis[0])) return static_cast<float>(y_vals[0]);
    if (x >= static_cast<float>(x_axis[K - 1])) return static_cast<float>(y_vals[K - 1]);

    int l = 0, r = K;
    while (l < r) {
        int m = l + ((r - l) >> 1);
        if (static_cast<float>(x_axis[m]) <= x) l = m + 1;
        else r = m;
    }
    int low = l - 1;
    if (low < 0) low = 0;
    if (low >= K - 1) low = K - 2;

    float x0 = static_cast<float>(x_axis[low]);
    float x1 = static_cast<float>(x_axis[low + 1]);
    float dx = x1 - x0;
    float t = (dx != 0.0f) ? (x - x0) / dx : 0.0f;
    float y0 = static_cast<float>(y_vals[low]);
    float y1 = static_cast<float>(y_vals[low + 1]);
    return y0 + t * (y1 - y0);
}

template <typename T>
__device__ __forceinline__ void fusedMultilayerGrain_impl(
    const T* __restrict__ densityCmy,
    const T* __restrict__ densityCurvesContig,
    const T* __restrict__ densityCurvesLayersContig,
    const T* __restrict__ densityMaxLayers,
    const T* __restrict__ particlesPerPixel,
    const T* __restrict__ particleAreaLayers,
    const T* __restrict__ uniformity,
    const T* __restrict__ densityMin,
    const T* __restrict__ densityMinLayers,
    T* __restrict__ output,
    int width,
    int height,
    int K,
    int positiveFilm,
    float pixelSizeUm,
    float blurDamping,
    float microScale,
    float microSigma,
    float originXUm,
    float originYUm,
    uint32_t baseSeed
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float filmUmX = originXUm + (static_cast<float>(x) + 0.5f) * pixelSizeUm;
    float filmUmY = originYUm + (static_cast<float>(y) + 0.5f) * pixelSizeUm;

    float microFactor[3] = {1.0f, 1.0f, 1.0f};
    if (microSigma > 0.05f && microScale > 0.0f) {
        float fSigma = microSigma;
        uint32_t microSeed = grainFilmCellSeed(
            filmUmX, filmUmY,
            fmaxf(microScale, 1.0e-4f),
            baseSeed ^ 0x23d3c1f1u ^ 0xa349b329u
        );
        float logSigma = sqrtf(logf(1.0f + fSigma * fSigma));
        float logMean = -0.5f * logSigma * logSigma;
        microFactor[0] = expf(logMean + logSigma * grainGaussian(microSeed ^ 0x165667b1u));
        microFactor[1] = expf(logMean + logSigma * grainGaussian(microSeed ^ 0x27d4eb2du));
        microFactor[2] = expf(logMean + logSigma * grainGaussian(microSeed ^ 0x85ebca6bu));
    }

    size_t pixelIdx = static_cast<size_t>(y) * width + x;

    for (int ch = 0; ch < 3; ++ch) {
        float chVal = static_cast<float>(densityCmy[pixelIdx * 3 + ch]);
        float chAccum = 0.0f;
        float chUniformity = static_cast<float>(uniformity[ch]);

        for (int sl = 0; sl < 3; ++sl) {
            float queryX = positiveFilm ? -chVal : chVal;
            float layerDensity = interp1d<T>(
                queryX,
                &densityCurvesContig[ch * K],
                &densityCurvesLayersContig[(ch * 3 + sl) * K],
                K
            );
            layerDensity += static_cast<float>(densityMinLayers[sl * 3 + ch]);

            float dMax = static_cast<float>(densityMaxLayers[sl * 3 + ch]);
            float nParticles = static_cast<float>(particlesPerPixel[sl * 3 + ch]);
            float pArea = static_cast<float>(particleAreaLayers[sl * 3 + ch]);

            uint32_t cellSeed = grainFilmCellSeed(
                filmUmX, filmUmY,
                sqrtf(fmaxf(pArea, 1.0e-4f)),
                baseSeed ^ (static_cast<uint32_t>(ch) * 0x85ebca6bu) ^ (static_cast<uint32_t>(sl) * 0xc2b2ae35u)
            );

            float developed = grainParticleDevelopedDensity(
                layerDensity,
                dMax,
                nParticles,
                chUniformity,
                blurDamping,
                cellSeed
            );
            chAccum += developed;
        }

        chAccum *= microFactor[ch];
        chAccum -= static_cast<float>(densityMin[ch]);
        output[pixelIdx * 3 + ch] = static_cast<T>(fmaxf(chAccum, 0.0f));
    }
}

extern "C" {

__global__ void singleLayerGrainKernel_f32(
    const float* __restrict__ densityIn,
    const float* __restrict__ densityMax,
    const float* __restrict__ particlesPerPixel,
    const float* __restrict__ particleArea,
    const float* __restrict__ uniformity,
    const float* __restrict__ densityMin,
    float* __restrict__ output,
    int width,
    int height,
    int nSubLayers,
    float pixelSizeUm,
    float blurDamping,
    float originXUm,
    float originYUm,
    uint32_t baseSeed
) {
    singleLayerGrain_impl<float>(
        densityIn, densityMax, particlesPerPixel, particleArea, uniformity, densityMin, output,
        width, height, nSubLayers, pixelSizeUm, blurDamping, originXUm, originYUm, baseSeed
    );
}

__global__ void singleLayerGrainKernel_f64(
    const double* __restrict__ densityIn,
    const double* __restrict__ densityMax,
    const double* __restrict__ particlesPerPixel,
    const double* __restrict__ particleArea,
    const double* __restrict__ uniformity,
    const double* __restrict__ densityMin,
    double* __restrict__ output,
    int width,
    int height,
    int nSubLayers,
    double pixelSizeUm,
    double blurDamping,
    double originXUm,
    double originYUm,
    uint32_t baseSeed
) {
    singleLayerGrain_impl<double>(
        densityIn, densityMax, particlesPerPixel, particleArea, uniformity, densityMin, output,
        width, height, nSubLayers, static_cast<float>(pixelSizeUm), static_cast<float>(blurDamping),
        static_cast<float>(originXUm), static_cast<float>(originYUm), baseSeed
    );
}

__global__ void singleLayerGrainKernel(
    const double* __restrict__ densityIn,
    const double* __restrict__ densityMax,
    const double* __restrict__ particlesPerPixel,
    const double* __restrict__ particleArea,
    const double* __restrict__ uniformity,
    const double* __restrict__ densityMin,
    double* __restrict__ output,
    int width,
    int height,
    int nSubLayers,
    double pixelSizeUm,
    double blurDamping,
    double originXUm,
    double originYUm,
    uint32_t baseSeed
) {
    singleLayerGrain_impl<double>(
        densityIn, densityMax, particlesPerPixel, particleArea, uniformity, densityMin, output,
        width, height, nSubLayers, static_cast<float>(pixelSizeUm), static_cast<float>(blurDamping),
        static_cast<float>(originXUm), static_cast<float>(originYUm), baseSeed
    );
}

__global__ void multilayerGrainKernel_f32(
    const float* __restrict__ layers,
    const float* __restrict__ densityMaxLayers,
    const float* __restrict__ particlesPerPixel,
    const float* __restrict__ particleAreaLayers,
    const float* __restrict__ uniformity,
    const float* __restrict__ densityMin,
    float* __restrict__ output,
    int width,
    int height,
    float pixelSizeUm,
    float blurDamping,
    float microScale,
    float microSigma,
    float originXUm,
    float originYUm,
    uint32_t baseSeed
) {
    multilayerGrain_impl<float>(
        layers, densityMaxLayers, particlesPerPixel, particleAreaLayers, uniformity, densityMin, output,
        width, height, pixelSizeUm, blurDamping, microScale, microSigma, originXUm, originYUm, baseSeed
    );
}

__global__ void multilayerGrainKernel_f64(
    const double* __restrict__ layers,
    const double* __restrict__ densityMaxLayers,
    const double* __restrict__ particlesPerPixel,
    const double* __restrict__ particleAreaLayers,
    const double* __restrict__ uniformity,
    const double* __restrict__ densityMin,
    double* __restrict__ output,
    int width,
    int height,
    double pixelSizeUm,
    double blurDamping,
    double microScale,
    double microSigma,
    double originXUm,
    double originYUm,
    uint32_t baseSeed
) {
    multilayerGrain_impl<double>(
        layers, densityMaxLayers, particlesPerPixel, particleAreaLayers, uniformity, densityMin, output,
        width, height, static_cast<float>(pixelSizeUm), static_cast<float>(blurDamping),
        static_cast<float>(microScale), static_cast<float>(microSigma),
        static_cast<float>(originXUm), static_cast<float>(originYUm), baseSeed
    );
}

__global__ void multilayerGrainKernel(
    const double* __restrict__ layers,
    const double* __restrict__ densityMaxLayers,
    const double* __restrict__ particlesPerPixel,
    const double* __restrict__ particleAreaLayers,
    const double* __restrict__ uniformity,
    const double* __restrict__ densityMin,
    double* __restrict__ output,
    int width,
    int height,
    double pixelSizeUm,
    double blurDamping,
    double microScale,
    double microSigma,
    double originXUm,
    double originYUm,
    uint32_t baseSeed
) {
    multilayerGrain_impl<double>(
        layers, densityMaxLayers, particlesPerPixel, particleAreaLayers, uniformity, densityMin, output,
        width, height, static_cast<float>(pixelSizeUm), static_cast<float>(blurDamping),
        static_cast<float>(microScale), static_cast<float>(microSigma),
        static_cast<float>(originXUm), static_cast<float>(originYUm), baseSeed
    );
}

__global__ void fusedMultilayerGrainKernel_f32(
    const float* __restrict__ densityCmy,
    const float* __restrict__ densityCurvesContig,
    const float* __restrict__ densityCurvesLayersContig,
    const float* __restrict__ densityMaxLayers,
    const float* __restrict__ particlesPerPixel,
    const float* __restrict__ particleAreaLayers,
    const float* __restrict__ uniformity,
    const float* __restrict__ densityMin,
    const float* __restrict__ densityMinLayers,
    float* __restrict__ output,
    int width,
    int height,
    int K,
    int positiveFilm,
    float pixelSizeUm,
    float blurDamping,
    float microScale,
    float microSigma,
    float originXUm,
    float originYUm,
    uint32_t baseSeed
) {
    fusedMultilayerGrain_impl<float>(
        densityCmy, densityCurvesContig, densityCurvesLayersContig, densityMaxLayers, particlesPerPixel,
        particleAreaLayers, uniformity, densityMin, densityMinLayers, output,
        width, height, K, positiveFilm, pixelSizeUm, blurDamping, microScale, microSigma, originXUm, originYUm, baseSeed
    );
}

__global__ void fusedMultilayerGrainKernel_f64(
    const double* __restrict__ densityCmy,
    const double* __restrict__ densityCurvesContig,
    const double* __restrict__ densityCurvesLayersContig,
    const double* __restrict__ densityMaxLayers,
    const double* __restrict__ particlesPerPixel,
    const double* __restrict__ particleAreaLayers,
    const double* __restrict__ uniformity,
    const double* __restrict__ densityMin,
    const double* __restrict__ densityMinLayers,
    double* __restrict__ output,
    int width,
    int height,
    int K,
    int positiveFilm,
    double pixelSizeUm,
    double blurDamping,
    double microScale,
    double microSigma,
    double originXUm,
    double originYUm,
    uint32_t baseSeed
) {
    fusedMultilayerGrain_impl<double>(
        densityCmy, densityCurvesContig, densityCurvesLayersContig, densityMaxLayers, particlesPerPixel,
        particleAreaLayers, uniformity, densityMin, densityMinLayers, output,
        width, height, K, positiveFilm, static_cast<float>(pixelSizeUm), static_cast<float>(blurDamping),
        static_cast<float>(microScale), static_cast<float>(microSigma),
        static_cast<float>(originXUm), static_cast<float>(originYUm), baseSeed
    );
}

__global__ void fusedMultilayerGrainKernel(
    const double* __restrict__ densityCmy,
    const double* __restrict__ densityCurvesContig,
    const double* __restrict__ densityCurvesLayersContig,
    const double* __restrict__ densityMaxLayers,
    const double* __restrict__ particlesPerPixel,
    const double* __restrict__ particleAreaLayers,
    const double* __restrict__ uniformity,
    const double* __restrict__ densityMin,
    const double* __restrict__ densityMinLayers,
    double* __restrict__ output,
    int width,
    int height,
    int K,
    int positiveFilm,
    double pixelSizeUm,
    double blurDamping,
    double microScale,
    double microSigma,
    double originXUm,
    double originYUm,
    uint32_t baseSeed
) {
    fusedMultilayerGrain_impl<double>(
        densityCmy, densityCurvesContig, densityCurvesLayersContig, densityMaxLayers, particlesPerPixel,
        particleAreaLayers, uniformity, densityMin, densityMinLayers, output,
        width, height, K, positiveFilm, static_cast<float>(pixelSizeUm), static_cast<float>(blurDamping),
        static_cast<float>(microScale), static_cast<float>(microSigma),
        static_cast<float>(originXUm), static_cast<float>(originYUm), baseSeed
    );
}

}
"""

_GRAIN_MODULE = None


def _get_grain_module():
    global _GRAIN_MODULE
    if _GRAIN_MODULE is None:
        import cupy as cp
        _GRAIN_MODULE = cp.RawModule(code=CUDA_GRAIN_SOURCE)
    return _GRAIN_MODULE


def cuda_layer_particle_model(
    density: Any,
    density_max: float = 2.2,
    n_particles_per_pixel: float = 10.0,
    grain_uniformity: float = 0.98,
    seed: int | None = None,
    blur_particle: float = 0.0,
) -> Any:
    """Evaluate stochastic particle emulsion model on GPU.

    Simulates the stochastic development of silver halide grains via
    Poisson-distributed initial seeds and Binomial development trials.
    """
    import cupy as cp

    target_dtype = cp.float32 if getattr(density, "dtype", None) in (np.float32, cp.float32) else cp.float64
    d_density = to_device(density, dtype=target_dtype)

    if seed is not None:
        cp.random.seed(seed)

    prob = cp.clip(d_density / density_max, 1e-6, 1.0 - 1e-6)
    od_particle = density_max / n_particles_per_pixel

    saturation = 1.0 - prob * grain_uniformity * (1.0 - 1e-6)
    lam = n_particles_per_pixel / saturation

    seeds = cp.random.poisson(lam)
    grain = cp.random.binomial(seeds, prob)
    grain_f = grain.astype(target_dtype) * (od_particle * saturation)

    if blur_particle > 0.0:
        grain_f = cuda_gaussian_filter(grain_f, blur_particle * np.sqrt(od_particle))

    return grain_f


def cuda_add_micro_structure(density_cmy_out: Any, micro_structure: Any, pixel_size_um: float) -> Any:
    """Add lognormal micro-structure spatial clumping to density on GPU."""
    import cupy as cp

    blur_px = float(micro_structure[0]) / pixel_size_um
    sigma = float(micro_structure[1]) * 0.001 / pixel_size_um  # in nm -> um

    if sigma > 0.05:
        sigma2 = float(np.log(1.0 + sigma ** 2))
        mu = -0.5 * sigma2
        std_norm = float(np.sqrt(sigma2))

        clumping = cp.exp(cp.random.normal(mu, std_norm, size=density_cmy_out.shape))
        if blur_px > 0.4:
            clumping = cuda_gaussian_filter(clumping, blur_px)
        density_cmy_out = density_cmy_out * clumping

    return density_cmy_out


def cuda_interp_density_cmy_layers(
    density_cmy: Any,
    density_curves: Any,
    density_curves_layers: Any,
    positive_film: bool = False,
) -> Any:
    """Interpolate density into 3x3 emulsion sub-layers on GPU."""
    import cupy as cp

    target_dtype = cp.float32 if getattr(density_cmy, "dtype", None) in (np.float32, cp.float32) else cp.float64
    d_cmy = to_device(density_cmy, dtype=target_dtype)
    H, W = d_cmy.shape[:2]
    out_layers = cp.zeros((H, W, 3, 3), dtype=target_dtype)

    dc = np.asarray(density_curves)
    dcl = np.asarray(density_curves_layers)

    for ch in range(3):
        ch_density = d_cmy[:, :, ch, None]
        repeated_density = cp.repeat(ch_density, 3, axis=2)

        if positive_film:
            x_axis = -dc[:, ch]
            y_vals = dcl[:, :, ch]
            out_layers[:, :, :, ch] = cuda_fast_interp(-repeated_density, x_axis, y_vals)
        else:
            x_axis = dc[:, ch]
            y_vals = dcl[:, :, ch]
            out_layers[:, :, :, ch] = cuda_fast_interp(repeated_density, x_axis, y_vals)

    return out_layers


def cuda_apply_grain_to_density_layers(
    density_cmy_layers: Any,
    density_max_layers: Any,
    pixel_size_um: float = 10.0,
    particle_area_um2: float = 0.2,
    particle_scale: Any = (1.0, 0.8, 3.0),
    particle_scale_layers: Any = (3.0, 1.0, 0.3),
    density_min: Any = (0.03, 0.06, 0.04),
    grain_uniformity: Any = (0.98, 0.98, 0.98),
    grain_blur: float = 1.0,
    grain_blur_dye_clouds_um: float = 1.0,
    grain_micro_structure: Any = (0.1, 30.0),
    fixed_seed: Any = None,
    origin_x_um: float = 0.0,
    origin_y_um: float = 0.0,
) -> Any:
    """Run full multilayer grain simulation on GPU with spatial coordinate anchoring."""
    import cupy as cp

    module = _get_grain_module()
    target_dtype = cp.float32 if getattr(density_cmy_layers, "dtype", None) in (np.float32, cp.float32) else cp.float64
    is_f32 = (target_dtype == cp.float32)
    f_type = np.float32 if is_f32 else np.float64
    kernel_name = "multilayerGrainKernel_f32" if is_f32 else "multilayerGrainKernel_f64"
    multi_kernel = module.get_function(kernel_name)

    H, W = density_cmy_layers.shape[:2]
    max_layers = np.asarray(density_max_layers)
    density_max_total = np.sum(max_layers, axis=0)
    density_max_fractions = max_layers / density_max_total[None, :]
    density_min_layers = density_max_fractions * np.array(density_min)[None, :]
    density_max_layers_total = max_layers + density_min_layers

    pixel_area_um2 = pixel_size_um ** 2
    particle_area_um2_layers = (
        particle_area_um2
        * np.array(particle_scale)[None, :]
        * np.array(particle_scale_layers)[:, None]
    )
    n_particles_per_pixel = pixel_area_um2 * density_max_fractions / particle_area_um2_layers

    d_layers = cp.ascontiguousarray(density_cmy_layers + cp.asarray(density_min_layers, dtype=target_dtype), dtype=target_dtype)
    d_max_layers = cp.ascontiguousarray(cp.asarray(density_max_layers_total, dtype=target_dtype), dtype=target_dtype)
    d_particles = cp.ascontiguousarray(cp.asarray(n_particles_per_pixel, dtype=target_dtype), dtype=target_dtype)
    d_p_area = cp.ascontiguousarray(cp.asarray(particle_area_um2_layers, dtype=target_dtype), dtype=target_dtype)
    d_uniformity = cp.ascontiguousarray(cp.asarray(grain_uniformity, dtype=target_dtype), dtype=target_dtype)
    d_density_min = cp.ascontiguousarray(cp.asarray(density_min, dtype=target_dtype), dtype=target_dtype)
    d_output = cp.empty((H, W, 3), dtype=target_dtype)

    blur_damping = 1.0 / np.sqrt(1.0 + 0.35 * grain_blur + 0.12 * grain_blur_dye_clouds_um)
    micro_scale = float(grain_micro_structure[0])
    micro_sigma = float(grain_micro_structure[1]) * 0.001 / pixel_size_um
    base_seed = np.uint32(1 if fixed_seed is None else int(fixed_seed))

    threads = (16, 16)
    blocks = ((W + 15) // 16, (H + 15) // 16)
    multi_kernel(
        blocks, threads,
        (
            d_layers, d_max_layers, d_particles, d_p_area, d_uniformity, d_density_min, d_output,
            int(W), int(H), f_type(pixel_size_um), f_type(blur_damping),
            f_type(micro_scale), f_type(micro_sigma), f_type(origin_x_um), f_type(origin_y_um),
            base_seed,
        )
    )

    if grain_blur > 0.4:
        d_output = cuda_gaussian_filter(d_output, grain_blur)

    return d_output


def cuda_apply_grain(
    density_cmy: Any,
    pixel_size_um: float,
    grain_params: Any,
    density_curves: Any,
    density_curves_layers: Any,
    profile_type: str,
    bypass_grain: bool = False,
    origin_x_um: float = 0.0,
    origin_y_um: float = 0.0,
    seed: int | None = None,
) -> Any:
    """Dispatch grain application on GPU with physical film coordinate anchoring."""
    if not grain_params.active or bypass_grain:
        return density_cmy

    import cupy as cp
    target_dtype = cp.float32 if getattr(density_cmy, "dtype", None) in (np.float32, cp.float32) else cp.float64
    is_f32 = (target_dtype == cp.float32)
    f_type = np.float32 if is_f32 else np.float64
    d_cmy = to_device(density_cmy, dtype=target_dtype)

    effective_seed = seed if seed is not None else getattr(grain_params, "seed", None)
    base_seed = np.uint32(1 if effective_seed is None else (int(effective_seed) & 0xFFFFFFFF))

    if not grain_params.sublayers_active:
        module = _get_grain_module()
        kernel_name = "singleLayerGrainKernel_f32" if is_f32 else "singleLayerGrainKernel_f64"
        single_kernel = module.get_function(kernel_name)

        H, W = d_cmy.shape[:2]
        density_max = np.nanmax(density_curves, axis=0)
        d_min = np.array(grain_params.density_min)
        d_max_curves = density_max + d_min
        pixel_area = pixel_size_um ** 2
        particle_area = grain_params.particle_area_um2 * np.array(grain_params.particle_scale)
        n_particles = pixel_area / particle_area
        n_sub = grain_params.n_sub_layers
        if n_sub > 1:
            n_particles /= n_sub

        d_output = cp.empty((H, W, 3), dtype=target_dtype)
        blur_damping = 1.0 / np.sqrt(1.0 + 0.35 * grain_params.blur + 0.12 * grain_params.blur_dye_clouds_um)

        threads = (16, 16)
        blocks = ((W + 15) // 16, (H + 15) // 16)
        single_kernel(
            blocks, threads,
            (
                cp.ascontiguousarray(d_cmy),
                cp.ascontiguousarray(cp.asarray(d_max_curves, dtype=target_dtype)),
                cp.ascontiguousarray(cp.asarray(n_particles, dtype=target_dtype)),
                cp.ascontiguousarray(cp.asarray(particle_area, dtype=target_dtype)),
                cp.ascontiguousarray(cp.asarray(grain_params.uniformity, dtype=target_dtype)),
                cp.ascontiguousarray(cp.asarray(d_min, dtype=target_dtype)),
                d_output,
                int(W), int(H), int(n_sub), f_type(pixel_size_um), f_type(blur_damping),
                f_type(origin_x_um), f_type(origin_y_um), base_seed,
            )
        )
        if grain_params.blur > 0.4:
            d_output = cuda_gaussian_filter(d_output, grain_params.blur)
        return d_output

    # Multilayer path
    module = _get_grain_module()
    kernel_name = "fusedMultilayerGrainKernel_f32" if is_f32 else "fusedMultilayerGrainKernel_f64"
    fused_kernel = module.get_function(kernel_name)

    H, W = d_cmy.shape[:2]
    positive_film = int(profile_type == "positive")

    dc = np.asarray(density_curves)
    dcl = np.asarray(density_curves_layers)
    K = dc.shape[0]

    dc_input = -dc.T if positive_film else dc.T
    dc_contig = cp.ascontiguousarray(cp.asarray(dc_input, dtype=target_dtype))
    dcl_contig = cp.ascontiguousarray(cp.asarray(dcl.transpose(2, 1, 0), dtype=target_dtype))

    max_layers = np.asarray(np.nanmax(dcl, axis=0))
    density_max_total = np.sum(max_layers, axis=0)
    density_max_fractions = max_layers / density_max_total[None, :]
    density_min_layers = density_max_fractions * np.array(grain_params.density_min)[None, :]
    density_max_layers_total = max_layers + density_min_layers

    pixel_area_um2 = pixel_size_um ** 2
    particle_area_um2_layers = (
        grain_params.particle_area_um2
        * np.array(grain_params.particle_scale)[None, :]
        * np.array(grain_params.particle_scale_layers)[:, None]
    )
    n_particles_per_pixel = pixel_area_um2 * density_max_fractions / particle_area_um2_layers

    d_max_layers = cp.ascontiguousarray(cp.asarray(density_max_layers_total, dtype=target_dtype))
    d_particles = cp.ascontiguousarray(cp.asarray(n_particles_per_pixel, dtype=target_dtype))
    d_p_area = cp.ascontiguousarray(cp.asarray(particle_area_um2_layers, dtype=target_dtype))
    d_uniformity = cp.ascontiguousarray(cp.asarray(grain_params.uniformity, dtype=target_dtype))
    d_density_min = cp.ascontiguousarray(cp.asarray(grain_params.density_min, dtype=target_dtype))
    d_density_min_layers = cp.ascontiguousarray(cp.asarray(density_min_layers, dtype=target_dtype))

    blur_damping = 1.0 / np.sqrt(1.0 + 0.35 * grain_params.blur + 0.12 * grain_params.blur_dye_clouds_um)
    micro_scale = float(grain_params.micro_structure[0])
    micro_sigma = float(grain_params.micro_structure[1]) * 0.001 / pixel_size_um

    d_output = cp.empty((H, W, 3), dtype=target_dtype)
    threads = (16, 16)
    blocks = ((W + 15) // 16, (H + 15) // 16)

    fused_kernel(
        blocks, threads,
        (
            cp.ascontiguousarray(d_cmy), dc_contig, dcl_contig, d_max_layers, d_particles, d_p_area,
            d_uniformity, d_density_min, d_density_min_layers, d_output,
            np.int32(W), np.int32(H), np.int32(K), np.int32(positive_film),
            f_type(pixel_size_um), f_type(blur_damping),
            f_type(micro_scale), f_type(micro_sigma),
            f_type(origin_x_um), f_type(origin_y_um),
            base_seed,
        )
    )

    if grain_params.blur > 0.4:
        d_output = cuda_gaussian_filter(d_output, grain_params.blur)

    return d_output


