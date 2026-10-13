from __future__ import annotations

import colour
import numpy as np
from opt_einsum import contract

from spektrafilm.config import STANDARD_OBSERVER_CMFS
from spektrafilm.model.diffusion import apply_gaussian_blur, apply_unsharp_mask
from spektrafilm.model.develop import compute_density_spectral
from spektrafilm.model.glare import add_glare
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.utils.conversions import density_to_light
from spektrafilm.utils.gamut_compression import compress_rgb


class ScanningStage:
    def __init__(
        self,
        film,
        film_render_params,
        print_profile,
        print_render_params,
        scanner_params,
        io_params,
        settings_params,
        lut_service,
        color_reference_service,
    ):
        self._film = film
        self._film_render = film_render_params
        self._print = print_profile
        self._print_render = print_render_params
        self._scanner = scanner_params
        self._io = io_params
        self._settings = settings_params
        self._lut_service = lut_service
        self._color_reference_service = color_reference_service
        
        self.cmy_to_log_xyz = self._return_callable_cmy_to_log_xyz()
        
        # communicate to the color reference service the callable to convert cmy densities to log xyz
        self._color_reference_service.cmy_to_log_xyz = self.cmy_to_log_xyz
        
    # public methods

    def scan(self, density_channels: np.ndarray) -> np.ndarray:
        rgb = self._density_to_rgb(density_channels, use_lut=self._settings.use_scanner_lut)
        rgb = self._apply_blur_and_unsharp(rgb)
        return self._apply_cctf_encoding(rgb)

    # private methods

    def _density_to_rgb(self, density_channels: np.ndarray, *, use_lut: bool) -> np.ndarray:
        if self._io.scan_film:
            glare = None
            density_min = -np.array(self._film_render.grain.density_min)
            density_max = np.nanmax(self._film.data.density_curves, axis=0)
            scan_illuminant = standard_illuminant(self._film.info.viewing_illuminant)
        else:
            glare = self._print_render.glare
            density_min = np.nanmin(self._print.data.density_curves, axis=0)
            density_max = np.nanmax(self._print.data.density_curves, axis=0)
            scan_illuminant = standard_illuminant(self._print.info.viewing_illuminant)
            
        normalization = np.sum(scan_illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)

        log_xyz = self._lut_service.spectral_compute_scanner(
            density_channels,
            spectral_calculation=self.cmy_to_log_xyz,
            data_min=density_min,
            data_max=density_max,
            use_lut=use_lut,
        )
        xyz = 10 ** log_xyz
        xyz = self._color_reference_service.black_white_xyz_correction(xyz)
        illuminant_xyz = contract("k,kl->l", scan_illuminant, STANDARD_OBSERVER_CMFS[:]) / normalization
        illuminant_xy = colour.XYZ_to_xy(illuminant_xyz)
        xyz = add_glare(xyz, illuminant_xyz, glare)
        from spektrafilm.cuda.device import is_cupy_array
        if is_cupy_array(xyz):
            from spektrafilm.cuda.color import get_xyz_to_rgb_matrix, cuda_apply_matrix
            from spektrafilm.cuda.gamut_compression import cuda_compress_rgb
            m = get_xyz_to_rgb_matrix(self._io.output_color_space, tuple(illuminant_xy), cat="CAT02")
            rgb = cuda_apply_matrix(xyz, m)
            is_preview = getattr(self._settings, "preview_mode", False)
            rgb = cuda_compress_rgb(
                rgb, self._io.output_gamut_compress,
                output_color_space=self._io.output_color_space,
                preview_mode=is_preview,
            )
        else:
            rgb = colour.XYZ_to_RGB(
                xyz,
                colourspace=self._io.output_color_space,
                apply_cctf_encoding=False,
                illuminant=illuminant_xy,
            )
            # Output gamut compression. Compresses chromaticities the
            # simulation reached that fall outside the output primaries
            # cube; for perceptual algorithms (oklch / oklrab / jzazbz /
            # cam16ucs) the spec's lightness_compression also pulls
            # super-bright pixels back into the cube via a one-sided soft
            # roll-off on the perceptual lightness axis (black stays at 0).
            # With both in place the output is in [0, 1] without a
            # downstream clip; see n100 / n110 for the design and b40 for
            # the smoothness analysis.
            rgb = compress_rgb(
                rgb, self._io.output_gamut_compress,
                output_color_space=self._io.output_color_space,
            )
        return rgb

    def _return_callable_cmy_to_log_xyz(self):
        if self._io.scan_film:
            channel_density = self._film.data.channel_density
            base_density = self._film.data.base_density
            scan_illuminant = standard_illuminant(self._film.info.viewing_illuminant)
        else:
            channel_density = self._print.data.channel_density
            base_density = self._print.data.base_density
            scan_illuminant = standard_illuminant(self._print.info.viewing_illuminant)
            
        normalization = np.sum(scan_illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)

        def cmy_to_log_xyz(density_cmy: np.ndarray) -> np.ndarray:
            from spektrafilm.cuda.device import is_cupy_array, get_array_module
            if is_cupy_array(density_cmy):
                from spektrafilm.cuda.spectral import cuda_fused_spectral_integrate
                xyz = cuda_fused_spectral_integrate(
                    density_cmy,
                    channel_density,
                    base_density=base_density,
                    illuminant=scan_illuminant,
                    response_curves=STANDARD_OBSERVER_CMFS[:],
                    normalization=normalization,
                )
            else:
                H = density_cmy.shape[0] if density_cmy.ndim == 3 else 1
                if density_cmy.ndim == 3 and H > 256:
                    chunk_size = 256
                    target_dtype = density_cmy.dtype
                    xyz = np.empty((*density_cmy.shape[:2], 3), dtype=target_dtype)
                    cd = np.asarray(channel_density, dtype=target_dtype)
                    bd = np.asarray(base_density, dtype=target_dtype) if base_density is not None else None
                    s_ill = np.asarray(scan_illuminant, dtype=target_dtype)
                    cmfs = np.asarray(STANDARD_OBSERVER_CMFS[:], dtype=target_dtype)
                    norm = np.asarray(normalization, dtype=target_dtype)
                    for y_start in range(0, H, chunk_size):
                        y_end = min(y_start + chunk_size, H)
                        chunk_cmy = density_cmy[y_start:y_end]
                        d_spec = compute_density_spectral(cd, chunk_cmy, bd)
                        chunk_light = density_to_light(d_spec, s_ill)
                        xyz[y_start:y_end] = contract("ijk,kl->ijl", chunk_light, cmfs) / norm
                        del d_spec, chunk_light
                else:
                    density_spectral = compute_density_spectral(
                        channel_density,
                        density_cmy,
                        base_density,
                    )
                    light = density_to_light(density_spectral, scan_illuminant)
                    xyz = contract("ijk,kl->ijl", light, STANDARD_OBSERVER_CMFS[:]) / normalization
            xp = get_array_module(xyz)
            return xp.log10(xp.fmax(xyz, 0.0) + 1e-10)
        return cmy_to_log_xyz

    def _apply_blur_and_unsharp(self, rgb: np.ndarray) -> np.ndarray:
        from spektrafilm.cuda.device import is_cupy_array
        if is_cupy_array(rgb):
            from spektrafilm.cuda.filters import cuda_gaussian_filter
            if self._scanner.lens_blur > 0:
                rgb = cuda_gaussian_filter(rgb, self._scanner.lens_blur)
            sigma, amount = self._scanner.unsharp_mask
            if sigma > 0 and amount > 0:
                blurred = cuda_gaussian_filter(rgb, sigma)
                rgb = rgb + amount * (rgb - blurred)
            return rgb

        rgb = apply_gaussian_blur(rgb, self._scanner.lens_blur)
        sigma, amount = self._scanner.unsharp_mask
        if sigma > 0 and amount > 0:
            rgb = apply_unsharp_mask(rgb, sigma=sigma, amount=amount)
        return rgb

    def _apply_cctf_encoding(self, rgb: np.ndarray) -> np.ndarray:
        if self._io.output_cctf_encoding:
            from spektrafilm.cuda.device import is_cupy_array
            if is_cupy_array(rgb):
                from spektrafilm.cuda.color import cuda_cctf_encoding
                return cuda_cctf_encoding(rgb, self._io.output_color_space)
            return colour.RGB_to_RGB(
                rgb,
                self._io.output_color_space,
                self._io.output_color_space,
                apply_cctf_decoding=False,
                apply_cctf_encoding=True,
            )
        return rgb



