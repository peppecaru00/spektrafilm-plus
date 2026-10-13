from __future__ import annotations

import numpy as np
from opt_einsum import contract

from spektrafilm.model.diffusion import apply_diffusion_filter_um
from spektrafilm.model.develop import compute_density_spectral, develop_print_morph
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.utils.conversions import density_to_light


class PrintingStage:
    def __init__(
        self,
        film,
        film_render_params,
        print_profile,
        print_render_params,
        enlarger_params,
        settings_params,
        lut_service,
        enlarger_service,
        resize_service,
        color_reference_service,
    ):
        self._film = film
        self._film_render = film_render_params
        self._print = print_profile
        self._print_render = print_render_params
        self._enlarger = enlarger_params
        self._settings = settings_params
        self._lut_service = lut_service
        self._enlarger_service = enlarger_service
        self._resize_service = resize_service
        self._color_reference_service = color_reference_service

    # public methods

    def expose(self, cmy_film_density: np.ndarray) -> np.ndarray:
        
        cmy_film_black = np.zeros((1,1,3)) - np.array(self._film_render.grain.density_min)
        cmy_film_white = np.nanmax(self._film.data.density_curves, axis=0)[None, None, :]
        self._color_reference_service.log_raw_print_black = self._film_cmy_to_print_log_raw(cmy_film_black)
        self._color_reference_service.log_raw_print_white = self._film_cmy_to_print_log_raw(cmy_film_white)
        
        log_raw_print = self._lut_service.spectral_compute_enlarger(
            cmy_film_density,
            spectral_calculation=self._film_cmy_to_print_log_raw,
            data_min=-np.array(self._film_render.grain.density_min),
            data_max=np.nanmax(self._film.data.density_curves, axis=0),
            use_lut=self._settings.use_enlarger_lut,
        )    
        raw = 10**log_raw_print
        raw *= self._enlarger.print_exposure
        from spektrafilm.cuda.device import is_cupy_array, to_device
        bw_corr = self._color_reference_service.black_white_printing_exposure_correction()
        if is_cupy_array(raw):
            bw_corr = to_device(bw_corr)
        raw *= bw_corr
        raw = apply_diffusion_filter_um(
            raw,
            self._enlarger.diffusion_filter,
            pixel_size_um=self._resize_service.pixel_size_um,
        )
        from spektrafilm.cuda.device import get_array_module
        xp = get_array_module(raw)
        return xp.log10(xp.fmax(raw, 0.0) + 1e-10)

    def develop(self, log_raw: np.ndarray) -> np.ndarray:
                
        return develop_print_morph(
            log_raw,
            self._print.data.log_exposure,
            self._print.data.density_curves_model,
            density_curves_morph=self._print_render.density_curves_morph,
            profile_type=self._print.info.type,
        )

    # private methods

    def _film_cmy_to_print_log_raw(self, cmy_film_density: np.ndarray) -> np.ndarray:
        sensitivity = 10 ** self._print.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)
        enlarger_light_source = standard_illuminant(self._enlarger.illuminant)
        print_illuminant = self._enlarger_service.enlarger_filtered_illuminant(enlarger_light_source)

        from spektrafilm.cuda.device import is_cupy_array, to_device, get_array_module
        if is_cupy_array(cmy_film_density):
            from spektrafilm.cuda.spectral import cuda_fused_spectral_integrate
            raw = cuda_fused_spectral_integrate(
                cmy_film_density,
                self._film.data.channel_density,
                base_density=self._film.data.base_density,
                illuminant=print_illuminant,
                response_curves=sensitivity,
                normalization=None,
            )
        else:
            H = cmy_film_density.shape[0] if cmy_film_density.ndim == 3 else 1
            if cmy_film_density.ndim == 3 and H > 256:
                chunk_size = 256
                target_dtype = cmy_film_density.dtype
                raw = np.empty((*cmy_film_density.shape[:2], 3), dtype=target_dtype)
                cd = np.asarray(self._film.data.channel_density, dtype=target_dtype)
                bd = np.asarray(self._film.data.base_density, dtype=target_dtype) if self._film.data.base_density is not None else None
                p_ill = np.asarray(print_illuminant, dtype=target_dtype)
                sens = np.asarray(sensitivity, dtype=target_dtype)
                for y_start in range(0, H, chunk_size):
                    y_end = min(y_start + chunk_size, H)
                    chunk_cmy = cmy_film_density[y_start:y_end]
                    d_spec = compute_density_spectral(cd, chunk_cmy, base_density=bd)
                    chunk_light = density_to_light(d_spec, p_ill)
                    raw[y_start:y_end] = contract("ijk, kl->ijl", chunk_light, sens)
                    del d_spec, chunk_light
            else:
                density_spectral = compute_density_spectral(
                    self._film.data.channel_density,
                    cmy_film_density,
                    base_density=self._film.data.base_density,
                )
                light = density_to_light(density_spectral, print_illuminant)
                raw = contract("ijk, kl->ijl", light, sensitivity)
        factor = self._compute_exposure_factor_midgray(sensitivity, print_illuminant)
        if is_cupy_array(raw):
            factor = to_device(factor)
        raw *= factor
        preflash = self._compute_raw_preflash(enlarger_light_source, sensitivity)
        if is_cupy_array(raw) and not is_cupy_array(preflash):
            preflash = to_device(preflash)
        raw += preflash
        xp = get_array_module(raw)
        return xp.log10(xp.fmax(raw, 0.0) + 1e-10)

    def _compute_raw_preflash(self, light_source, sensitivity):
        if self._enlarger.preflash_exposure > 0:
            preflash_illuminant = self._enlarger_service.preflash_filtered_illuminant(light_source)
            density_base = np.asarray(self._film.data.base_density)[None, None, :]
            light_preflash = density_to_light(density_base, preflash_illuminant)
            raw_preflash = contract("ijk, kl->ijl", light_preflash, sensitivity)
            return raw_preflash * self._enlarger.preflash_exposure
        return np.zeros((3,))

    def _compute_exposure_factor_midgray(self, sensitivity, print_illuminant):
        factor_midgray = _exposure_factor(sensitivity, print_illuminant, self._enlarger_service.density_spectral_midgray)
        if self._enlarger_service.density_spectral_midgray_comp is not None:
            factor_midgray_comp = _exposure_factor(sensitivity, print_illuminant,
                                                    self._enlarger_service.density_spectral_midgray_comp)
        else:
            factor_midgray_comp = 1.0
        if self._enlarger.print_exposure_compensation and not self._enlarger.normalize_print_exposure:
            return factor_midgray_comp / factor_midgray
        elif self._enlarger.normalize_print_exposure and self._enlarger.print_exposure_compensation:
            return factor_midgray_comp
        elif self._enlarger.normalize_print_exposure and not self._enlarger.print_exposure_compensation:
            return factor_midgray
        else:
            return 1.0

def _exposure_factor(sensitivity, print_illuminant, density_spectral_midgray):
    light_midgray = density_to_light(density_spectral_midgray, print_illuminant)
    raw_midgray = contract("ijk, kl->ijl", light_midgray, sensitivity)
    raw_midgray = np.fmax(raw_midgray, 1e-10)
    # use the geometric mean to normalize the exposure
    raw_midgray_geomean = np.exp(np.mean(np.log(raw_midgray), axis=2, keepdims=True))
    return 1 / raw_midgray_geomean