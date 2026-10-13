from __future__ import annotations

import numpy as np
from typing import Any, Callable
from colour import RGB_to_RGB

from spektrafilm.model.develop import develop_simple

class ColorReferenceService:
    """Manages the color reference for black and white corrections."""
    def __init__(self, film_profile,  film_render,
                       print_profile, print_render,
                       black_correction, white_correction,
                       black_level, white_level,
                       io_params):
        self._film = film_profile
        self._film_render = film_render
        self._print = print_profile
        self._print_render = print_render
        self._scan_film = io_params.scan_film
        self._output_color_space = io_params.output_color_space
        self._output_cctf_encoding = io_params.output_cctf_encoding

        self._black_correction = black_correction
        self._white_correction = white_correction
        self._black_level = _remove_sRGB_cctf(black_level)
        self._white_level = _remove_sRGB_cctf(white_level)
        
        # local memory for black and white reference densities, to avoid redundant calculations during correction
        self._y_black = None # positive film or print cmy density for black
        self._y_white = None # positive film or print cmy density for white
        self._black_white_exposure_correction = None # exposure correction factor to apply to the print exposure based on the black and white reference densities
        
        # communication with the scanning stage
        self.cmy_to_log_xyz: Any | None = None # callable to convert cmy densities to log xyz, to be defined in the scanning stage during pipeline init!
        self.log_raw_print_black = None
        self.log_raw_print_white = None

    # public methods

    def _update_cmy_black_white_references(self, in_print=False):
        if not self._black_correction and not self._white_correction:
            return
        elif self._scan_film and self._film.info.type == 'negative':
            return
        else:
            if self.cmy_to_log_xyz is None:
                raise RuntimeError("cmy_to_log_xyz must be configured before black/white correction is used.")
            cmy_to_log_xyz: Callable[[np.ndarray], np.ndarray] = self.cmy_to_log_xyz
            from spektrafilm.cuda.device import to_cpu
            if self._scan_film and self._film.info.type == 'positive' and not in_print:
                cmy_black = np.nanmax(self._film.data.density_curves, axis=0)[None, None, :]
                cmy_white = np.zeros((1, 1, 3))
                log_xyz_black = cmy_to_log_xyz.__call__(cmy_black)
                log_xyz_white = cmy_to_log_xyz.__call__(cmy_white)
                self._y_black = float(np.asarray(to_cpu(10 ** log_xyz_black))[..., 1].ravel()[0])
                self._y_white = float(np.asarray(to_cpu(10 ** log_xyz_white))[..., 1].ravel()[0])
            elif not self._scan_film and self._print.info.type == 'negative' and in_print:
                cmy_black = develop_simple(
                    self.log_raw_print_black,
                    self._print.data.log_exposure,
                    self._print.data.density_curves,
                    gamma_factor=1.0,
                )
                cmy_white = develop_simple(
                    self.log_raw_print_white,
                    self._print.data.log_exposure,
                    self._print.data.density_curves,
                    gamma_factor=1.0,
                )
                log_xyz_black = cmy_to_log_xyz.__call__(cmy_black)
                log_xyz_white = cmy_to_log_xyz.__call__(cmy_white)
                self._y_black = float(np.asarray(to_cpu(10 ** log_xyz_black))[..., 1].ravel()[0])
                self._y_white = float(np.asarray(to_cpu(10 ** log_xyz_white))[..., 1].ravel()[0])
            return
    
    def black_white_filming_exposure_correction(self): # in filming and printing
        if not self._black_correction and not self._white_correction:
            return 1.0
        elif self._film.info.type == 'negative':
            return 1.0
        elif self._scan_film and self._film.info.type == 'positive':
            density_midgray = -np.log10(0.184)
            self._update_cmy_black_white_references(in_print=False)
            midgray_corrected = float(self._correction_fucntion()[1])
            density_midgray_corrected = float(-np.log10(midgray_corrected))
            density_curve_av = np.nanmean(self._film.data.density_curves, axis=1)
            density_min_av = float(np.nanmean(self._film.data.base_density))
            log_exposure = self._film.data.log_exposure
            log_exposure_midgray_corrected = float(-np.interp(-(density_midgray_corrected-density_min_av), 
                                                -density_curve_av, log_exposure))
            log_exposure_midgray = float(-np.interp(-(density_midgray-density_min_av),
                                                -density_curve_av, log_exposure))
            exposure_correction = float(10 ** (log_exposure_midgray_corrected - log_exposure_midgray))
            return float(1.0 / exposure_correction)
        else:
            return 1.0

    def black_white_printing_exposure_correction(self): # in printing
        if not self._black_correction and not self._white_correction:
            return 1.0
        elif self._print.info.type == 'negative':
            density_midgray = -np.log10(0.184)
            self._update_cmy_black_white_references(in_print=True)
            midgray_corrected = float(self._correction_fucntion()[1])
            density_midgray_corrected = float(-np.log10(midgray_corrected))
            density_curve_av = np.nanmean(self._print.data.density_curves, axis=1)
            density_min_av = float(np.nanmean(self._print.data.base_density))
            log_exposure = self._print.data.log_exposure
            log_exposure_midgray_corrected = float(np.interp(density_midgray_corrected-density_min_av, 
                                                density_curve_av, log_exposure))
            log_exposure_midgray = float(np.interp(density_midgray-density_min_av,
                                                density_curve_av, log_exposure))
            exposure_correction = float(10 ** (log_exposure_midgray_corrected - log_exposure_midgray))
            return float(exposure_correction)

    def black_white_xyz_correction(self, xyz): # in scanning
        """Apply black and white correction to the XYZ values,
        based on the black and white reference densities
        cmy_to_log_xyz() must be defined in the scanning stage and passed to convert cmy densities
        """
        if not self._black_correction and not self._white_correction:
            return xyz
        if self._scan_film and self._film.info.type == 'negative':
            return xyz # do not correct negative film scans
        else:
            if self._y_black is None or self._y_white is None:
                self._update_cmy_black_white_references(in_print=not self._scan_film)
            correction_func, _ = self._correction_fucntion()
            y = xyz[:, :, 1]
            y_corrected = correction_func(y)
            scale = y_corrected / (y + 1e-10)
            return xyz * scale[:, :, None]


    def _correction_fucntion(self):
        white_level = float(self._white_level)
        black_level = float(self._black_level)
        m = 1.0
        q = 0.0
        correction_func = lambda y: y
        if self._black_correction and not self._white_correction:
            white_level = float(self._y_white) if self._y_white is not None else white_level
        if self._white_correction and not self._black_correction:
            black_level = float(self._y_black) if self._y_black is not None else black_level
        if self._black_correction or self._white_correction:
            y_w = float(self._y_white) if self._y_white is not None else white_level
            y_b = float(self._y_black) if self._y_black is not None else black_level
            m = float((white_level - black_level) / (y_w - y_b + 1e-10))
            q = float(black_level - m * y_b)
            def clipped_linear_correction(y):
                from spektrafilm.cuda.device import is_cupy_array
                if is_cupy_array(y):
                    import cupy as cp
                    return cp.clip(m * y + q, 0.0, 1.0)
                return np.clip(m * y + q, 0.0, 1.0)
            correction_func = clipped_linear_correction
        midgray_black_white_corrected = float((0.184 - q) / m)
        return correction_func, midgray_black_white_corrected

    _correction_function = _correction_fucntion

# private functions
    
def _remove_sRGB_cctf(y_input):
    return float(RGB_to_RGB(y_input*np.ones((1,1,3)),
                    'sRGB',
                    'sRGB',
                    apply_cctf_decoding=True,
                    apply_cctf_encoding=False,
                ).mean())
    
