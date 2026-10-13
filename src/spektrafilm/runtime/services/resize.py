from __future__ import annotations

import numpy as np
from skimage.transform import rescale

from spektrafilm.utils.crop_resize import crop_image


class ResizingService:
    """Shared resize/crop operations used by runtime stages."""

    def __init__(self, io_params, film_format_mm: float):
        self._io = io_params
        self.film_format_mm = film_format_mm
        self.pixel_size_um = None
        self.origin_x_um: float = 0.0
        self.origin_y_um: float = 0.0

    def crop_and_rescale(self, image: np.ndarray) -> np.ndarray:
        self.pixel_size_um = self.film_format_mm * 1000 / np.max(image.shape[0:2])
        self.origin_x_um = 0.0
        self.origin_y_um = 0.0

        if self._io.crop:
            center = np.flip(self._io.crop_center)
            shape = image.shape[0:2]
            cn = np.round(shape * np.array(center))
            sz = np.round(np.double(np.max(shape)) * np.flip(np.array(self._io.crop_size)))
            x0 = np.round(cn - sz / 2).astype(np.int64)
            sz = sz.astype(np.int64)
            x0[x0 < 0] = 0
            if x0[0] + sz[0] > shape[0]:
                x0[0] = shape[0] - sz[0]
            if x0[1] + sz[1] > shape[1]:
                x0[1] = shape[1] - sz[1]

            self.origin_x_um = float(x0[1] * self.pixel_size_um)
            self.origin_y_um = float(x0[0] * self.pixel_size_um)
            image = crop_image(image, center=self._io.crop_center, size=self._io.crop_size)            

        if self._io.upscale_factor != 1.0:
            self.pixel_size_um /= self._io.upscale_factor
            image = rescale(
                image,
                self._io.upscale_factor,
                channel_axis=2,
                order=3,
            )
        return image
    
    def small_preview(self, image: np.ndarray,
                      max_size: int = 256) -> np.ndarray:
        max_dim = max(image.shape[0:2])
        if max_dim > max_size:
            step = int(np.ceil(max_dim / max_size))
            return image[::step, ::step]
        return image
    