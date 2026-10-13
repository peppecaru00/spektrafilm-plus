from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from spektrafilm.profiles.io import Hanatos2025SensitivityAdaptation
from spektrafilm.utils.gamut_compression import InputGamutCompressSpec
from spektrafilm.utils.lut import compute_with_lut
from spektrafilm.utils.spectral_upsampling import compute_hanatos2025_tc_lut, compute_tc_lut, lut_descriptor
from spektrafilm.utils.timings import timeit


def _key_equal(a, b) -> bool:
    """Structural equality for cache keys: tuples that may hold ndarrays, nested
    tuples, strings, frozen specs, or None."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_key_equal(x, y) for x, y in zip(a, b))
    return a == b


def _key_snapshot(part):
    """Deep-copy the mutable (ndarray) parts of a key so a later in-place
    mutation of a caller's array can't silently match a stale cache."""
    if isinstance(part, np.ndarray):
        return np.array(part, copy=True)
    if isinstance(part, tuple):
        return tuple(_key_snapshot(p) for p in part)
    return part


@dataclass
class _MemoLUT:
    lut: np.ndarray | None = None
    _key: tuple | None = None

    def invalidate(self) -> None:
        self.lut = None
        self._key = None

    def get(self, key: tuple, compute: Callable[[], np.ndarray]) -> np.ndarray:
        if self.lut is not None and self._key is not None and _key_equal(self._key, key):
            return self.lut
        self.lut = compute()
        self._key = _key_snapshot(key)
        return self.lut


def _adaptation_key(adaptation: Hanatos2025SensitivityAdaptation | None) -> tuple:
    if adaptation is None:
        return ()
    return (
        bool(adaptation.apply_window),
        bool(adaptation.apply_surface),
        float(adaptation.spectral_gaussian_blur),
        adaptation.reference_illuminant,
        np.asarray(adaptation.window_params) if adaptation.window_params is not None else None,
        np.asarray(adaptation.surface_params) if adaptation.surface_params is not None else None,
    )


class SpectralLUTService:
    def __init__(self, lut_resolution: int):
        self._lut_resolution = lut_resolution

        self.timings = {}
        self.hanatos2025_adaptation = None # to be set by filming stage with info from film profile and settings
        # Input gamut compression spec (set by the pipeline from
        # params.io.input_gamut_compress). Drives the build-time bake
        # of the filming tc_lut; changes invalidate the cache below.
        self.input_gamut_compress: InputGamutCompressSpec = InputGamutCompressSpec()

        # One memo per rgb_to_raw method (lazily created, keyed by identifier)
        self._tc_lut_memos: dict[str, _MemoLUT] = {}

        # external memory
        self.filming_tc_lut_memory : np.ndarray | None = None # tc_lut memory
        self.enlarger_lut_memory : np.ndarray | None = None # enlarger lut memory
        self.scanner_lut_memory : np.ndarray | None = None # scanner lut memory

        # local memory
        self._film_sensitivity = None # to track if tc_lut needs to be recomputed when film sensitivity changes
        self._cached_filming_adaptation = None # full adaptation state for which the cached tc_lut was computed
        self._cached_input_gamut_compress: InputGamutCompressSpec | None = None
        self._enlarger_test_results_memory = None # to test if enlarger LUTs are identical for same input
        self._scanner_test_results_memory = None # to test if scanner LUTs are identical for same input
        
        self._cmy_test_values = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                                          [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]) # to test if LUTs are identical

    def set_hanatos2025_adaptation(self, adaptation: Hanatos2025SensitivityAdaptation) -> None:
        adaptation_copy = self._copy_hanatos2025_adaptation(adaptation)
        self.hanatos2025_adaptation = adaptation_copy
        if not self._same_hanatos2025_adaptation(self._cached_filming_adaptation, adaptation_copy):
            self.filming_tc_lut_memory = None
            self._film_sensitivity = None
            self._cached_filming_adaptation = None
            if 'hanatos2025' in self._tc_lut_memos:
                self._tc_lut_memos['hanatos2025'].invalidate()

    def set_input_gamut_compress(self, spec: InputGamutCompressSpec) -> None:
        """Set the input-gamut-compression spec used when baking the
        filming tc_lut. A change invalidates the cached LUT.
        """
        if spec != self.input_gamut_compress:
            self.input_gamut_compress = spec
            # Force a rebuild of the filming tc_lut on the next call.
            self.filming_tc_lut_memory = None
            self._film_sensitivity = None
            self._cached_filming_adaptation = None
            self._cached_input_gamut_compress = None
            self._tc_lut_memos.clear()

    @staticmethod
    def _copy_hanatos2025_adaptation(
        adaptation: Hanatos2025SensitivityAdaptation | None,
    ) -> Hanatos2025SensitivityAdaptation | None:
        if adaptation is None:
            return None
        return Hanatos2025SensitivityAdaptation(
            window_params=np.array(adaptation.window_params, copy=True),
            surface_params=np.array(adaptation.surface_params, copy=True),
            spectral_gaussian_blur=float(adaptation.spectral_gaussian_blur),
            reference_illuminant=adaptation.reference_illuminant,
            apply_window=bool(adaptation.apply_window),
            apply_surface=bool(adaptation.apply_surface),
            active=adaptation.active,
        )

    @staticmethod
    def _same_hanatos2025_adaptation(
        left: Hanatos2025SensitivityAdaptation | None,
        right: Hanatos2025SensitivityAdaptation | None,
    ) -> bool:
        if left is None or right is None:
            return left is right
        return (
            bool(left.apply_window) == bool(right.apply_window)
            and bool(left.apply_surface) == bool(right.apply_surface)
            and float(left.spectral_gaussian_blur) == float(right.spectral_gaussian_blur)
            and left.reference_illuminant == right.reference_illuminant
            and np.array_equal(left.window_params, right.window_params)
            and np.array_equal(left.surface_params, right.surface_params)
        )

    @timeit("spectral_compute_enlarger")
    def spectral_compute_enlarger(self,
        cmy_data,
        spectral_calculation: Callable,
        data_min,
        data_max,
        *,
        use_lut: bool = False,
    ):
        if not use_lut:
            return spectral_calculation(cmy_data)

        test_results = spectral_calculation(np.array(self._cmy_test_values))

        if (
            self.enlarger_lut_memory is not None
            and self._enlarger_test_results_memory is not None
            and np.array_equal(test_results, self._enlarger_test_results_memory)
        ):
            data_out, _ = compute_with_lut(cmy_data,
                                           spectral_calculation,
                                           xmin=data_min,
                                           xmax=data_max,
                                           steps=self._lut_resolution,
                                           lut=self.enlarger_lut_memory)
        else:
            data_out, lut = compute_with_lut(cmy_data,
                                             spectral_calculation,
                                             xmin=data_min,
                                             xmax=data_max,
                                             steps=self._lut_resolution)
            self.enlarger_lut_memory = lut
            self._enlarger_test_results_memory = np.array(test_results, copy=True)

        if data_out is None:
            raise RuntimeError('LUT computation did not produce an output')
        return data_out

    @timeit("spectral_compute_scanner")
    def spectral_compute_scanner(self,
        cmy_data,
        spectral_calculation: Callable,
        data_min,
        data_max,
        *,
        use_lut: bool = False,
    ):
        if not use_lut:
            return spectral_calculation(cmy_data)

        test_results = spectral_calculation(np.array(self._cmy_test_values))

        if (
            self.scanner_lut_memory is not None
            and self._scanner_test_results_memory is not None
            and np.array_equal(test_results, self._scanner_test_results_memory)
        ):
            data_out, _ = compute_with_lut(cmy_data,
                                           spectral_calculation,
                                           xmin=data_min,
                                           xmax=data_max,
                                           steps=self._lut_resolution,
                                           lut=self.scanner_lut_memory)
        else:
            data_out, lut = compute_with_lut(cmy_data,
                                             spectral_calculation,
                                             xmin=data_min,
                                             xmax=data_max,
                                             steps=self._lut_resolution)
            self.scanner_lut_memory = lut
            self._scanner_test_results_memory = np.array(test_results, copy=True)

        if data_out is None:
            raise RuntimeError('LUT computation did not produce an output')
        return data_out

    def _tc_lut_memo(self, method: str) -> _MemoLUT:
        memo = self._tc_lut_memos.get(method)
        if memo is None:
            memo = self._tc_lut_memos[method] = _MemoLUT()
        return memo

    @timeit("get_filming_tc_lut")
    def get_filming_tc_lut(self, method_or_sensitivity, sensitivity=None, reference_illuminant=None):
        """tc_lut for any rgb_to_raw `method`, dispatched on its descriptor kind and
        memoized per method. REFLECTANCE methods relight under the film reference
        illuminant; IRRADIANCE methods integrate directly."""
        if isinstance(method_or_sensitivity, str):
            method = method_or_sensitivity
            sens = np.asarray(sensitivity)
            ref_illu = reference_illuminant or 'D55'
        else:
            method = "hanatos2025"
            sens = np.asarray(method_or_sensitivity)
            ref_illu = getattr(self.hanatos2025_adaptation, 'reference_illuminant', 'D55') if self.hanatos2025_adaptation else 'D55'

        descriptor = lut_descriptor(method)
        if descriptor.get('kind') == 'irradiance':
            adaptation = self.hanatos2025_adaptation
            key = (method, sens, _adaptation_key(adaptation), self.input_gamut_compress)
            compute = lambda: compute_hanatos2025_tc_lut(
                sens, adaptation, gamut_compress=self.input_gamut_compress)
        else:
            scene_illuminant = descriptor.get('reflectance', {}).get('scene_illuminant', 'D65')
            key = (method, sens, ref_illu, scene_illuminant, self.input_gamut_compress)
            compute = lambda: compute_tc_lut(
                method, sens, ref_illu, gamut_compress=self.input_gamut_compress)

        result = self._tc_lut_memo(method).get(key, compute)
        self.filming_tc_lut_memory = result
        self._film_sensitivity = sens
        self._cached_filming_adaptation = self._copy_hanatos2025_adaptation(self.hanatos2025_adaptation)
        self._cached_input_gamut_compress = self.input_gamut_compress
        return result

