from types import SimpleNamespace

import numpy as np
import pytest

from spektrafilm.utils import raw_file_processor


def _stub_raw_reader(monkeypatch, raw_image: np.ndarray) -> None:
    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def postprocess(self, **_kwargs):
            return raw_image

    monkeypatch.setattr(
        raw_file_processor,
        'rawpy',
        SimpleNamespace(
            imread=lambda path: Reader(),
            ColorSpace=SimpleNamespace(ACES='ACES'),
        ),
    )


def _stub_exif(monkeypatch, **overrides) -> None:
    values = {
        'make': 'Canon',
        'model': 'Canon EOS 5D Mark IV',
        'lens_make': 'Canon',
        'lens_model': 'Canon EF 50mm f/1.8 STM',
        'focal_length': 50.0,
        'f_number': 2.8,
    }
    values.update(overrides)
    exif_metadata = raw_file_processor.ExifData(**values)
    monkeypatch.setattr(raw_file_processor, '_read_exif_metadata', lambda path: exif_metadata)


def test_process_raw_file_daylight_uses_linear_output_and_colour_conversion(monkeypatch):
    raw_image = np.full((1, 1, 3), 16384, dtype=np.uint16)
    captured = {}

    class Reader:
        daylight_whitebalance = [2.0, 1.0, 1.5, 1.0]
        rgb_xyz_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def postprocess(self, **kwargs):
            captured['postprocess'] = kwargs
            return raw_image

    def imread(path):
        captured['path'] = path
        return Reader()

    def fake_rgb_to_rgb(image, **kwargs):
        captured['image'] = image
        captured['colour'] = kwargs
        return image + 0.25

    monkeypatch.setattr(
        raw_file_processor,
        'rawpy',
        SimpleNamespace(
            imread=imread,
            ColorSpace=SimpleNamespace(ACES='ACES'),
        ),
    )
    monkeypatch.setattr(raw_file_processor.colour, 'RGB_to_RGB', fake_rgb_to_rgb)

    result = raw_file_processor.load_and_process_raw_file(
        'example.nef',
        white_balance='daylight',
        output_colorspace='sRGB',
        output_cctf_encoding=False,
    )

    expected_linear = raw_image.astype(np.float32) / 65535.0
    np.testing.assert_allclose(captured['image'], expected_linear)
    np.testing.assert_allclose(result, expected_linear + 0.25)
    assert captured['path'] == 'example.nef'
    assert captured['postprocess']['output_color'] == 'ACES'
    assert captured['postprocess']['output_bps'] == 16
    assert captured['postprocess']['no_auto_bright'] is True
    assert captured['postprocess']['gamma'] == (1, 1)
    assert 'user_wb' not in captured['postprocess']
    assert 'use_camera_wb' not in captured['postprocess']
    assert captured['colour']['input_colourspace'] == raw_file_processor.colour.RGB_COLOURSPACES['ACES2065-1']
    assert captured['colour']['output_colourspace'] == raw_file_processor.colour.RGB_COLOURSPACES['sRGB']
    assert captured['colour']['apply_cctf_decoding'] is False
    assert captured['colour']['apply_cctf_encoding'] is False


def test_process_raw_file_as_shot_uses_camera_white_balance(monkeypatch):
    raw_image = np.full((1, 1, 3), 32768, dtype=np.uint16)
    captured = {}

    class Reader:
        daylight_whitebalance = [2.0, 1.0, 1.5, 1.0]
        rgb_xyz_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def postprocess(self, **kwargs):
            captured['postprocess'] = kwargs
            return raw_image

    monkeypatch.setattr(
        raw_file_processor,
        'rawpy',
        SimpleNamespace(
            imread=lambda path: Reader(),
            ColorSpace=SimpleNamespace(ACES='ACES'),
        ),
    )

    result = raw_file_processor.load_and_process_raw_file('example.nef', white_balance='as_shot')

    np.testing.assert_allclose(result, raw_image.astype(np.float32) / 65535.0)
    assert captured['postprocess']['use_camera_wb'] is True
    assert 'user_wb' not in captured['postprocess']


@pytest.mark.parametrize(
    (
        'white_balance',
        'temperature',
        'tint',
        'expected_result_scale',
        'expected_source_white',
    ),
    [
        ('tungsten', None, None, np.array([1.0, 1.0, 1.0], dtype=np.float32), np.array([2.85, 1.0, 1.425])),
        ('custom', 3200.0, 0.85, np.array([1.0, 0.85, 1.0], dtype=np.float32), np.array([3.2, 1.0, 1.6])),
        ('custom', 6504.0, 1.0, np.array([1.0, 1.0, 1.0], dtype=np.float32), None),
    ],
    ids=['tungsten', 'custom-tungsten', 'custom-daylight'],
)
def test_process_raw_file_temperature_white_balance_variants(monkeypatch, white_balance, temperature, tint, expected_result_scale, expected_source_white):
    raw_image = np.full((1, 1, 3), 16384, dtype=np.uint16)
    postprocess_calls: list[dict[str, object]] = []
    adaptation_calls = []

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def postprocess(self, **kwargs):
            postprocess_calls.append(kwargs)
            return raw_image

    monkeypatch.setattr(
        raw_file_processor,
        'rawpy',
        SimpleNamespace(
            imread=lambda path: Reader(),
            ColorSpace=SimpleNamespace(ACES='ACES'),
        ),
    )
    monkeypatch.setattr(raw_file_processor.colour, 'CCT_to_xy', lambda temperature, method='CIE Illuminant D Series': np.array([temperature, 0.0]))
    monkeypatch.setattr(raw_file_processor.colour, 'xy_to_XYZ', lambda xy: np.array([xy[0] / 1000.0, 1.0, xy[0] / 2000.0]))
    monkeypatch.setattr(raw_file_processor.colour, 'RGB_to_XYZ', lambda image, **kwargs: image)

    def fake_chromatic_adaptation(xyz, source_white_xyz, target_white_xyz, **kwargs):
        adaptation_calls.append(
            {
                'source': source_white_xyz,
                'target': target_white_xyz,
                'kwargs': kwargs,
            }
        )
        return xyz + np.float32(0.25)

    monkeypatch.setattr(raw_file_processor.colour, 'chromatic_adaptation', fake_chromatic_adaptation)
    monkeypatch.setattr(raw_file_processor.colour, 'XYZ_to_RGB', lambda xyz, **kwargs: xyz)

    result = raw_file_processor.load_and_process_raw_file(
        'example.nef',
        white_balance=white_balance,
        temperature=temperature,
        tint=tint,
    )

    expected_linear = raw_image.astype(np.float32) / 65535.0
    expected = expected_linear if expected_source_white is None else expected_linear + 0.25
    expected = expected * expected_result_scale
    np.testing.assert_allclose(result, expected)
    assert all('user_wb' not in call for call in postprocess_calls)
    assert all('use_camera_wb' not in call for call in postprocess_calls)
    if expected_source_white is None:
        assert adaptation_calls == []
    else:
        assert len(adaptation_calls) == 1
        np.testing.assert_allclose(adaptation_calls[0]['source'], expected_source_white)
        np.testing.assert_allclose(adaptation_calls[0]['target'], [6.504, 1.0, 3.252])
        assert adaptation_calls[0]['kwargs']['method'] == 'Von Kries'


@pytest.mark.parametrize(
    ('lens_model', 'correction_summary', 'expect_database_call', 'expected_lens_info'),
    [
        ('', None, False, {}),
        ('Canon EF 50mm f/1.8 STM', '', True, {}),
        (
            'Canon EF 50mm f/1.8 STM',
            'Canon EF 50mm f/1.8 STM @ 50.0mm f/2.8',
            True,
            {'summary': 'Canon EF 50mm f/1.8 STM @ 50.0mm f/2.8'},
        ),
    ],
    ids=['missing-lens-model', 'no-correction-summary', 'correction-summary'],
)
def test_process_raw_file_lens_info_reporting(
    monkeypatch,
    lens_model: str,
    correction_summary: str | None,
    expect_database_call: bool,
    expected_lens_info: dict[str, str],
):
    raw_image = np.full((1, 1, 3), 16384, dtype=np.uint16)
    lens_info_out: dict[str, str] = {}

    _stub_raw_reader(monkeypatch, raw_image)
    _stub_exif(monkeypatch, lens_model=lens_model)

    if not expect_database_call:
        monkeypatch.setattr(
            raw_file_processor.lensfunpy,
            'Database',
            lambda: (_ for _ in ()).throw(AssertionError('lensfun database should not be queried without a lens model')),
        )
    else:
        monkeypatch.setattr(
            raw_file_processor,
            '_apply_lens_correction',
            lambda rgb, exif_metadata: (rgb, '' if correction_summary is None else correction_summary),
        )

    result = raw_file_processor.load_and_process_raw_file(
        'example.nef',
        lens_correction=True,
        lens_info_out=lens_info_out,
    )

    np.testing.assert_allclose(result, raw_image.astype(np.float32) / 65535.0)
    assert lens_info_out == expected_lens_info


def test_clean_maker_name():
    assert raw_file_processor._clean_maker_name('Panasonic Corporation') == 'Panasonic'
    assert raw_file_processor._clean_maker_name('Sony Corporation') == 'Sony'
    assert raw_file_processor._clean_maker_name('NIKON CORPORATION') == 'Nikon'
    assert raw_file_processor._clean_maker_name('Canon Inc.') == 'Canon'
    assert raw_file_processor._clean_maker_name('LEICA CAMERA AG') == 'Leica'
    assert raw_file_processor._clean_maker_name('FUJIFILM') == 'Fujifilm'
    assert raw_file_processor._clean_maker_name('') == ''
    assert raw_file_processor._clean_maker_name(None) == ''


def test_lens_model_score_bidirectional():
    score = raw_file_processor._lens_model_score("FE 50mm f/1.8", "Sony FE 50mm F1.8")
    assert score[1] == 1
    assert score[2] >= 2
    assert score > (0, 0, 0)


def test_read_exif_metadata_with_panasonic_maker_note(monkeypatch):
    class FakeDatum:
        def __init__(self, val):
            self._val = val

        def value(self):
            return self._val

        def toFloat(self):
            return float(self._val)

    class FakeImage:
        def readMetadata(self):
            pass

        def exifData(self):
            return {
                "Exif.Image.Make": FakeDatum("Panasonic"),
                "Exif.Image.Model": FakeDatum("DC-S5"),
                "Exif.Panasonic.LensType": FakeDatum("LUMIX S 20-60/F3.5-5.6"),
                "Exif.Photo.FocalLength": FakeDatum("20.0"),
                "Exif.Photo.FNumber": FakeDatum("3.5"),
            }

        def xmpData(self):
            return {}

    fake_exiv2 = SimpleNamespace(
        ImageFactory=SimpleNamespace(open=lambda path: FakeImage()),
        make=lambda exif: None,
        model=lambda exif: None,
        lensName=lambda exif: None,
        focalLength=lambda exif: None,
        fNumber=lambda exif: None,
    )
    monkeypatch.setattr(raw_file_processor, "exiv2", fake_exiv2)
    meta = raw_file_processor._read_exif_metadata("sample.rw2")
    assert meta.make == "Panasonic"
    assert meta.model == "DC-S5"
    assert meta.lens_model == "LUMIX S 20-60/F3.5-5.6"
    assert meta.focal_length == 20.0
    assert meta.f_number == 3.5


def test_read_exif_metadata_with_xmp_tags(monkeypatch):
    class FakeDatum:
        def __init__(self, val):
            self._val = val

        def value(self):
            return self._val

        def toFloat(self):
            return float(self._val)

    class FakeImage:
        def readMetadata(self):
            pass

        def exifData(self):
            return {}

        def xmpData(self):
            return {
                "Xmp.tiff.Make": FakeDatum("Sony"),
                "Xmp.tiff.Model": FakeDatum("ILCE-7M3"),
                "Xmp.aux.Lens": FakeDatum("FE 50mm F1.8"),
                "Xmp.exif.FocalLength": FakeDatum("50.0"),
                "Xmp.exif.FNumber": FakeDatum("1.8"),
            }

    fake_exiv2 = SimpleNamespace(
        ImageFactory=SimpleNamespace(open=lambda path: FakeImage()),
        make=lambda exif: None,
        model=lambda exif: None,
        lensName=lambda exif: None,
        focalLength=lambda exif: None,
        fNumber=lambda exif: None,
    )
    monkeypatch.setattr(raw_file_processor, "exiv2", fake_exiv2)
    meta = raw_file_processor._read_exif_metadata("sample.arw")
    assert meta.make == "Sony"
    assert meta.model == "ILCE-7M3"
    assert meta.lens_model == "FE 50mm F1.8"
    assert meta.focal_length == 50.0
    assert meta.f_number == 1.8


def test_find_lens_candidates_with_corporate_maker_suffix():
    db = raw_file_processor.lensfunpy.Database()
    cams = db.find_cameras('Panasonic', 'DC-S5', loose_search=True)
    assert cams
    cam = cams[0]
    exif = raw_file_processor.ExifData(
        make='Panasonic Corporation',
        model='DC-S5',
        lens_make='Panasonic Corporation',
        lens_model='Lumix S 20-60/F3.5-5.6',
        focal_length=20.0,
        f_number=3.5,
    )
    candidates = raw_file_processor._find_lens_candidates(db, cam, exif)
    assert len(candidates) > 0
    selected = raw_file_processor._select_lens_candidate(candidates, exif)
    assert '20-60' in getattr(selected, 'model', '')


def test_apply_lens_correction_camera_corporate_make_fallback():
    rgb = np.zeros((10, 10, 3), dtype=np.float32)
    exif = raw_file_processor.ExifData(
        make='Panasonic Corporation',
        model='DC-S5',
        lens_make='Panasonic Corporation',
        lens_model='Lumix S 20-60/F3.5-5.6',
        focal_length=20.0,
        f_number=3.5,
    )
    corrected_rgb, info = raw_file_processor._apply_lens_correction(rgb, exif)
    assert '20-60' in info
    assert corrected_rgb.shape == (10, 10, 3)


def test_apply_lens_correction_unlisted_camera_compatible_mount():
    rgb = np.zeros((10, 10, 3), dtype=np.float32)
    exif = raw_file_processor.ExifData(
        make='Panasonic',
        model='DC-S9',
        lens_make='Panasonic',
        lens_model='LUMIX S 20-60/F3.5-5.6',
        focal_length=20.0,
        f_number=3.5,
    )
    corrected_rgb, info = raw_file_processor._apply_lens_correction(rgb, exif)
    assert 'Lumix S 20-60' in info
    assert '12-60' not in info
    assert corrected_rgb.shape == (10, 10, 3)


def test_parse_focal_range():
    assert raw_file_processor._parse_focal_range("28-75mm") == (28.0, 75.0)
    assert raw_file_processor._parse_focal_range("28 - 75 mm") == (28.0, 75.0)
    assert raw_file_processor._parse_focal_range("50mm") == (50.0, 50.0)
    assert raw_file_processor._parse_focal_range("50.0 mm") == (50.0, 50.0)
    assert raw_file_processor._parse_focal_range("") is None
    assert raw_file_processor._parse_focal_range("Unknown lens") is None


def test_select_lens_candidate_prefers_compatible_mount_and_maker():
    db = raw_file_processor.lensfunpy.Database()
    cams = db.find_cameras('Sony', 'ILCE-7M3', loose_search=True)
    assert cams
    cam = cams[0]

    # Test generic "50mm F1.8" matches Sony E mount, not Canon RF
    exif_sony_50 = raw_file_processor.ExifData(
        make='Sony',
        model='ILCE-7M3',
        lens_make='Sony',
        lens_model='50mm F1.8',
        focal_length=50.0,
        f_number=1.8,
    )
    candidates = raw_file_processor._find_lens_candidates(db, cam, exif_sony_50)
    all_lenses = getattr(db, 'lenses', [])
    global_candidates = [
        l for l in all_lenses
        if raw_file_processor._lens_model_score(getattr(l, 'model', ''), exif_sony_50.lens_model) > (0, 0, 0)
    ]
    best = raw_file_processor._select_lens_candidate(candidates + global_candidates, exif_sony_50, camera=cam)
    assert getattr(best, 'maker', '') == 'Sony'
    assert 'Sony E' in getattr(best, 'mounts', [])
    assert 'Canon' not in getattr(best, 'maker', '')


def test_apply_lens_correction_geometry_distortion_fallback(monkeypatch):
    rgb = np.zeros((10, 10, 3), dtype=np.float32)
    exif = raw_file_processor.ExifData(
        make='Sony',
        model='ILCE-7M3',
        lens_make='Sony',
        lens_model='FE 50mm F1.8',
        focal_length=50.0,
        f_number=1.8,
    )

    geom_coords = np.zeros((10, 10, 2), dtype=np.float32)
    called = {'color': False, 'geom': False}

    class FakeModifier:
        def __init__(self, *args, **kwargs):
            pass

        def initialize(self, *args, **kwargs):
            pass

        def apply_color_modification(self, img):
            called['color'] = True

        def apply_subpixel_geometry_distortion(self):
            # Simulate TCA data missing
            return None

        def apply_geometry_distortion(self):
            called['geom'] = True
            return geom_coords

    monkeypatch.setattr(raw_file_processor.lensfunpy, 'Modifier', FakeModifier)
    corrected_rgb, info = raw_file_processor._apply_lens_correction(rgb, exif)
    assert called['color'] is True
    assert called['geom'] is True
    assert corrected_rgb.shape == (10, 10, 3)


def test_postprocess_params_chroma_denoise(monkeypatch):
    """Verify that _postprocess_params configures LibRaw median passes and FBDD when requested."""
    import rawpy

    # Default: chroma denoise disabled
    params_off, _, _ = raw_file_processor._postprocess_params('as_shot', None, None, chroma_denoise=False)
    assert 'median_filter_passes' not in params_off
    assert 'fbdd_noise_reduction' not in params_off

    # Enabled with strength 1.0
    params_1, _, _ = raw_file_processor._postprocess_params(
        'as_shot', None, None, chroma_denoise=True, chroma_denoise_strength=1.0
    )
    assert params_1.get('median_filter_passes') == 2
    assert params_1.get('fbdd_noise_reduction') == rawpy.FBDDNoiseReductionMode.Light

    # Enabled with strength 2.0
    params_2, _, _ = raw_file_processor._postprocess_params(
        'as_shot', None, None, chroma_denoise=True, chroma_denoise_strength=2.0
    )
    assert params_2.get('median_filter_passes') == 3
    assert params_2.get('fbdd_noise_reduction') == rawpy.FBDDNoiseReductionMode.Full


def test_map_highlight_mode():
    import rawpy
    assert raw_file_processor._map_highlight_mode('clip') == rawpy.HighlightMode.Clip
    assert raw_file_processor._map_highlight_mode('ignore') == rawpy.HighlightMode.Ignore
    assert raw_file_processor._map_highlight_mode('blend') == rawpy.HighlightMode.Blend
    assert raw_file_processor._map_highlight_mode('reconstruct') == rawpy.HighlightMode.ReconstructDefault
    assert raw_file_processor._map_highlight_mode(2) == rawpy.HighlightMode.Blend
    assert raw_file_processor._map_highlight_mode(3) == 3
    assert raw_file_processor._map_highlight_mode(rawpy.HighlightMode.Blend) == rawpy.HighlightMode.Blend


def test_postprocess_params_highlight_and_exposure():
    import rawpy

    # Default: reconstruct highlight mode, no exposure shift, no highlight preservation
    params_default, _, _ = raw_file_processor._postprocess_params('as_shot', None, None)
    assert params_default.get('highlight_mode') == rawpy.HighlightMode.ReconstructDefault
    assert 'exp_shift' not in params_default
    assert 'exp_preserve_highlights' not in params_default

    # Custom highlight recovery and exposure shift
    params_custom, _, _ = raw_file_processor._postprocess_params(
        'as_shot', None, None,
        raw_exposure_ev=-1.5,
        highlight_recovery=0.75,
        highlight_mode='blend',
    )
    assert params_custom.get('highlight_mode') == rawpy.HighlightMode.Blend
    assert np.isclose(params_custom.get('exp_shift'), 2.0 ** -1.5)
    assert np.isclose(params_custom.get('exp_preserve_highlights'), 0.75)


def test_cct_tint_from_wb_multipliers():
    # Valid daylight-like multipliers
    cct, tint = raw_file_processor._cct_tint_from_wb_multipliers([2.0, 1.0, 1.5, 1.0])
    assert 4000.0 <= cct <= 7500.0
    assert 0.5 <= tint <= 2.0

    # Degenerate multipliers fallback to 5500.0, 1.0
    assert raw_file_processor._cct_tint_from_wb_multipliers([]) == (5500.0, 1.0)
    assert raw_file_processor._cct_tint_from_wb_multipliers([0.0, 0.0, 0.0]) == (5500.0, 1.0)
    assert raw_file_processor._cct_tint_from_wb_multipliers([-1.0, 1.0, 1.0]) == (5500.0, 1.0)


def test_read_raw_white_balance_from_rawpy(monkeypatch, tmp_path):
    raw_file = tmp_path / "test.cr2"
    raw_file.write_bytes(b"\x00" * 100)

    class FakeRaw:
        camera_whitebalance = [1.8, 1.0, 1.6, 1.0]
        daylight_whitebalance = [2.0, 1.0, 1.5, 1.0]
        rgb_xyz_matrix = np.eye(3)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(raw_file_processor.rawpy, "imread", lambda p: FakeRaw())
    # Mock exiv2 to fail
    monkeypatch.setattr(
        raw_file_processor,
        "exiv2",
        SimpleNamespace(ImageFactory=SimpleNamespace(open=lambda p: (_ for _ in ()).throw(RuntimeError))),
    )

    cct, tint = raw_file_processor.read_raw_white_balance(raw_file)
    assert 4000.0 <= cct <= 7500.0
    assert 0.5 <= tint <= 2.0


def test_read_raw_white_balance_from_exif(monkeypatch, tmp_path):
    raw_file = tmp_path / "test.nef"
    raw_file.write_bytes(b"\x00" * 100)

    class FakeExifVal:
        def __init__(self, val):
            self._val = val
        def value(self):
            return self._val

    class FakeImage:
        def readMetadata(self):
            pass
        def exifData(self):
            return {
                "Exif.Photo.ColorTemperature": FakeExifVal("5600"),
            }
        def xmpData(self):
            return {
                "Xmp.crs.Tint": FakeExifVal("10"),
            }

    monkeypatch.setattr(raw_file_processor.rawpy, "imread", lambda p: (_ for _ in ()).throw(RuntimeError))
    monkeypatch.setattr(
        raw_file_processor,
        "exiv2",
        SimpleNamespace(ImageFactory=SimpleNamespace(open=lambda p: FakeImage())),
    )

    cct, tint = raw_file_processor.read_raw_white_balance(raw_file)
    assert cct == 5600.0
    assert tint == 1.10  # 1.0 + 10/100


def test_safe_get_exif_dict_and_xmp_dict():
    # Test None
    assert raw_file_processor._safe_get_exif_dict(None) == {}
    assert raw_file_processor._safe_get_xmp_dict(None) == {}

    # Test dict pass-through
    d = {"Exif.Image.Make": "Test"}
    assert raw_file_processor._safe_get_exif_dict(d) == d
    assert raw_file_processor._safe_get_xmp_dict(d) == d

    # Test iterable with key attribute
    class MockDatum:
        def __init__(self, k, v):
            self._k = k
            self._v = v
        def key(self):
            return self._k
        def value(self):
            return self._v

    mock_items = [MockDatum("Exif.Image.Model", "CameraX")]
    res = raw_file_processor._safe_get_exif_dict(mock_items)
    assert "Exif.Image.Model" in res
    assert res["Exif.Image.Model"].value() == "CameraX"


def test_read_exif_metadata_safe_when_lens_not_in_exiv2(monkeypatch, tmp_path):
    """Ensure that non-existent lens keys do not fail or raise when checking maker fallbacks."""
    raw_file = tmp_path / "test_unregistered.rw2"
    raw_file.write_bytes(b"\x00" * 100)

    class MockDatum:
        def __init__(self, k, v):
            self._k = k
            self._v = v
        def key(self):
            return self._k
        def value(self):
            return self._v
        def toFloat(self):
            return float(self._v)

    class FakeImage:
        def readMetadata(self):
            pass
        def exifData(self):
            # Returns an iterable of datums (simulating real exiv2 object, not dict)
            return [
                MockDatum("Exif.Image.Make", "Panasonic"),
                MockDatum("Exif.Image.Model", "DC-S9"),
                MockDatum("Exif.Photo.FocalLength", "28.0"),
            ]
        def xmpData(self):
            return []

    fake_exiv2 = SimpleNamespace(
        ImageFactory=SimpleNamespace(open=lambda p: FakeImage()),
        make=lambda exif: None,
        model=lambda exif: None,
        lensName=lambda exif: None,
        focalLength=lambda exif: None,
        fNumber=lambda exif: None,
    )
    monkeypatch.setattr(raw_file_processor, "exiv2", fake_exiv2)

    meta = raw_file_processor._read_exif_metadata(raw_file)
    assert meta.make == "Panasonic"
    assert meta.model == "DC-S9"
    assert meta.focal_length == 28.0
    assert meta.lens_model == ""


def test_resolve_raw_path_reuses_existing_dng(tmp_path):
    nef = tmp_path / "photo.nef"
    nef.write_bytes(b"dummy_nef")
    dng = tmp_path / "photo.dng"
    dng.write_bytes(b"dummy_dng")

    resolved = raw_file_processor.resolve_raw_path(nef)
    assert resolved == dng


def test_resolve_raw_path_converts_when_unsupported(monkeypatch, tmp_path):
    nef = tmp_path / "he_photo.nef"
    nef.write_bytes(b"dummy_nef")
    dng_target = tmp_path / "he_photo.dng"

    class FailingReader:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_value, traceback):
            return False
        def unpack(self):
            raise RuntimeError("LibRawDataError")

    monkeypatch.setattr(raw_file_processor.rawpy, "imread", lambda p: FailingReader())

    converted_called = []
    def fake_convert(raw_path, output_dir=None):
        converted_called.append(raw_path)
        dng_target.write_bytes(b"converted_dng")
        return dng_target

    monkeypatch.setattr(raw_file_processor, "convert_raw_to_dng", fake_convert)

    resolved = raw_file_processor.resolve_raw_path(nef)
    assert resolved == dng_target
    assert len(converted_called) == 1
    assert converted_called[0] == nef


def test_read_raw_white_balance_from_exif_wb_levels(monkeypatch, tmp_path):
    test_file = tmp_path / "nikon_z.nef"
    test_file.write_bytes(b"dummy")

    # Simulate rawpy failing
    monkeypatch.setattr(
        raw_file_processor,
        "rawpy",
        SimpleNamespace(imread=lambda p: (_ for _ in ()).throw(RuntimeError("Unsupported format"))),
    )

    class MockDatum:
        def __init__(self, key, val):
            self._k = key
            self._v = val
        def key(self):
            return self._k
        def value(self):
            return self._v

    class FakeImage:
        def readMetadata(self):
            pass
        def exifData(self):
            return [
                MockDatum("Exif.Nikon3.WB_RBLevels", "825/512 747/512 512/512 512/512"),
            ]
        def xmpData(self):
            return []

    fake_exiv2 = SimpleNamespace(
        ImageFactory=SimpleNamespace(open=lambda p: FakeImage()),
    )
    monkeypatch.setattr(raw_file_processor, "exiv2", fake_exiv2)

    cct, tint = raw_file_processor.read_raw_white_balance(test_file)
    assert 6500.0 <= cct <= 6600.0
    assert 1.0 <= tint <= 1.1









