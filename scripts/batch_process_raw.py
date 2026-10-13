#!/usr/bin/env python3
"""Batch processing script for RAW images using Spektrafilm simulation profiles.

Applies film simulation to RAW files in an input directory using settings from an optional JSON preset,
preserving metadata and outputting processed images into a target directory (creating it if it does not exist).
"""

from __future__ import annotations

import argparse
import datetime
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# Ensure spektrafilm src is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from spektrafilm.runtime.params_builder import digest_params
from spektrafilm.runtime.pipeline import SimulationPipeline
from spektrafilm.utils.io import (
    read_image_metadata,
    save_image_oiio,
    write_image_metadata,
)
from spektrafilm.utils.raw_file_processor import load_and_process_raw_file
from spektrafilm_gui.params_mapper import build_params_from_state
from spektrafilm_gui.persistence import default_gui_state_path, load_gui_state_from_path
from spektrafilm_gui.gallery_sidecar import load_sidecar
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE, clone_gui_state


def setup_fast_scale(pipeline: SimulationPipeline) -> bool:
    """Attach GPU-accelerated antialiased CuPy zoom to pipeline resizing service.
    
    This replaces CPU-bound skimage rescale with CUDA-accelerated zoom,
    speeding up downscaling by ~70-80x while matching anti-aliasing behavior.
    """
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cnd

        orig_resize = pipeline._resize_service
        orig_method = orig_resize.crop_and_rescale

        def fast_crop_and_rescale(image: np.ndarray) -> np.ndarray:
            orig_resize.pixel_size_um = (
                orig_resize.film_format_mm * 1000.0 / float(np.max(image.shape[0:2]))
            )
            orig_resize.origin_x_um = 0.0
            orig_resize.origin_y_um = 0.0

            if orig_resize._io.crop:
                from spektrafilm.utils.crop_resize import crop_image

                image = crop_image(
                    image,
                    center=orig_resize._io.crop_center,
                    size=orig_resize._io.crop_size,
                )

            factor = float(orig_resize._io.upscale_factor)
            if factor != 1.0:
                orig_resize.pixel_size_um /= factor
                g = cp.asarray(image, dtype=np.float32)
                if factor < 1.0:
                    s = (1.0 / factor - 1.0) / 2.0
                    g = cnd.gaussian_filter(g, (s, s, 0.0))
                z = cnd.zoom(g, (factor, factor, 1.0), order=3)
                target_dtype = (
                    np.float32
                    if getattr(getattr(pipeline, "settings", None), "precision", "float32") == "float32"
                    else np.float64
                )
                image = cp.asnumpy(z).astype(target_dtype)
            return image

        orig_resize.crop_and_rescale = fast_crop_and_rescale
        return True
    except Exception as exc:
        print(f"[WARN] GPU fast scale setup failed ({exc}), using standard rescale.")
        return False


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS string."""
    seconds = max(0.0, seconds)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def find_raw_files(input_dir: Path, recursive: bool = True) -> list[Path]:
    """Find all supported RAW image files in input directory."""
    patterns = ("*.rw2", "*.RW2", "*.arw", "*.ARW", "*.cr2", "*.CR2", "*.cr3", "*.CR3", "*.nef", "*.NEF", "*.dng", "*.DNG")
    files: list[Path] = []
    seen: set[str] = set()

    for pattern in patterns:
        it = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        for path in it:
            abs_str = str(path.resolve()).lower()
            if abs_str not in seen:
                seen.add(abs_str)
                files.append(path)

    # Sort files naturally by filename
    files.sort(key=lambda p: (p.name.lower(), str(p)))
    return files


def batch_process(
    input_dir: Path,
    output_dir: Path,
    preset_path: Optional[Path] = None,
    film_stock_override: Optional[str] = None,
    print_paper_override: Optional[str] = None,
    output_ext: str = "jpg",
    precision: Optional[str] = None,
    scale_override: Optional[float] = None,
    grain_blur_override: Optional[float] = None,
    lens_correction: Optional[bool] = None,
    skip_existing: bool = True,
    fast_scale: bool = True,
    random_grain: bool = True,
    grain_seed: Optional[int] = None,
    chroma_denoise: Optional[bool] = None,
    chroma_denoise_strength: Optional[float] = None,
    limit: Optional[int] = None,
    recursive: bool = True,
) -> None:
    """Run batch film simulation over RAW files."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Ensure Windows desktop interactivity is never frozen by background processing
    if sys.platform == "win32":
        try:
            import psutil
            p = psutil.Process()
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print("Set process priority to BELOW_NORMAL (keeps Windows desktop responsive).")
        except Exception:
            try:
                import ctypes
                # BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
                ctypes.windll.kernel32.SetPriorityClass(
                    ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000
                )
                print("Set process priority to BELOW_NORMAL (keeps Windows desktop responsive).")
            except Exception:
                pass

    # Check and create output directory if it does not exist
    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        print("Creating output directory...")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory exists: {output_dir}")

    print("=" * 80)
    print("Spektrafilm Batch RAW Processor")
    print("=" * 80)
    print(f"Input Directory:   {input_dir}")
    print(f"Output Directory:  {output_dir}")
    print(f"Settings JSON:     {preset_path if preset_path is not None else '(None - using default settings)'}")
    print(f"Output Format:     .{output_ext}")
    print(f"Upscale Factor:    {scale_override if scale_override is not None else '(from preset/default)'}")
    print(f"Grain Blur:        {grain_blur_override if grain_blur_override is not None else '(from preset/default)'}")
    print(f"Random Grain:      {random_grain}")
    print(f"Skip Existing:     {skip_existing}")

    # Load settings: from optional JSON file or defaults
    if preset_path is not None:
        if not preset_path.exists():
            raise FileNotFoundError(f"Settings JSON file does not exist: {preset_path}")
        print(f"\nLoading settings from JSON: {preset_path}")
        gui_state = load_gui_state_from_path(preset_path)
    else:
        saved_default = default_gui_state_path()
        if saved_default.exists():
            print(f"\nNo settings JSON specified; loading user default settings from: {saved_default}")
            gui_state = load_gui_state_from_path(saved_default)
        else:
            print("\nNo settings JSON specified; using default Spektrafilm configuration.")
            gui_state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)

    # Apply optional film stock and paper overrides
    if film_stock_override is not None:
        print(f"Overriding film stock: {film_stock_override}")
        if hasattr(gui_state, "simulation") and hasattr(gui_state.simulation, "selection"):
            gui_state.simulation.selection.film_stock = film_stock_override
    if print_paper_override is not None:
        print(f"Overriding print paper: {print_paper_override}")
        if hasattr(gui_state, "simulation") and hasattr(gui_state.simulation, "selection"):
            gui_state.simulation.selection.print_paper = print_paper_override

    params = build_params_from_state(gui_state)
    if precision is not None:
        print(f"Overriding precision: {getattr(params.settings, 'precision', 'float32')} -> {precision}")
        if hasattr(params, "settings"):
            params.settings.precision = precision

    if scale_override is not None:
        print(f"Overriding upscale_factor: {params.io.upscale_factor} -> {scale_override}")
        params.io.upscale_factor = float(scale_override)

    if grain_blur_override is not None:
        print(f"Overriding grain blur: {params.film_render.grain.blur} -> {grain_blur_override}")
        params.film_render.grain.blur = float(grain_blur_override)

    active_scale = params.io.upscale_factor
    print(f"Active upscale_factor: {active_scale}")

    # Digest params and initialize single pipeline
    print("Initializing SimulationPipeline...")
    digested_params = digest_params(params)
    pipeline = SimulationPipeline(digested_params)

    if grain_blur_override is not None:
        pipeline._filming_stage._film_render.grain.blur = float(grain_blur_override)
        if hasattr(pipeline, "film_render") and hasattr(pipeline.film_render, "grain"):
            pipeline.film_render.grain.blur = float(grain_blur_override)

    if fast_scale and active_scale != 1.0:
        fast_ok = setup_fast_scale(pipeline)
        print(f"CuPy GPU fast-rescale active: {fast_ok}")
    else:
        print("Using standard CPU rescale.")

    # Determine color space & encoding settings
    workflow = getattr(gui_state.simulation, "workflow", None)
    saving_color_space = getattr(workflow, "saving_color_space", "sRGB") if workflow else "sRGB"
    saving_cctf_encoding = getattr(workflow, "saving_cctf_encoding", True) if workflow else True

    load_raw = getattr(gui_state.gui_only, "load_raw", None)
    raw_wb = getattr(load_raw, "white_balance", "as_shot") if load_raw else "as_shot"
    raw_temp = getattr(load_raw, "temperature", 5500.0) if load_raw else 5500.0
    raw_tint = getattr(load_raw, "tint", 1.0) if load_raw else 1.0
    raw_lens_corr = (
        lens_correction
        if lens_correction is not None
        else (getattr(load_raw, "lens_correction", False) if load_raw else False)
    )
    raw_chroma_denoise = (
        chroma_denoise
        if chroma_denoise is not None
        else (getattr(load_raw, "chroma_denoise", False) if load_raw else False)
    )
    raw_chroma_denoise_strength = (
        chroma_denoise_strength
        if chroma_denoise_strength is not None
        else (getattr(load_raw, "chroma_denoise_strength", 1.0) if load_raw else 1.0)
    )

    input_io = getattr(gui_state.input_image, "io", None)
    input_cs = getattr(input_io, "input_color_space", "ProPhoto RGB") if input_io else "ProPhoto RGB"
    input_cctf = getattr(input_io, "input_cctf_decoding", False) if input_io else False

    selection = getattr(gui_state.simulation, "selection", None)
    film_stock = getattr(selection, "film_stock", getattr(params.film, "name", "unknown")) if selection else getattr(params.film, "name", "unknown")
    print_paper = getattr(selection, "print_paper", getattr(params.print, "name", "unknown")) if selection else getattr(params.print, "name", "unknown")

    print(f"Film Stock:        {film_stock}")
    print(f"Print Paper:       {print_paper}")
    print(f"Input ColorSpace:  {input_cs}")
    print(f"Save ColorSpace:   {saving_color_space} (cctf={saving_cctf_encoding})")
    if raw_wb == "as_shot":
        wb_info = "as_shot (camera metadata / embedded)"
    elif raw_wb == "custom":
        wb_info = f"custom ({raw_temp:.0f}K, tint={raw_tint})"
    else:
        wb_info = f"{raw_wb}"
    print(f"RAW WhiteBalance:  {wb_info}")
    print(f"Lens Correction:   {raw_lens_corr}")
    print(f"Chroma Denoise:    {raw_chroma_denoise} (strength={raw_chroma_denoise_strength})")
    print(f"Precision:         {getattr(params.settings, 'precision', 'float32')}")

    # Discover files
    print(f"\nScanning for RAW files in {input_dir}...")
    raw_files = find_raw_files(input_dir, recursive=recursive)
    total_found = len(raw_files)
    print(f"Found {total_found} RAW image(s).")

    if limit is not None and limit > 0:
        raw_files = raw_files[:limit]
        print(f"Applied limit: processing {len(raw_files)} file(s).")

    to_process: list[tuple[Path, Path]] = []
    skipped_count = 0

    for rf in raw_files:
        out_filename = f"{rf.stem}.{output_ext}"
        out_path = output_dir / out_filename
        if skip_existing and out_path.exists():
            skipped_count += 1
        else:
            to_process.append((rf, out_path))

    if skipped_count > 0:
        print(f"Skipping {skipped_count} already processed file(s).")

    total_tasks = len(to_process)
    print(f"Total files to process: {total_tasks}")
    print("-" * 80)

    if total_tasks == 0:
        print("All files are already processed! Nothing to do.")
        return

    start_time = time.perf_counter()
    processed_count = 0
    failed_count = 0
    durations: list[float] = []

    for idx, (raw_path, out_path) in enumerate(to_process, 1):
        item_t0 = time.perf_counter()
        raw_name = raw_path.name
        out_name = out_path.name

        img_data = None
        out_img = None
        source_meta = None

        try:
            # Check for per-photo .spktrflm sidecar
            photo_state = load_sidecar(raw_path)
            if photo_state is not None:
                item_load_raw = getattr(photo_state.gui_only, "load_raw", None)
                item_wb = getattr(item_load_raw, "white_balance", raw_wb) if item_load_raw else raw_wb
                item_temp = getattr(item_load_raw, "temperature", raw_temp) if item_load_raw else raw_temp
                item_tint = getattr(item_load_raw, "tint", raw_tint) if item_load_raw else raw_tint
                item_lens = lens_correction if lens_correction is not None else (getattr(item_load_raw, "lens_correction", raw_lens_corr) if item_load_raw else raw_lens_corr)
                item_denoise = chroma_denoise if chroma_denoise is not None else (getattr(item_load_raw, "chroma_denoise", raw_chroma_denoise) if item_load_raw else raw_chroma_denoise)
                item_denoise_strength = chroma_denoise_strength if chroma_denoise_strength is not None else (getattr(item_load_raw, "chroma_denoise_strength", raw_chroma_denoise_strength) if item_load_raw else raw_chroma_denoise_strength)

                item_io = getattr(photo_state.input_image, "io", None)
                item_input_cs = getattr(item_io, "input_color_space", input_cs) if item_io else input_cs
                item_input_cctf = getattr(item_io, "input_cctf_decoding", input_cctf) if item_io else input_cctf

                item_workflow = getattr(photo_state.simulation, "workflow", None)
                item_saving_cs = getattr(item_workflow, "saving_color_space", saving_color_space) if item_workflow else saving_color_space
                item_saving_cctf = getattr(item_workflow, "saving_cctf_encoding", saving_cctf_encoding) if item_workflow else saving_cctf_encoding

                item_params = build_params_from_state(photo_state)
                if precision is not None and hasattr(item_params, "settings"):
                    item_params.settings.precision = precision
                if scale_override is not None:
                    item_params.io.upscale_factor = float(scale_override)
                if grain_blur_override is not None:
                    item_params.film_render.grain.blur = float(grain_blur_override)

                item_pipeline = SimulationPipeline(digest_params(item_params))
                if grain_blur_override is not None:
                    item_pipeline._filming_stage._film_render.grain.blur = float(grain_blur_override)
                if fast_scale and item_params.io.upscale_factor != 1.0:
                    setup_fast_scale(item_pipeline)
            else:
                item_wb = raw_wb
                item_temp = raw_temp
                item_tint = raw_tint
                item_lens = raw_lens_corr
                item_denoise = raw_chroma_denoise
                item_denoise_strength = raw_chroma_denoise_strength
                item_input_cs = input_cs
                item_input_cctf = input_cctf
                item_saving_cs = saving_color_space
                item_saving_cctf = saving_cctf_encoding
                item_pipeline = pipeline

            # 1. Read metadata from RAW
            source_meta = read_image_metadata(str(raw_path))

            # 2. Demosaic RAW
            load_t0 = time.perf_counter()
            lens_info: dict[str, str] = {}
            img_data = load_and_process_raw_file(
                str(raw_path),
                white_balance=item_wb,
                temperature=item_temp,
                tint=item_tint,
                lens_correction=item_lens,
                chroma_denoise=item_denoise,
                chroma_denoise_strength=item_denoise_strength,
                output_colorspace=item_input_cs,
                output_cctf_encoding=item_input_cctf,
                lens_info_out=lens_info,
            )
            load_elapsed = time.perf_counter() - load_t0

            # 3. Simulate (randomize grain seed per photo if random_grain is True)
            if random_grain:
                item_pipeline._filming_stage._film_render.grain.seed = int(np.random.randint(1, 2**31 - 1))
            elif grain_seed is not None:
                item_pipeline._filming_stage._film_render.grain.seed = int(grain_seed)

            sim_t0 = time.perf_counter()
            out_img = item_pipeline.process(img_data)
            sim_elapsed = time.perf_counter() - sim_t0

            # 4. Save image + metadata using local SSD tempfile staging to avoid
            # multi-second synchronous I/O stalls on external/removable drives
            save_t0 = time.perf_counter()
            with tempfile.NamedTemporaryFile(suffix=f".{output_ext}", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            try:
                save_image_oiio(
                    str(tmp_path),
                    out_img,
                    color_space=item_saving_cs,
                    cctf_encoding=item_saving_cctf,
                )
                write_image_metadata(
                    str(tmp_path),
                    source_meta,
                    saving_color_space=item_saving_cs,
                    saving_cctf_encoding=item_saving_cctf,
                )
                shutil.move(str(tmp_path), str(out_path))
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

            save_elapsed = time.perf_counter() - save_t0

            item_elapsed = time.perf_counter() - item_t0
            durations.append(item_elapsed)
            processed_count += 1

            # Calculate ETA
            avg_duration = sum(durations[-20:]) / len(durations[-20:])
            remaining_items = total_tasks - idx
            eta_seconds = avg_duration * remaining_items
            pct = (idx / total_tasks) * 100.0

            print(
                f"[{idx}/{total_tasks}] ({pct:5.1f}%) {raw_name} -> {out_name} | "
                f"Raw: {load_elapsed:4.2f}s | Sim: {sim_elapsed:4.2f}s | Save: {save_elapsed:4.2f}s | "
                f"Total: {item_elapsed:4.2f}s | ETA: {format_duration(eta_seconds)}",
                flush=True,
            )
            if item_lens:
                lens_summary = lens_info.get("summary")
                if lens_summary:
                    print(f"       -> Lens: {lens_summary} (Lensfun corrected)", flush=True)
                else:
                    print(f"       -> Lens: No matching Lensfun profile", flush=True)
            if photo_state is not None:
                print(f"       -> [Sidecar] Applied settings from {raw_path.stem}.spktrflm", flush=True)

        except Exception as exc:
            failed_count += 1
            print(f"[{idx}/{total_tasks}] [ERROR] Failed to process {raw_name}: {exc}", flush=True)

        finally:
            del img_data
            del out_img
            del source_meta
            from spektrafilm.cuda.device import is_cuda_available, free_memory
            if is_cuda_available():
                free_memory()
            import gc
            gc.collect()

    total_elapsed = time.perf_counter() - start_time
    avg_per_image = total_elapsed / max(1, processed_count)

    print("=" * 80)
    print("Batch Processing Summary")
    print("=" * 80)
    print(f"Total RAWs found:       {total_found}")
    print(f"Already existing/skip:  {skipped_count}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed:                 {failed_count}")
    print(f"Total time:             {format_duration(total_elapsed)} ({total_elapsed:.1f}s)")
    if processed_count > 0:
        print(f"Average time per image: {avg_per_image:.2f}s")
    print(f"Output directory:       {output_dir}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Spektrafilm emulation to a directory of RAW images with optional settings JSON, input, and output folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        "--input-dir",
        dest="input_dir",
        type=Path,
        default=Path(r"F:\DCIM"),
        help="Input directory containing RAW images",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path(r"F:\DCIM\output"),
        help="Output directory to save processed images (created automatically if not existing)",
    )
    parser.add_argument(
        "-s",
        "--settings",
        "-p",
        "--preset",
        "--settings-json",
        dest="preset",
        type=Path,
        default=None,
        help="Optional path to Spektrafilm settings JSON file (e.g. profiles/cineblur.json). If omitted, default profile settings are used.",
    )
    parser.add_argument(
        "--film-stock",
        type=str,
        default=None,
        help="Optional film stock profile override (e.g. kodak_vision3_250d, kodak_gold_200)",
    )
    parser.add_argument(
        "--print-paper",
        type=str,
        default=None,
        help="Optional print paper profile override (e.g. fujifilm_crystal_archive_typeii, kodak_supra_endura)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["jpg", "jpeg", "png", "tif", "tiff"],
        default="jpg",
        help="Output image format",
    )
    parser.add_argument(
        "-u",
        "--upscale-factor",
        "--upscale_factor",
        "--scale",
        "--scale-factor",
        dest="scale",
        type=float,
        default=None,
        help="Override upscale_factor / rescale ratio (e.g. 0.4, 0.3, 0.5, 1.0; default: use value from settings JSON or 1.0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping them",
    )
    parser.add_argument(
        "--exact-scale",
        action="store_true",
        help="Use exact CPU skimage bicubic rescale instead of CuPy GPU zoom",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (useful for testing)",
    )
    parser.add_argument(
        "--random-grain",
        dest="random_grain",
        action="store_true",
        default=True,
        help="Generate a unique random grain pattern for each photo (default: True)",
    )
    parser.add_argument(
        "--fixed-grain",
        dest="random_grain",
        action="store_false",
        help="Use a fixed deterministic grain pattern across all photos",
    )
    parser.add_argument(
        "--grain-seed",
        type=int,
        default=None,
        help="Set an explicit integer seed for the grain generator",
    )
    parser.add_argument(
        "--grain-blur",
        "--grain_blur",
        dest="grain_blur",
        type=float,
        default=None,
        help="Override grain blur sigma (e.g. 0.4, 0.65, 1.0, 1.5; default: use value from settings/preset, typically 0.65)",
    )
    parser.add_argument(
        "--lens-correction",
        dest="lens_correction",
        action="store_true",
        default=None,
        help="Enable Lensfun lens correction (vignetting, TCA, distortion)",
    )
    parser.add_argument(
        "--no-lens-correction",
        dest="lens_correction",
        action="store_false",
        default=None,
        help="Disable Lensfun lens correction (overriding preset settings)",
    )
    parser.add_argument(
        "--chroma-denoise",
        dest="chroma_denoise",
        action="store_true",
        default=None,
        help="Enable luminance-guided chrominance denoising before film emulation",
    )
    parser.add_argument(
        "--no-chroma-denoise",
        dest="chroma_denoise",
        action="store_false",
        default=None,
        help="Disable chrominance denoising (overriding preset settings)",
    )
    parser.add_argument(
        "--chroma-denoise-strength",
        type=float,
        default=None,
        help="Chroma denoiser filter strength (default: 1.0 or value from settings JSON)",
    )
    parser.add_argument(
        "--precision",
        choices=["float32", "float64"],
        default=None,
        help="Optional compute precision override ('float32' or 'float64'). If omitted, uses precision from settings JSON or defaults to float32.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not scan subdirectories recursively",
    )

    args = parser.parse_args()

    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        preset_path=args.preset,
        film_stock_override=args.film_stock,
        print_paper_override=args.print_paper,
        output_ext=args.format,
        precision=args.precision,
        scale_override=args.scale,
        grain_blur_override=args.grain_blur,
        lens_correction=args.lens_correction,
        skip_existing=not args.overwrite,
        fast_scale=not args.exact_scale,
        random_grain=args.random_grain,
        grain_seed=args.grain_seed,
        chroma_denoise=args.chroma_denoise,
        chroma_denoise_strength=args.chroma_denoise_strength,
        limit=args.limit,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main()
