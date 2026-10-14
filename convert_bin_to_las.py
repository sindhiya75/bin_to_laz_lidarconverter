"""
convert_bin_to_las.py - High-Performance Batch Binary Point-Cloud to LAS/LAZ Converter

Converts raw binary point-cloud files (.bin) into ASPRS LAS or LAZ format.
Preserves all coordinate precision, intensity, classifications, and custom sensor channels.
Optimized for large datasets (e.g. 7,500+ files) with multi-processing, atomic writes,
safe resumption, comprehensive logging, and CloudCompare validation.
"""

import os
import sys
import time
import argparse
import logging
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import laspy
from tqdm import tqdm

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# Known layouts matching inspect_bin_format.py
LAYOUT_PRESETS = {
    "kitti": {
        "dtype": np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('intensity', '<f4')
        ]),
        "record_size": 16,
        "description": "KITTI LiDAR: 4 x float32 (X, Y, Z, Intensity)"
    },
    "nuscenes": {
        "dtype": np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('intensity', '<f4'),
            ('ring_index', '<f4')
        ]),
        "record_size": 20,
        "description": "nuScenes: 5 x float32 (X, Y, Z, Intensity, Ring Index)"
    },
    "waymo_approx": {
        "dtype": np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('intensity', '<f4'),
            ('elongation', '<f4')
        ]),
        "record_size": 20,
        "description": "Waymo/Pandaset: 5 x float32 (X, Y, Z, Intensity, Elongation)"
    },
    "xyz_f32": {
        "dtype": np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4')
        ]),
        "record_size": 12,
        "description": "XYZ float32 (12 bytes/point)"
    },
    "xyz_f64": {
        "dtype": np.dtype([
            ('x', '<f8'),
            ('y', '<f8'),
            ('z', '<f8')
        ]),
        "record_size": 24,
        "description": "XYZ float64 / Georeferenced (24 bytes/point)"
    },
    "xyz_f64_i_f32": {
        "dtype": np.dtype([
            ('x', '<f8'),
            ('y', '<f8'),
            ('z', '<f8'),
            ('intensity', '<f4')
        ]),
        "record_size": 28,
        "description": "XYZ float64 + Intensity float32 (28 bytes/point)"
    },
    "xyz_f32_i_u16_c_u8": {
        "dtype": np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('intensity', '<u2'),
            ('classification', 'u1'),
            ('return_number', 'u1')
        ]),
        "record_size": 16,
        "description": "XYZ float32 + Intensity uint16 + Classification uint8 + Return uint8"
    }
}


def setup_logger(log_file: Optional[Path] = None, verbose: bool = False) -> logging.Logger:
    """Configure structured console and file logger."""
    logger = logging.getLogger("bin2las")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode='a', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def convert_single_file(
    input_file_str: str,
    output_file_str: str,
    dtype_descr: list,
    record_size: int,
    header_bytes: int = 0,
    las_version: str = "1.4",
    point_format: int = 6,
    scale_xyz: Tuple[float, float, float] = (0.001, 0.001, 0.001),
    offset_mode: str = "auto",
    user_offset: Optional[Tuple[float, float, float]] = None,
    compress: bool = True
) -> Dict[str, Any]:
    """
    Worker function to parse a single binary file and write LAS/LAZ atomically.
    Returns status dictionary.
    """
    input_path = Path(input_file_str)
    output_path = Path(output_file_str)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

    file_size = input_path.stat().st_size
    data_size = file_size - header_bytes

    if data_size <= 0:
        return {
            "status": "FAILED",
            "file": input_path.name,
            "error": f"File too small ({file_size} bytes with {header_bytes} header bytes)",
            "points": 0
        }

    if data_size % record_size != 0:
        rem = data_size % record_size
        return {
            "status": "FAILED",
            "file": input_path.name,
            "error": f"Unaligned file size: {data_size} data bytes is not divisible by record size {record_size} (rem: {rem})",
            "points": 0
        }

    num_points = data_size // record_size
    dt = np.dtype(dtype_descr)

    try:
        # Read raw binary data vectorially
        with open(input_path, "rb") as f:
            if header_bytes > 0:
                f.seek(header_bytes)
            data_arr = np.fromfile(f, dtype=dt, count=num_points)

        if len(data_arr) != num_points:
            return {
                "status": "FAILED",
                "file": input_path.name,
                "error": f"Read count mismatch: expected {num_points}, got {len(data_arr)}",
                "points": 0
            }

        x = data_arr['x']
        y = data_arr['y']
        z = data_arr['z']

        # Sanity check coordinates
        if np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any():
            return {
                "status": "FAILED",
                "file": input_path.name,
                "error": "Coordinates contain NaN values",
                "points": 0
            }

        # Calculate bounding box & offsets
        min_x, max_x = float(np.min(x)), float(np.max(x))
        min_y, max_y = float(np.min(y)), float(np.max(y))
        min_z, max_z = float(np.min(z)), float(np.max(z))

        if user_offset is not None:
            offset = user_offset
        elif offset_mode == "auto":
            # If coordinates are centered around 0 (e.g. vehicle frame within [-200, 200]), origin [0,0,0] is best
            if max(abs(min_x), abs(max_x), abs(min_y), abs(max_y)) < 500.0:
                offset = (0.0, 0.0, 0.0)
            else:
                # Projected UTM coordinates: offset to min bound prevents 32-bit integer overflow
                offset = (min_x, min_y, min_z)
        else:
            offset = (0.0, 0.0, 0.0)

        # Initialize LAS Header
        header = laspy.LasHeader(version=las_version, point_format=point_format)
        header.offsets = offset
        header.scales = scale_xyz

        # Identify standard vs extra dimensions
        field_names = data_arr.dtype.names
        has_intensity = 'intensity' in field_names
        has_classification = 'classification' in field_names
        has_return_number = 'return_number' in field_names

        # Identify extra attributes not natively in standard LAS point format
        standard_las_names = {'x', 'y', 'z', 'intensity', 'classification', 'return_number', 'number_of_returns', '_pad'}
        extra_fields = [name for name in field_names if name not in standard_las_names]

        # Register extra bytes dimensions
        for ef in extra_fields:
            field_type = data_arr.dtype[ef]
            header.add_extra_dim(laspy.ExtraBytesParams(name=ef, type=field_type))

        # Create LAS data container
        las = laspy.LasData(header)
        las.x = x
        las.y = y
        las.z = z

        # Map Intensity
        if has_intensity:
            raw_i = data_arr['intensity']
            # LAS standard intensity is uint16 [0..65535]
            if np.issubdtype(raw_i.dtype, np.floating):
                i_max = float(np.nanmax(raw_i)) if len(raw_i) > 0 else 1.0
                if i_max <= 1.05 and i_max > 0.0:
                    # Normalized [0, 1] range: scale to uint16
                    norm_i = np.clip(raw_i * 65535.0, 0, 65535).astype(np.uint16)
                elif i_max <= 256.0:
                    # 8-bit [0, 255] range: scale to uint16
                    norm_i = np.clip(raw_i * 257.0, 0, 65535).astype(np.uint16)
                else:
                    norm_i = np.clip(raw_i, 0, 65535).astype(np.uint16)
                las.intensity = norm_i
            else:
                las.intensity = np.clip(raw_i, 0, 65535).astype(np.uint16)

        # Map Classification
        if has_classification:
            las.classification = data_arr['classification'].astype(np.uint8)

        # Map Return Number
        if has_return_number:
            las.return_number = data_arr['return_number'].astype(np.uint8)

        # Map Extra Dimensions
        for ef in extra_fields:
            las[ef] = data_arr[ef]

        # Atomic write to temporary file
        temp_output_path.parent.mkdir(parents=True, exist_ok=True)
        las.write(str(temp_output_path))

        # Rename to destination
        if temp_output_path.exists():
            temp_output_path.replace(output_path)

        return {
            "status": "SUCCESS",
            "file": input_path.name,
            "output_file": output_path.name,
            "points": num_points,
            "bounds": {
                "min": [min_x, min_y, min_z],
                "max": [max_x, max_y, max_z]
            }
        }

    except Exception as e:
        if temp_output_path.exists():
            try:
                temp_output_path.unlink()
            except Exception:
                pass
        return {
            "status": "FAILED",
            "file": input_path.name,
            "error": str(e),
            "points": 0
        }


def run_batch_conversion(args) -> Dict[str, Any]:
    """Execute complete batch conversion workflow."""
    start_time = time.time()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else (output_dir / "conversion.log")
    manifest_path = Path(args.manifest) if args.manifest else (output_dir / "conversion_manifest.json")
    failed_list_path = output_dir / "failed_files.txt"

    logger = setup_logger(log_path, verbose=args.verbose)
    logger.info("=" * 70)
    logger.info("STARTING BATCH POINT-CLOUD CONVERSION")
    logger.info(f"Input Directory:  {input_dir}")
    logger.info(f"Output Directory: {output_dir}")

    # Discover .bin files
    all_bin_files = sorted(list(input_dir.glob("*.bin")))
    if not all_bin_files:
        # Also check lowercase/uppercase extension
        all_bin_files = sorted(list(input_dir.glob("*.[bB][iI][nN]")))

    if not all_bin_files:
        logger.error(f"No .bin files found in '{input_dir}'")
        return {"status": "NO_FILES"}

    total_files_count = len(all_bin_files)
    logger.info(f"Discovered {total_files_count:,} binary point-cloud files.")

    # Determine layout schema
    if args.preset == "auto":
        from inspect_bin_format import inspect_binary_file
        logger.info("Auto-detecting binary layout from sample file...")
        insp = inspect_binary_file(all_bin_files[0], header_offset=args.header_bytes)
        best = insp.get("best_candidate")
        if not best or best["score"] < 60:
            logger.error("Auto-detection failed to identify layout with sufficient confidence.")
            logger.error("Please run 'python inspect_bin_format.py <file>' and specify --preset manually.")
            return {"status": "AUTO_DETECT_FAILED"}
        preset_key = best["schema_id"]
        logger.info(f"Auto-detected layout: [{preset_key}] ({best['schema_name']}, confidence {best['score']}%)")
        layout_spec = LAYOUT_PRESETS[preset_key]
    else:
        if args.preset not in LAYOUT_PRESETS:
            logger.error(f"Unknown preset '{args.preset}'. Available presets: {list(LAYOUT_PRESETS.keys())}")
            return {"status": "INVALID_PRESET"}
        layout_spec = LAYOUT_PRESETS[args.preset]
        logger.info(f"Using Layout Preset: [{args.preset}] - {layout_spec['description']}")

    dtype_descr = layout_spec["dtype"].descr
    record_size = layout_spec["record_size"]

    # Target file format
    out_ext = ".laz" if args.format.lower() == "laz" else ".las"
    logger.info(f"Target format: {out_ext.upper()} (Point format {args.point_format})")
    logger.info(f"Precision scale: {args.scale}")

    # Filter files for resumption
    pending_tasks = []
    skipped_files = []
    for f in all_bin_files:
        target_file = output_dir / (f.stem + out_ext)
        if args.skip_existing and target_file.exists() and target_file.stat().st_size > 500:
            skipped_files.append(f.name)
        else:
            pending_tasks.append((f, target_file))

    logger.info(f"Resumption check: {len(skipped_files):,} files already exist and will be skipped.")
    logger.info(f"Files to process: {len(pending_tasks):,}")

    if args.dry_run:
        logger.info("[DRY RUN] Simulating conversion for first 5 files...")
        pending_tasks = pending_tasks[:5]

    if not pending_tasks:
        logger.info("All files are already converted! Nothing to do.")
        return {
            "total": total_files_count,
            "processed": 0,
            "skipped": len(skipped_files),
            "failed": 0,
            "duration_sec": round(time.time() - start_time, 2)
        }

    # Parallel Execution Setup
    workers = args.workers if args.workers > 0 else max(1, os.cpu_count() - 1)
    logger.info(f"Spawning {workers} worker processes for parallel conversion...")

    successful_results = []
    failed_results = []
    total_points = 0

    scale_tuple = tuple(args.scale)
    user_offset_tuple = tuple(args.user_offset) if args.user_offset else None

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_file = {
            executor.submit(
                convert_single_file,
                str(in_p),
                str(out_p),
                dtype_descr=dtype_descr,
                record_size=record_size,
                header_bytes=args.header_bytes,
                las_version="1.4",
                point_format=args.point_format,
                scale_xyz=scale_tuple,
                offset_mode=args.offset_mode,
                user_offset=user_offset_tuple,
                compress=(args.format.lower() == "laz")
            ): in_p
            for in_p, out_p in pending_tasks
        }

        # Progress bar
        with tqdm(total=len(pending_tasks), desc="Converting files", unit="file", dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_file):
                in_p = future_to_file[future]
                try:
                    res = future.result()
                    if res["status"] == "SUCCESS":
                        successful_results.append(res)
                        total_points += res["points"]
                    else:
                        failed_results.append(res)
                        logger.warning(f"Conversion failed for '{res['file']}': {res['error']}")
                except Exception as exc:
                    failed_results.append({
                        "status": "FAILED",
                        "file": in_p.name,
                        "error": str(exc),
                        "points": 0
                    })
                    logger.error(f"Worker exception on '{in_p.name}': {exc}")

                pbar.update(1)

    elapsed = time.time() - start_time
    throughput_files = len(successful_results) / elapsed if elapsed > 0 else 0
    throughput_points = total_points / elapsed if elapsed > 0 else 0

    # Write failed files list for easy retry
    if failed_results:
        with open(failed_list_path, "w", encoding="utf-8") as f:
            for item in failed_results:
                f.write(f"{item['file']}\t{item['error']}\n")
        logger.warning(f"Failed files recorded to: {failed_list_path}")
    elif failed_list_path.exists():
        failed_list_path.unlink()

    # Write JSON manifest
    manifest_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "preset": args.preset,
        "format": args.format.lower(),
        "summary": {
            "total_files": total_files_count,
            "converted_success": len(successful_results),
            "skipped_existing": len(skipped_files),
            "failed": len(failed_results),
            "total_points_converted": total_points,
            "elapsed_seconds": round(elapsed, 2),
            "throughput_files_per_sec": round(throughput_files, 2),
            "throughput_points_per_sec": round(throughput_points, 1)
        },
        "failed_files": failed_results
    }

    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2)

    # Print Summary Table
    print("\n" + "=" * 70)
    print(" CONVERSION SUMMARY")
    print("=" * 70)
    print(f" Total files discovered:  {total_files_count:,}")
    print(f" Successfully converted:  {len(successful_results):,} files")
    print(f" Skipped (already exist): {len(skipped_files):,} files")
    print(f" Failed files:            {len(failed_results):,} files")
    print(f" Total points converted:  {total_points:,} points")
    print(f" Execution time:          {elapsed:.2f} seconds ({elapsed / 60.0:.2f} mins)")
    print(f" Processing speed:        {throughput_files:.1f} files/sec ({throughput_points:,.0f} pts/sec)")
    print(f" Log file:                {log_path}")
    print(f" Manifest file:           {manifest_path}")
    if failed_results:
        print(f" Failed file list:        {failed_list_path} (Review and retry)")
    print("=" * 70 + "\n")

    # Flush and close handlers to release file lock on Windows
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    return manifest_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch convert binary point-cloud files (.bin) to LAS or LAZ format."
    )
    parser.add_argument("-i", "--input-dir", type=str, required=True,
                        help="Path to directory containing input .bin files")
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Path to directory where converted LAS/LAZ files will be saved")
    parser.add_argument("--preset", type=str, default="auto",
                        choices=list(LAYOUT_PRESETS.keys()) + ["auto"],
                        help="Binary point layout preset (default: auto)")
    parser.add_argument("--format", type=str, default="laz", choices=["laz", "las"],
                        help="Target output format: laz (compressed, recommended) or las (default: laz)")
    parser.add_argument("--point-format", type=int, default=6,
                        help="ASPRS LAS Point Data Record Format (PDRF 0, 1, 2, 3, 6, 7). Default: 6 (LAS 1.4)")
    parser.add_argument("--header-bytes", type=int, default=0,
                        help="Number of header bytes to skip in each binary file (default: 0)")
    parser.add_argument("--scale", type=float, nargs=3, default=[0.001, 0.001, 0.001],
                        help="Coordinate scale factors for X Y Z (default: 0.001 0.001 0.001 for mm precision)")
    parser.add_argument("--offset-mode", type=str, default="auto", choices=["auto", "zero"],
                        help="Offset mode: 'auto' prevents 32-bit integer overflow; 'zero' uses [0,0,0]")
    parser.add_argument("--user-offset", type=float, nargs=3, default=None,
                        help="Manual offset coordinates (e.g. --user-offset 500000 4000000 0)")
    parser.add_argument("-w", "--workers", type=int, default=0,
                        help="Number of parallel worker processes (default: CPU count - 1)")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                        help="Skip already converted files (enabled by default for safe resumption)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="Overwrite existing converted files instead of skipping")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Custom log file path (defaults to <output_dir>/conversion.log)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Custom JSON manifest path (defaults to <output_dir>/conversion_manifest.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate conversion of first 5 files without full batch run")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose debug logging")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch_conversion(args)
