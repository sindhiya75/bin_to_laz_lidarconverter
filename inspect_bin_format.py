"""
inspect_bin_format.py - Binary Point Cloud Layout & Schema Inspector

Analyzes binary point-cloud files (.bin) to identify:
1. Exact byte size and record divisibility.
2. Candidate data layouts and field types (float32, float64, uint8, uint16, etc.).
3. Coordinate ranges (local vs UTM / georeferenced), offsets, and scale factors.
4. Additional scalar attributes (intensity, return number, classification, ring index).
5. Confidence scoring for candidate schemas.
6. Recommended LAS/LAZ conversion parameters.
"""

import os
import sys
import argparse
import struct
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


KNOWN_SCHEMAS = {
    "kitti": {
        "name": "KITTI LiDAR Format",
        "record_size": 16,
        "dtype": np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('intensity', '<f4')
        ]),
        "description": "4 x float32 (X, Y, Z, Intensity)"
    },
    "nuscenes": {
        "name": "nuScenes LiDAR Format",
        "record_size": 20,
        "dtype": np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('intensity', '<f4'),
            ('ring_index', '<f4')
        ]),
        "description": "5 x float32 (X, Y, Z, Intensity, Ring Index)"
    },
    "waymo_approx": {
        "name": "Waymo / Pandaset Format",
        "record_size": 20,
        "dtype": np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('intensity', '<f4'),
            ('elongation', '<f4')
        ]),
        "description": "5 x float32 (X, Y, Z, Intensity, Elongation)"
    },
    "xyz_f32": {
        "name": "Pure XYZ float32",
        "record_size": 12,
        "dtype": np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4')
        ]),
        "description": "3 x float32 (X, Y, Z)"
    },
    "xyz_f64": {
        "name": "Pure XYZ float64 (Georeferenced / Double precision)",
        "record_size": 24,
        "dtype": np.dtype([
            ('x', '<f8'),
            ('y', '<f8'),
            ('z', '<f8')
        ]),
        "description": "3 x float64 (X, Y, Z)"
    },
    "xyz_f64_i_f32": {
        "name": "XYZ float64 + Intensity float32",
        "record_size": 28,
        "dtype": np.dtype([
            ('x', '<f8'),
            ('y', '<f8'),
            ('z', '<f8'),
            ('intensity', '<f4')
        ]),
        "description": "3 x float64 + 1 x float32"
    },
    "xyz_f64_i_u16": {
        "name": "XYZ float64 + Intensity uint16",
        "record_size": 26,
        "dtype": np.dtype([
            ('x', '<f8'),
            ('y', '<f8'),
            ('z', '<f8'),
            ('intensity', '<u2')
        ]),
        "description": "3 x float64 + 1 x uint16"
    },
    "xyz_f32_i_u16_c_u8": {
        "name": "XYZ float32 + Intensity uint16 + Classification uint8 + Return uint8",
        "record_size": 16,
        "dtype": np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('intensity', '<u2'),
            ('classification', 'u1'),
            ('return_number', 'u1')
        ]),
        "description": "3 x float32 + 1 x uint16 + 2 x uint8"
    },
    "xyz_f32_i_f32_c_u8_r_u8": {
        "name": "XYZ float32 + Intensity float32 + Classification uint8 + Return uint8 + 2-byte pad",
        "record_size": 20,
        "dtype": np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('intensity', '<f4'),
            ('classification', 'u1'),
            ('return_number', 'u1'),
            ('_pad', 'u2')
        ]),
        "description": "4 x float32 + 2 x uint8 + 2 bytes padding"
    }
}


def analyze_file_header(raw_bytes: bytes) -> Dict[str, Any]:
    """Examine initial bytes for headers, magic numbers, or ASCII strings."""
    result = {
        "has_ascii_header": False,
        "header_offset": 0,
        "details": []
    }
    
    # Check if starts with common magic signatures or ASCII text
    first_64 = raw_bytes[:64]
    try:
        printable_ascii = sum(1 for b in first_64 if 32 <= b <= 126 or b in (9, 10, 13))
        if printable_ascii > 50:
            clean_text = "".join(chr(b) if (32 <= b <= 126 or b in (9, 10, 13)) else '?' for b in first_64)
            result["has_ascii_header"] = True
            result["details"].append(f"Detected likely ASCII header preview: {repr(clean_text[:40])}")
    except Exception:
        pass

    # Check if first 4 bytes might be uint32 point count
    if len(raw_bytes) >= 16:
        u32_val = struct.unpack('<I', raw_bytes[:4])[0]
        # Check if remaining bytes divide evenly by u32_val
        rem_bytes = len(raw_bytes) - 4
        if 100 < u32_val < 50_000_000 and rem_bytes % u32_val == 0:
            bytes_per_pt = rem_bytes // u32_val
            result["details"].append(
                f"Potential 4-byte header detected: point_count={u32_val:,}, giving {bytes_per_pt} bytes/point"
            )
            result["candidate_header_count"] = u32_val
            result["candidate_header_bytes_per_point"] = bytes_per_pt
            result["header_offset"] = 4

    return result


def evaluate_schema(raw_bytes: bytes, schema_id: str, schema_info: Dict[str, Any], header_offset: int = 0) -> Dict[str, Any]:
    """Evaluate candidate schema against raw byte data and calculate confidence score."""
    rec_size = schema_info["record_size"]
    data_bytes = raw_bytes[header_offset:]
    total_bytes = len(data_bytes)
    
    remainder = total_bytes % rec_size
    num_points = total_bytes // rec_size
    
    if num_points == 0:
        return {"valid": False, "score": 0, "reason": "Too few points"}
    
    # Attempt parsing sample points (up to 20,000 points to keep inspection fast)
    eval_count = min(num_points, 20_000)
    eval_bytes = data_bytes[:eval_count * rec_size]
    
    with np.errstate(all='ignore'):
        try:
            arr = np.frombuffer(eval_bytes, dtype=schema_info["dtype"], count=eval_count)
        except Exception as e:
            return {"valid": False, "score": 0, "reason": f"Dtype parse error: {e}"}
            
        score = 100.0
        penalties = []
        
        if remainder != 0:
            score -= 40.0
            penalties.append(f"Trailing {remainder} unaligned bytes")
        
    # Coordinate validation
    x = arr['x']
    y = arr['y']
    z = arr['z']
    
    for coord_name, c_arr in [('X', x), ('Y', y), ('Z', z)]:
        if np.isnan(c_arr).any():
            score -= 50.0
            penalties.append(f"{coord_name} contains NaN")
        if np.isinf(c_arr).any():
            score -= 50.0
            penalties.append(f"{coord_name} contains Inf")
            
        c_min = float(np.min(c_arr))
        c_max = float(np.max(c_arr))
        c_std = float(np.std(c_arr))
        
        # Check for unphysical exponent values typical of mismatched float representations
        if abs(c_min) > 1e7 or abs(c_max) > 1e7:
            # Could be global UTM coordinates (e.g. UTM Northing can be up to ~10,000,000)
            if abs(c_min) > 1e10 or abs(c_max) > 1e10:
                score -= 45.0
                penalties.append(f"{coord_name} has unphysical extreme magnitude [{c_min:.2e}, {c_max:.2e}]")
                
        # Zero variance means all coordinates are identical
        c_mean = float(np.mean(c_arr, dtype=np.float64))
        c_std = float(np.std(c_arr, dtype=np.float64))
        if c_std < 1e-6 and len(c_arr) > 10:
            score -= 25.0
            penalties.append(f"{coord_name} has zero variance (flat/degenerate)")

    # Intensity validation if present
    intensity_info = None
    if 'intensity' in arr.dtype.names:
        i_arr = arr['intensity']
        if np.issubdtype(i_arr.dtype, np.floating):
            if np.isnan(i_arr).any() or np.isinf(i_arr).any():
                score -= 50.0
                penalties.append("Intensity contains NaN/Inf")
            else:
                i_min = float(np.min(i_arr))
                i_max = float(np.max(i_arr))
                i_mean = float(np.mean(i_arr, dtype=np.float64))
                i_std = float(np.std(i_arr, dtype=np.float64))
                intensity_info = {"min": i_min, "max": i_max, "mean": i_mean, "std": i_std}
                
                # Check for subnormals / packed integers masquerading as float
                # In IEEE 754, small integer bytes (e.g. class=1, return=1) yield floats ~ 1e-38 to 1e-45
                subnormal_count = np.sum((i_arr > 0) & (i_arr < 1e-6))
                if subnormal_count > (len(i_arr) * 0.01):
                    score -= 60.0
                    penalties.append(f"Intensity contains {subnormal_count} subnormal float values (< 1e-6), likely packed integer bytes")
                    
                if i_min < -0.01:
                    score -= 30.0
                    penalties.append(f"Intensity has negative values (min: {i_min})")
                if i_max > 100_000:
                    score -= 35.0
                    penalties.append(f"Intensity abnormally large (max: {i_max:.2e})")
        else:
            # Integer intensity
            i_min = int(np.min(i_arr))
            i_max = int(np.max(i_arr))
            i_mean = float(np.mean(i_arr, dtype=np.float64))
            i_std = float(np.std(i_arr, dtype=np.float64))
            intensity_info = {"min": i_min, "max": i_max, "mean": i_mean, "std": i_std}

    # Classification validation if present
    class_info = None
    if 'classification' in arr.dtype.names:
        c_arr = arr['classification']
        unique_classes = np.unique(c_arr).tolist()
        class_info = {
            "unique_values": unique_classes[:20],
            "total_unique": len(unique_classes)
        }
        # ASPRS standard classification is 0-31 (or custom 0-10)
        max_cls = max(unique_classes) if unique_classes else 0
        if max_cls > 64:
            score -= 40.0
            penalties.append(f"Classification contains values > 64 (max {max_cls}), unlikely class ID")

    # Return number validation if present
    if 'return_number' in arr.dtype.names:
        r_arr = arr['return_number']
        r_max = int(np.max(r_arr))
        r_min = int(np.min(r_arr))
        # ASPRS LAS return numbers are 1..15 (0 is sometimes used for unknown)
        if r_max > 15:
            score -= 60.0
            penalties.append(f"Return number max is {r_max} (> 15 is invalid for LiDAR returns)")
        if r_max == 0:
            score -= 15.0
            penalties.append("All return numbers are 0")

    score = max(0.0, score)
    
    # Calculate coordinate stats
    stats = {
        "x_range": (float(np.min(x)), float(np.max(x))),
        "y_range": (float(np.min(y)), float(np.max(y))),
        "z_range": (float(np.min(z)), float(np.max(z))),
        "num_points": num_points,
        "sample_points": arr[:5].tolist()
    }
    
    return {
        "valid": score > 20.0,
        "score": round(score, 1),
        "schema_id": schema_id,
        "schema_name": schema_info["name"],
        "record_size": rec_size,
        "num_points": num_points,
        "remainder_bytes": remainder,
        "stats": stats,
        "intensity_info": intensity_info,
        "class_info": class_info,
        "penalties": penalties
    }


def inspect_binary_file(file_path, header_offset: int = 0) -> Dict[str, Any]:
    """Perform in-depth analysis on a single .bin point-cloud file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    file_size = file_path.stat().st_size
    with open(file_path, "rb") as f:
        # Read up to first 2MB for analysis
        raw_head = f.read(min(file_size, 2 * 1024 * 1024))
        
    header_analysis = analyze_file_header(raw_head)
    effective_header = header_offset if header_offset > 0 else header_analysis["header_offset"]
    
    evaluations = []
    for schema_id, schema_info in KNOWN_SCHEMAS.items():
        res = evaluate_schema(raw_head, schema_id, schema_info, header_offset=effective_header)
        evaluations.append(res)
        
    # Sort candidate schemas by score descending
    evaluations.sort(key=lambda item: item["score"], reverse=True)
    
    best_candidate = evaluations[0] if evaluations and evaluations[0]["score"] >= 50.0 else None
    
    # Divisibility analysis
    divisibility = {}
    effective_data_size = file_size - effective_header
    for step in [12, 14, 16, 18, 20, 22, 24, 26, 28, 32, 36, 40, 48]:
        rem = effective_data_size % step
        if rem == 0:
            divisibility[step] = effective_data_size // step
            
    return {
        "file_path": str(file_path),
        "file_size": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 3),
        "header_analysis": header_analysis,
        "effective_header": effective_header,
        "divisibility_exact": divisibility,
        "evaluations": evaluations,
        "best_candidate": best_candidate
    }


def print_inspection_report(report: Dict[str, Any]):
    """Print readable analysis report to console."""
    print("=" * 78)
    print(f" BINARY POINT-CLOUD STRUCTURE INSPECTION REPORT")
    print("=" * 78)
    print(f" File: {report['file_path']}")
    print(f" Size: {report['file_size']:,} bytes (~{report['file_size_mb']} MB)")
    
    hdr = report["header_analysis"]
    if hdr["details"]:
        print("\n [Header Analysis]")
        for d in hdr["details"]:
            print(f"  * {d}")
    else:
        print("\n [Header Analysis]: No obvious file header found (continuous point stream)")
        
    div = report["divisibility_exact"]
    print("\n [Divisibility Check (Record Size Candidates)]")
    if div:
        for rec_len, pt_count in div.items():
            print(f"  * {rec_len} bytes/point -> exactly {pt_count:,} points")
    else:
        print("  * Warning: File size is not evenly divisible by standard record sizes (12, 16, 20, 24, 28, 32)")

    print("\n [Candidate Schema Evaluation]")
    print(f" {'Schema':<26} | {'Rec Size':<8} | {'Score':<6} | {'Estimated Points':<16} | {'Status'}")
    print("-" * 78)
    for ev in report["evaluations"][:6]:
        status = "EXCELLENT" if ev["score"] >= 90 else ("PLAUSIBLE" if ev["score"] >= 60 else "UNLIKELY")
        print(f" {ev['schema_id']:<26} | {ev['record_size']:<8} | {ev['score']:<6.1f} | {ev['num_points']:<16,} | {status}")
        if ev["penalties"]:
            print(f"   └─ Issues: {', '.join(ev['penalties'])}")

    best = report["best_candidate"]
    print("\n" + "=" * 78)
    if best and best["score"] >= 60:
        stats = best["stats"]
        print(f" >>> RECOMMENDED STRUCTURE: [{best['schema_id'].upper()}] - {best['schema_name']}")
        print(f" Record byte size: {best['record_size']} bytes")
        print(f" Estimated points: {report['file_size'] // best['record_size']:,} points per file")
        print(f" Coordinate Ranges (First 20k sample):")
        print(f"   X: [{stats['x_range'][0]:.4f}, {stats['x_range'][1]:.4f}]")
        print(f"   Y: [{stats['y_range'][0]:.4f}, {stats['y_range'][1]:.4f}]")
        print(f"   Z: [{stats['z_range'][0]:.4f}, {stats['z_range'][1]:.4f}]")
        
        # Check coordinate system type
        is_utm = max(abs(stats['x_range'][1]), abs(stats['y_range'][1])) > 1000.0
        crs_type = "Projected / UTM (Global)" if is_utm else "Local Sensor Frame (Vehicle / LiDAR Origin)"
        print(f" Estimated Frame: {crs_type}")
        
        if best["intensity_info"]:
            ii = best["intensity_info"]
            print(f" Intensity Range: [{ii['min']:.3f}, {ii['max']:.3f}] (Mean: {ii['mean']:.3f}, Std: {ii['std']:.3f})")
            
        if best["class_info"]:
            ci = best["class_info"]
            print(f" Classification classes: {ci['unique_values']}")

        print("\n Recommended Conversion Settings:")
        print(f"   Preset: --preset {best['schema_id']}")
        scale_val = "0.001 0.001 0.001" if is_utm else "0.001 0.001 0.001"
        print(f"   Scale:  --scale {scale_val} (millimeter precision)")
        print(f"   Format: --format laz (LASzip compressed, optimal for 7,500 files)")
        print(f"\n Conversion Command Example:")
        print(f"   python convert_bin_to_las.py --input-dir <INPUT_DIR> --output-dir <OUTPUT_DIR> --preset {best['schema_id']} --format laz")
    else:
        print(" >>> WARNING: Could not determine layout with high confidence.")
        print(" Please inspect raw bytes or verify sensor/logger documentation.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Inspect binary point-cloud (.bin) data layout and attributes.")
    parser.add_argument("input_path", type=str, help="Path to a .bin file or a directory containing .bin files")
    parser.add_argument("--header-bytes", type=int, default=0, help="Number of header bytes to skip (default: 0)")
    parser.add_argument("--json", action="store_true", help="Output report as structured JSON")
    
    args = parser.parse_args()
    target_path = Path(args.input_path)
    
    if target_path.is_dir():
        # Find first .bin file in directory
        bin_files = list(target_path.glob("*.bin"))
        if not bin_files:
            print(f"Error: No .bin files found in directory '{target_path}'", file=sys.stderr)
            sys.exit(1)
        print(f"Scanning directory: found {len(bin_files)} .bin files.")
        sample_file = bin_files[0]
        print(f"Selecting first file as representative sample: {sample_file.name}\n")
    elif target_path.is_file():
        sample_file = target_path
    else:
        print(f"Error: Path '{target_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    report = inspect_binary_file(sample_file, header_offset=args.header_bytes)
    
    if args.json:
        # Clean numpy types for JSON serialization
        def sanitize(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, bytes):
                return obj.hex()
            raise TypeError(f"Unserializable type {type(obj)}")
            
        print(json.dumps(report, indent=2, default=sanitize))
    else:
        print_inspection_report(report)


if __name__ == "__main__":
    main()
