# Binary Point Cloud to LAS/LAZ Batch Conversion Pipeline

A high-performance Python toolkit to inspect raw binary point-cloud files (`.bin`), deduce their byte structure and attributes, and batch-convert them into ASPRS LAS or compressed LAZ format with zero data loss.

Developed for LiDAR deep learning pipelines and verified for full compatibility with [CloudCompare](https://www.danielgm.net/cc/).

---

## 1. Pipeline Overview

Converting 7,500 binary files (~1 MB each, ~7.5 GB total) requires a two-phase workflow:

```text
[7,500 .bin Files]
        │
        ▼
 Phase 1: inspect_bin_format.py
        │   ├── Divisibility & Byte Record Size
        │   ├── Dtype Permutations (float32, float64, uint8, uint16)
        │   ├── Coordinate Range & CRS Check (Local vs UTM)
        │   └── Intensity & Classification Scoring
        ▼
 Phase 2: convert_bin_to_las.py
        │   ├── High-throughput Multi-processing (ProcessPoolExecutor)
        │   ├── Scale & Offset Preservation (Millimeter precision)
        │   ├── Extra Bytes Registration (Ring Index, Elongation, etc.)
        │   ├── Atomic Writes (.tmp -> target)
        │   └── Resumption & Error Manifest (failed_files.txt, conversion.log)
        ▼
[Validated .laz / .las Point Clouds]  ──▶ CloudCompare / LiDAR ML Training
```

---

## 2. Prerequisites

The following packages are required and pre-configured in your Python environment:
```powershell
pip install numpy laspy[lazrs] tqdm
```

---

## 3. Step 1: Identify Binary File Structure (Mandatory)

Before running the full batch conversion, analyze either a single `.bin` file or pass the entire input directory. The inspector checks record divisibility, runs physical statistical tests across candidate schemas, and computes a confidence score:

```powershell
python inspect_bin_format.py "C:\path\to\your\input_dir"
```

Or for a specific file:
```powershell
python inspect_bin_format.py "C:\path\to\your\sample_001.bin"
```

### What the Inspector Analyzes:
1. **Divisibility:** Tests exact file size against common record lengths (12, 16, 20, 24, 26, 28, 32 bytes).
2. **Coordinate Sanity:** Detects NaN/Inf, degenerate/flat clouds, and unphysical exponents.
3. **Coordinate Frame:** Differentiates local sensor frame (e.g. $[-100\text{m}, +100\text{m}]$) from projected UTM coordinates ($>1,000\text{m}$).
4. **Intensity & Scalar Attributes:** Validates ranges (normalized $[0, 1]$, 8-bit $[0, 255]$, or 16-bit $[0, 65535]$), subnormal float values, and classification codes.
5. **Output Recommendations:** Directly provides the exact `--preset`, `--scale`, and conversion command.

---

## 4. Step 2: High-Performance Batch Conversion

Once the layout is identified (or using `--preset auto`), run the batch converter:

### Recommended: Compressed LAZ with Auto-Detection & Multi-processing
```powershell
python convert_bin_to_las.py `
    --input-dir "C:\path\to\input_bin_folder" `
    --output-dir "C:\path\to\output_laz_folder" `
    --preset auto `
    --format laz `
    --workers 8
```

### Predefined Presets:
| Preset | Layout / Fields | Record Size | Common Source |
| :--- | :--- | :--- | :--- |
| `kitti` | $X, Y, Z$ (`f4`), Intensity (`f4`) | 16 bytes | KITTI LiDAR benchmark |
| `nuscenes` | $X, Y, Z$ (`f4`), Intensity (`f4`), Ring (`f4`) | 20 bytes | nuScenes dataset |
| `waymo_approx` | $X, Y, Z$ (`f4`), Intensity (`f4`), Elongation (`f4`) | 20 bytes | Waymo Open / Pandaset |
| `xyz_f32` | $X, Y, Z$ (`f4`) | 12 bytes | Unattributed coordinates |
| `xyz_f64` | $X, Y, Z$ (`f8`) | 24 bytes | Double precision / UTM scans |
| `xyz_f64_i_f32` | $X, Y, Z$ (`f8`), Intensity (`f4`) | 28 bytes | Georeferenced aerial scans |
| `xyz_f32_i_u16_c_u8` | $X,Y,Z$ (`f4`), Int (`u2`), Class (`u1`), Ret (`u1`) | 16 bytes | Compact classified LiDAR |

Example with explicit preset:
```powershell
python convert_bin_to_las.py -i "C:\data\raw_bins" -o "C:\data\converted_laz" --preset kitti --format laz
```

---

## 5. Resilience, Resumption & Error Handling

- **Automatic Resumption:** By default, files already successfully converted in the output folder are skipped (`--skip-existing`). If the conversion is interrupted, simply re-running the exact same command resumes immediately where it left off.
- **Atomic File Writing:** Converted files are written to `.tmp` files and only renamed upon successful `laspy` flush. Corrupted or aborted conversions never leave partial or broken files.
- **Dedicated Error Logging:**
  - `failed_files.txt`: Plain text list of failed filenames and reasons (e.g. unaligned byte sizes or I/O corruption).
  - `conversion.log`: Timestamped execution log.
  - `conversion_manifest.json`: Machine-readable audit file containing throughput, point counts, and bounding boxes.

---

## 6. CloudCompare Inspection Verification

When opening converted `.laz` or `.las` files in CloudCompare:
1. Coordinates are stored with optimal millimeter scale factors (`0.001`), preventing quantization artifacts.
2. In UTM/projected coordinates, offsets prevent 32-bit integer coordinate overflow.
3. Intensities are mapped to standard ASPRS unsigned 16-bit integers $[0, 65535]$.
4. Custom scalar fields (such as `ring_index` or `elongation`) are registered as standard LAS Extra Bytes and appear directly in CloudCompare's **Scalar Fields** selector.
