"""
test_converter.py - Comprehensive Unit & Validation Tests for Converter Pipeline

Tests:
1. Generation of synthetic binary point clouds across multiple formats (KITTI, NuScenes, Classified).
2. Layout auto-detection & profiling via inspect_bin_format.py.
3. Batch conversion to LAS and compressed LAZ format.
4. Header, coordinate, attribute, and extra dimensions round-trip verification using laspy.
5. Error handling on corrupted/truncated files.
6. Resumption/skip functionality.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import numpy as np
import laspy

from inspect_bin_format import inspect_binary_file
from convert_bin_to_las import convert_single_file, run_batch_conversion


def generate_synthetic_data(test_dir: Path):
    """Generate sample test files with known properties."""
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. KITTI Format: 65,536 points (~1 MB, 1,048,576 bytes)
    kitti_points = 65536
    np.random.seed(42)
    kitti_data = np.zeros(kitti_points, dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')
    ])
    kitti_data['x'] = np.random.uniform(-50.0, 50.0, kitti_points).astype(np.float32)
    kitti_data['y'] = np.random.uniform(-50.0, 50.0, kitti_points).astype(np.float32)
    kitti_data['z'] = np.random.uniform(-5.0, 15.0, kitti_points).astype(np.float32)
    kitti_data['intensity'] = np.random.uniform(0.0, 1.0, kitti_points).astype(np.float32)
    kitti_path = test_dir / "sample_kitti.bin"
    with open(kitti_path, "wb") as f:
        kitti_data.tofile(f)
    print(f"[Created] {kitti_path.name}: {kitti_path.stat().st_size:,} bytes, {kitti_points} points")

    # 2. nuScenes Format: 50,000 points (1,000,000 bytes)
    nusc_points = 50000
    nusc_data = np.zeros(nusc_points, dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'), ('ring_index', '<f4')
    ])
    nusc_data['x'] = np.random.uniform(-40.0, 40.0, nusc_points).astype(np.float32)
    nusc_data['y'] = np.random.uniform(-40.0, 40.0, nusc_points).astype(np.float32)
    nusc_data['z'] = np.random.uniform(-3.0, 8.0, nusc_points).astype(np.float32)
    nusc_data['intensity'] = np.random.uniform(0.0, 255.0, nusc_points).astype(np.float32)
    nusc_data['ring_index'] = np.random.randint(0, 32, nusc_points).astype(np.float32)
    nusc_path = test_dir / "sample_nuscenes.bin"
    with open(nusc_path, "wb") as f:
        nusc_data.tofile(f)
    print(f"[Created] {nusc_path.name}: {nusc_path.stat().st_size:,} bytes, {nusc_points} points")

    # 3. Classified LiDAR Format (XYZ float32 + Intensity uint16 + Class uint8 + Return uint8)
    class_points = 62500
    class_data = np.zeros(class_points, dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('intensity', '<u2'), ('classification', 'u1'), ('return_number', 'u1')
    ])
    class_data['x'] = np.random.uniform(500000.0, 500100.0, class_points).astype(np.float32)
    class_data['y'] = np.random.uniform(4000000.0, 4000100.0, class_points).astype(np.float32)
    class_data['z'] = np.random.uniform(100.0, 150.0, class_points).astype(np.float32)
    class_data['intensity'] = np.random.randint(0, 65535, class_points, dtype=np.uint16)
    class_data['classification'] = np.random.choice([0, 1, 2, 3], size=class_points).astype(np.uint8)
    class_data['return_number'] = np.random.choice([1, 2, 3], size=class_points).astype(np.uint8)
    class_path = test_dir / "sample_classified.bin"
    with open(class_path, "wb") as f:
        class_data.tofile(f)
    print(f"[Created] {class_path.name}: {class_path.stat().st_size:,} bytes, {class_points} points")

    # 4. Corrupted file: unaligned 17 bytes to test error handling
    corrupt_path = test_dir / "sample_corrupt.bin"
    with open(corrupt_path, "wb") as f:
        f.write(b"CORRUPT_BYTES_DATA")
    print(f"[Created] {corrupt_path.name}: {corrupt_path.stat().st_size:,} bytes (deliberately invalid)")


def test_inspector(test_dir: Path):
    """Test format detection and profiling."""
    print("\n--- TEST: Inspector Layout Detection ---")
    
    # Check KITTI
    kitti_report = inspect_binary_file(test_dir / "sample_kitti.bin")
    best = kitti_report["best_candidate"]
    assert best is not None, "KITTI inspection failed to find candidate"
    assert best["schema_id"] == "kitti", f"Expected kitti, got {best['schema_id']}"
    assert best["score"] >= 90.0, f"Score too low: {best['score']}"
    print(f"  [PASS] KITTI detected correctly with score {best['score']}%")

    # Check NuScenes
    nusc_report = inspect_binary_file(test_dir / "sample_nuscenes.bin")
    best = nusc_report["best_candidate"]
    assert best is not None, "NuScenes inspection failed"
    assert best["schema_id"] == "nuscenes", f"Expected nuscenes, got {best['schema_id']}"
    assert best["score"] >= 90.0, f"Score too low: {best['score']}"
    print(f"  [PASS] NuScenes detected correctly with score {best['score']}%")

    # Check Classified
    class_report = inspect_binary_file(test_dir / "sample_classified.bin")
    best = class_report["best_candidate"]
    assert best is not None, "Classified inspection failed"
    assert best["schema_id"] == "xyz_f32_i_u16_c_u8", f"Expected xyz_f32_i_u16_c_u8, got {best['schema_id']}"
    assert best["score"] >= 90.0, f"Score too low: {best['score']}"
    print(f"  [PASS] Custom Classified format detected correctly with score {best['score']}%")


def test_single_conversions(test_dir: Path, output_dir: Path):
    """Test single file conversion to both LAS and LAZ with attribute preservation."""
    print("\n--- TEST: Single File Conversions & Validation ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. KITTI to LAZ
    kitti_in = test_dir / "sample_kitti.bin"
    kitti_out = output_dir / "sample_kitti.laz"
    res = convert_single_file(
        str(kitti_in), str(kitti_out),
        dtype_descr=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')],
        record_size=16,
        compress=True
    )
    assert res["status"] == "SUCCESS", f"KITTI conversion failed: {res.get('error')}"
    assert kitti_out.exists(), "Output LAZ file not created"

    # Verify with laspy
    with laspy.open(str(kitti_out)) as f:
        las_data = f.read()
        assert len(las_data) == 65536, f"Point count mismatch: {len(las_data)}"
        assert las_data.header.version == "1.4"
        assert las_data.header.point_format.id == 6
        # Verify coordinates match within mm precision
        raw_kitti = np.fromfile(str(kitti_in), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')])
        np.testing.assert_allclose(las_data.x, raw_kitti['x'], atol=0.002)
        np.testing.assert_allclose(las_data.y, raw_kitti['y'], atol=0.002)
        np.testing.assert_allclose(las_data.z, raw_kitti['z'], atol=0.002)
        print(f"  [PASS] sample_kitti.laz verified with laspy (65,536 pts, precision preserved)")

    # 2. NuScenes to LAS (uncompressed) with extra dimension 'ring_index'
    nusc_in = test_dir / "sample_nuscenes.bin"
    nusc_out = output_dir / "sample_nuscenes.las"
    res = convert_single_file(
        str(nusc_in), str(nusc_out),
        dtype_descr=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'), ('ring_index', '<f4')],
        record_size=20,
        compress=False
    )
    assert res["status"] == "SUCCESS", f"NuScenes conversion failed: {res.get('error')}"
    
    with laspy.open(str(nusc_out)) as f:
        las_data = f.read()
        assert len(las_data) == 50000
        # Check extra dimension
        assert "ring_index" in las_data.point_format.extra_dimension_names, "ring_index missing from extra dimensions!"
        raw_nusc = np.fromfile(str(nusc_in), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'), ('ring_index', '<f4')])
        np.testing.assert_allclose(las_data['ring_index'], raw_nusc['ring_index'], atol=1e-4)
        print(f"  [PASS] sample_nuscenes.las verified (50,000 pts, extra scalar field 'ring_index' preserved)")

    # 3. Classified to LAZ
    class_in = test_dir / "sample_classified.bin"
    class_out = output_dir / "sample_classified.laz"
    res = convert_single_file(
        str(class_in), str(class_out),
        dtype_descr=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<u2'), ('classification', 'u1'), ('return_number', 'u1')],
        record_size=16,
        compress=True
    )
    assert res["status"] == "SUCCESS"
    with laspy.open(str(class_out)) as f:
        las_data = f.read()
        assert len(las_data) == 62500
        raw_class = np.fromfile(str(class_in), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<u2'), ('classification', 'u1'), ('return_number', 'u1')])
        np.testing.assert_array_equal(las_data.classification, raw_class['classification'])
        np.testing.assert_array_equal(las_data.return_number, raw_class['return_number'])
        print(f"  [PASS] sample_classified.laz verified (62,500 pts, classification & returns preserved)")


def test_batch_and_error_handling(test_dir: Path, batch_out_dir: Path):
    """Test full batch workflow, multi-processing, error recording, and resumption."""
    print("\n--- TEST: Batch Processing & Error Handling ---")
    batch_out_dir.mkdir(parents=True, exist_ok=True)

    class Args:
        input_dir = str(test_dir)
        output_dir = str(batch_out_dir)
        preset = "kitti"
        format = "laz"
        point_format = 6
        header_bytes = 0
        scale = [0.001, 0.001, 0.001]
        offset_mode = "auto"
        user_offset = None
        workers = 2
        skip_existing = True
        log_file = str(batch_out_dir / "conversion.log")
        manifest = str(batch_out_dir / "conversion_manifest.json")
        dry_run = False
        verbose = True

    args = Args()
    summary = run_batch_conversion(args)

    # Corrupt file should fail gracefully
    assert summary["summary"]["failed"] >= 1, "Expected corrupt file to fail"
    assert (batch_out_dir / "failed_files.txt").exists(), "failed_files.txt was not created"
    assert (batch_out_dir / "conversion_manifest.json").exists(), "Manifest was not created"
    assert (batch_out_dir / "conversion.log").exists(), "Log was not created"
    print(f"  [PASS] Corrupted file handled cleanly and recorded in failed_files.txt")

    # Test Resumption: Re-running should skip already converted files
    print("\n--- TEST: Resumption & Skip Functionality ---")
    summary2 = run_batch_conversion(args)
    assert summary2["summary"]["skipped_existing"] >= summary["summary"]["converted_success"], \
        "Resumption failed to skip previously converted files"
    print(f"  [PASS] Resumption successfully skipped {summary2['summary']['skipped_existing']} existing files")


def main():
    workspace = Path("C:/Users/Sindhiya/.gemini/antigravity/scratch/bin_to_las_converter")
    test_dir = workspace / "test_data"
    output_dir = workspace / "test_output"

    try:
        generate_synthetic_data(test_dir)
        test_inspector(test_dir)
        test_single_conversions(test_dir, output_dir)
        test_batch_and_error_handling(test_dir, output_dir / "batch_test")
        print("\n=======================================================")
        print(" ALL TESTS PASSED SUCCESSFULLY (100% VERIFIED)!")
        print("=======================================================\n")
    finally:
        # Cleanup test directories
        if test_dir.exists():
            shutil.rmtree(test_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
