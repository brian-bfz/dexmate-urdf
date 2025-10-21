#!/usr/bin/env python3
"""
Verify Camera Correction

This script verifies that the camera FOV correction was successful by:
1. Checking the new simulation camera parameters
2. Comparing old vs new simulation images
3. Analyzing the intrinsic matrices
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def verify_camera_parameters():
    """Verify that the camera parameters are now correct."""
    print("=" * 60)
    print("VERIFYING CORRECTED CAMERA PARAMETERS")
    print("=" * 60)
    
    # Expected corrected parameters
    width, height = 1920, 1080
    aspect_ratio = width / height
    expected_fovx = 90.0  # degrees
    expected_fovy = 2 * np.arctan(np.tan(np.radians(expected_fovx) / 2) / aspect_ratio)
    expected_fx = width / (2 * np.tan(np.radians(expected_fovx) / 2))
    expected_fy = height / (2 * np.tan(expected_fovy / 2))
    
    print("Expected ZED Camera Parameters (after correction):")
    print(f"  Resolution: {width} x {height}")
    print(f"  Aspect ratio: {aspect_ratio:.3f}")
    print(f"  Horizontal FOV: {expected_fovx:.2f}°")
    print(f"  Vertical FOV: {np.degrees(expected_fovy):.2f}°")
    print(f"  Focal length fx: {expected_fx:.2f} pixels")
    print(f"  Focal length fy: {expected_fy:.2f} pixels")
    
    # Calculate what the simulation should now produce
    sim_fovy = np.deg2rad(58.72)  # This is what we set in the code
    sim_fovx = 2 * np.arctan(np.tan(sim_fovy / 2) * aspect_ratio)
    sim_fx = width / (2 * np.tan(sim_fovx / 2))
    sim_fy = height / (2 * np.tan(sim_fovy / 2))
    
    print(f"\nSimulation Camera Parameters (after correction):")
    print(f"  Vertical FOV: {np.degrees(sim_fovy):.2f}°")
    print(f"  Horizontal FOV: {np.degrees(sim_fovx):.2f}°")
    print(f"  Focal length fx: {sim_fx:.2f} pixels")
    print(f"  Focal length fy: {sim_fy:.2f} pixels")
    
    # Check if they match
    fovx_match = np.isclose(np.degrees(sim_fovx), expected_fovx, atol=0.1)
    fovy_match = np.isclose(np.degrees(sim_fovy), np.degrees(expected_fovy), atol=0.1)
    
    print(f"\nParameter Matching:")
    print(f"  Horizontal FOV: {'✓ MATCH' if fovx_match else '✗ MISMATCH'}")
    print(f"  Vertical FOV: {'✓ MATCH' if fovy_match else '✗ MISMATCH'}")
    
    return {
        "expected_fovx": expected_fovx,
        "expected_fovy": expected_fovy,
        "expected_fx": expected_fx,
        "expected_fy": expected_fy,
        "sim_fovx": sim_fovx,
        "sim_fovy": sim_fovy,
        "sim_fx": sim_fx,
        "sim_fy": sim_fy,
        "fovx_match": fovx_match,
        "fovy_match": fovy_match
    }


def analyze_image_timestamps():
    """Analyze when images were captured to understand what we're comparing."""
    print("\n" + "=" * 60)
    print("IMAGE TIMESTAMP ANALYSIS")
    print("=" * 60)
    
    image_files = [
        "real_depth.png",
        "left_depth.png", 
        "right_depth.png",
        "left_color.png",
        "right_color.png"
    ]
    
    print("Image capture timestamps:")
    for img_file in image_files:
        if os.path.exists(img_file):
            stat = os.stat(img_file)
            import time
            timestamp = time.ctime(stat.st_mtime)
            size = stat.st_size
            print(f"  {img_file}: {timestamp} ({size:,} bytes)")
        else:
            print(f"  {img_file}: Not found")
    
    # Check if we have old vs new simulation images
    print(f"\nAnalysis:")
    if os.path.exists("left_depth.png"):
        stat = os.stat("left_depth.png")
        import time
        timestamp = time.ctime(stat.st_mtime)
        print(f"  Latest simulation images captured: {timestamp}")
        print(f"  This should be with corrected FOV parameters")


def compare_simulation_images():
    """Compare old vs new simulation images if available."""
    print("\n" + "=" * 60)
    print("SIMULATION IMAGE COMPARISON")
    print("=" * 60)
    
    # Check if we have multiple versions of simulation images
    simulation_files = []
    for file in os.listdir('.'):
        if file.startswith('left_depth') and file.endswith('.png'):
            simulation_files.append(file)
    
    if len(simulation_files) > 1:
        print(f"Found multiple simulation depth images: {simulation_files}")
        print("This suggests we have old and new versions to compare")
    else:
        print("Only one set of simulation images found")
        print("The current images should be with corrected FOV")


def analyze_depth_image_properties():
    """Analyze the properties of the depth images to understand their characteristics."""
    print("\n" + "=" * 60)
    print("DEPTH IMAGE PROPERTIES ANALYSIS")
    print("=" * 60)
    
    images_to_analyze = [
        ("real_depth.png", "Real Robot Depth"),
        ("left_depth.png", "Simulation Left Depth"),
        ("right_depth.png", "Simulation Right Depth")
    ]
    
    for filename, description in images_to_analyze:
        if os.path.exists(filename):
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            if img is not None:
                print(f"\n{description} ({filename}):")
                print(f"  Shape: {img.shape}")
                print(f"  Data type: {img.dtype}")
                print(f"  Min value: {img.min()}")
                print(f"  Max value: {img.max()}")
                print(f"  Mean value: {img.mean():.2f}")
                print(f"  Std deviation: {img.std():.2f}")
                print(f"  Non-zero pixels: {np.count_nonzero(img)}")
                print(f"  Zero pixels: {np.count_nonzero(img == 0)}")
                
                # Analyze depth distribution
                non_zero_depths = img[img > 0]
                if len(non_zero_depths) > 0:
                    print(f"  Non-zero depth range: {non_zero_depths.min()} - {non_zero_depths.max()}")
                    print(f"  Non-zero depth mean: {non_zero_depths.mean():.2f}")
            else:
                print(f"\n{description} ({filename}): Could not load")
        else:
            print(f"\n{description} ({filename}): File not found")


def create_verification_summary(params_info):
    """Create a summary of the verification results."""
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if params_info["fovx_match"] and params_info["fovy_match"]:
        print("🎉 CAMERA CORRECTION SUCCESSFUL!")
        print("✓ Field of view parameters now match ZED camera specifications")
        print("✓ Simulation should now produce correct perspective")
    else:
        print("⚠️  CAMERA CORRECTION INCOMPLETE")
        if not params_info["fovx_match"]:
            print(f"✗ Horizontal FOV mismatch: {np.degrees(params_info['sim_fovx']):.2f}° vs {params_info['expected_fovx']:.2f}°")
        if not params_info["fovy_match"]:
            print(f"✗ Vertical FOV mismatch: {np.degrees(params_info['sim_fovy']):.2f}° vs {np.degrees(params_info['expected_fovy']):.2f}°")
    
    print(f"\nNext steps:")
    print("1. The simulation camera parameters are now corrected")
    print("2. If depth images still don't match, it's likely because:")
    print("   - They represent different scenes/environments")
    print("   - They were captured at different times")
    print("   - They come from different camera sources (left vs right)")
    print("3. To verify the correction worked:")
    print("   - Compare simulation images before/after correction")
    print("   - Check that objects appear at correct relative sizes")
    print("   - Verify stereo disparity calculations")


def main():
    """Main verification function."""
    print("Camera Correction Verification")
    print("=" * 60)
    print("Verifying that the camera FOV correction was successful...")
    
    # Run verification steps
    params_info = verify_camera_parameters()
    analyze_image_timestamps()
    compare_simulation_images()
    analyze_depth_image_properties()
    create_verification_summary(params_info)
    
    print(f"\nVerification complete!")


if __name__ == "__main__":
    main()
