#!/usr/bin/env python3
"""
FOV Calculator for Camera Calibration
Calculates the correct field of view parameters to align simulation with actual robot hardware.
"""

import numpy as np
import math

def calculate_fov_from_horizontal(horizontal_fov_deg, width, height):
    """
    Calculate vertical FOV from horizontal FOV and image dimensions.
    
    Args:
        horizontal_fov_deg: Horizontal field of view in degrees
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        vertical_fov_deg: Vertical field of view in degrees
    """
    aspect_ratio = width / height
    horizontal_fov_rad = np.deg2rad(horizontal_fov_deg)
    
    # Calculate vertical FOV
    vertical_fov_rad = 2 * np.arctan(np.tan(horizontal_fov_rad / 2) / aspect_ratio)
    vertical_fov_deg = np.rad2deg(vertical_fov_rad)
    
    return vertical_fov_deg

def calculate_fov_from_focal_length(focal_length_px, height):
    """
    Calculate FOV from focal length and image height.
    
    Args:
        focal_length_px: Focal length in pixels
        height: Image height in pixels
    
    Returns:
        fov_deg: Field of view in degrees
    """
    fov_rad = 2 * np.arctan(height / (2 * focal_length_px))
    fov_deg = np.rad2deg(fov_rad)
    return fov_deg

def main():
    print("=== Camera FOV Calculator ===\n")
    
    # Current simulation parameters
    sim_width, sim_height = 640, 480
    sim_fovy_deg = 70  # Current simulation FOV
    
    # Actual robot parameters (from your image)
    robot_width, robot_height = 1920, 1080
    
    print(f"Current Simulation:")
    print(f"  Resolution: {sim_width}×{sim_height}")
    print(f"  Aspect ratio: {sim_width/sim_height:.3f}")
    print(f"  Current fovy: {sim_fovy_deg}°")
    print()
    
    print(f"Actual Robot Hardware:")
    print(f"  Resolution: {robot_width}×{robot_height}")
    print(f"  Aspect ratio: {robot_width/robot_height:.3f}")
    print()
    
    # Calculate FOV for different horizontal FOV assumptions
    horizontal_fovs = [85, 90, 95, 100]  # Common ZED camera horizontal FOVs
    
    print("Calculated Vertical FOV for different horizontal FOV assumptions:")
    print("Horizontal FOV → Vertical FOV (for 1920×1080)")
    print("-" * 50)
    
    for h_fov in horizontal_fovs:
        v_fov = calculate_fov_from_horizontal(h_fov, robot_width, robot_height)
        print(f"{h_fov:2d}° → {v_fov:.2f}°")
    
    print()
    
    # Calculate what horizontal FOV the current simulation represents
    sim_aspect = sim_width / sim_height
    sim_h_fov_rad = 2 * np.arctan(np.tan(np.deg2rad(sim_fovy_deg) / 2) * sim_aspect)
    sim_h_fov_deg = np.rad2deg(sim_h_fov_rad)
    
    print(f"Current simulation represents:")
    print(f"  Horizontal FOV: {sim_h_fov_deg:.2f}°")
    print()
    
    # Recommended settings
    print("=== RECOMMENDATIONS ===")
    print("1. Update simulation resolution to match robot hardware:")
    print(f"   width, height = {robot_width}, {robot_height}")
    print()
    print("2. Use appropriate FOV based on your camera specifications:")
    print("   For ZED camera with 90° horizontal FOV:")
    recommended_fovy = calculate_fov_from_horizontal(90, robot_width, robot_height)
    print(f"   fovy = np.deg2rad({recommended_fovy:.2f})  # {recommended_fovy:.2f}°")
    print()
    print("3. If you know the actual horizontal FOV of your robot cameras,")
    print("   use the calculate_fov_from_horizontal() function to get the exact value.")
    
    # Calculate intrinsic matrix parameters
    print("\n=== INTRINSIC MATRIX PARAMETERS ===")
    fovy_rad = np.deg2rad(recommended_fovy)
    focal_length_y = robot_height / (2 * np.tan(fovy_rad / 2))
    focal_length_x = focal_length_y  # Assuming square pixels
    
    print(f"For resolution {robot_width}×{robot_height} with fovy={recommended_fovy:.2f}°:")
    print(f"  Focal length (fx, fy): ({focal_length_x:.1f}, {focal_length_y:.1f}) pixels")
    print(f"  Principal point (cx, cy): ({robot_width/2:.1f}, {robot_height/2:.1f}) pixels")
    
    # Intrinsic matrix
    K = np.array([
        [focal_length_x, 0, robot_width/2],
        [0, focal_length_y, robot_height/2],
        [0, 0, 1]
    ])
    print(f"\nIntrinsic matrix K:")
    print(K)

if __name__ == "__main__":
    main()
