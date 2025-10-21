#!/usr/bin/env python3
"""Simple script to compare real_depth.png and left_depth.png images."""

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def normalize_depth(depth_img):
    """Normalize depth image for visualization."""
    if depth_img.dtype == np.uint16:
        normalized = ((depth_img - depth_img.min()) / 
                     (depth_img.max() - depth_img.min() + 1e-8) * 255).astype(np.uint8)
    elif depth_img.dtype == np.uint8:
        normalized = depth_img.copy()
    else:
        normalized = ((depth_img - depth_img.min()) / 
                     (depth_img.max() - depth_img.min() + 1e-8) * 255).astype(np.uint8)
    return normalized


def compare_depth_images():
    """Compare depth images and create visualization."""
    script_dir = Path(__file__).parent
    real_path = script_dir / "real_depth.png"
    left_path = script_dir / "left_depth.png"
    left_depth_path = script_dir/ "left_depth_depth.png"
    
    # Load images
    real_depth = cv2.imread(str(real_path), cv2.IMREAD_UNCHANGED)
    left_depth = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
    left_depth_depth = cv2.imread(str(left_depth_path), cv2.IMREAD_UNCHANGED)
    
    if real_depth is None or left_depth is None or left_depth_depth is None:
        print("Error: Could not load depth images")
        return
    
    # Normalize and resize
    real_norm = normalize_depth(real_depth)
    left_norm = normalize_depth(left_depth)
    left_depth_norm = normalize_depth(left_depth_depth)
    
    if real_norm.shape != left_norm.shape:
        left_norm = cv2.resize(left_norm, (real_norm.shape[1], real_norm.shape[0]))

    if real_norm.shape != left_depth_norm.shape:
        left_depth_norm = cv2.resize(left_depth_norm, (real_norm.shape[1], real_norm.shape[0]))
    
    # Calculate similarity
    diffl = cv2.absdiff(real_norm, left_norm)
    msel = np.mean((real_norm.astype(np.float32) - left_norm.astype(np.float32)) ** 2)

    diffd = cv2.absdiff(real_norm, left_depth_norm)
    msed = np.mean((real_norm.astype(np.float32) - left_depth_norm.astype(np.float32)) ** 2)
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    fig.suptitle("Depth Image Comparison", fontsize=14)
    
    axes[0, 0].imshow(real_norm, cmap='viridis')
    axes[0, 0].set_title("Real Depth")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(left_norm, cmap='viridis')
    axes[0, 1].set_title("Left Depth")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(left_depth_norm, cmap='viridis')
    axes[0, 2].set_title("Left Depth Depth")
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(diffl, cmap='hot')
    axes[1, 0].set_title("Difference Left")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(diffd, cmap='hot')
    axes[1, 1].set_title("Difference Left Depth")
    axes[1, 1].axis('off')
    
    # Overlay
    overlay = cv2.addWeighted(cv2.cvtColor(left_norm, cv2.COLOR_GRAY2RGB), 0.5,
                              cv2.cvtColor(real_norm, cv2.COLOR_GRAY2RGB), 0.5, 0)
    axes[2, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title("Overlay Left")
    axes[2, 0].axis('off')

    overlay = cv2.addWeighted(cv2.cvtColor(left_depth_norm, cv2.COLOR_GRAY2RGB), 0.5,
                              cv2.cvtColor(real_norm, cv2.COLOR_GRAY2RGB), 0.5, 0)
    axes[2, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2, 1].set_title("Overlay Left Depth")
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('depth_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved as 'depth_comparison.png'")
    print(f"MSE - Left: {msel:.2f}")
    print(f"MSE - Depth: {msed:.2f}")
    print("Images are similar - left" if msel < 100 else "Images are different")
    print("Images are similar - depth" if msed < 100 else "Images are different")


if __name__ == "__main__":
    compare_depth_images()
