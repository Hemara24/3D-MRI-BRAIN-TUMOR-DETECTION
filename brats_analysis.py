"""
BraTS-Africa Brain Tumor Analysis Script
Complete standalone script for analyzing brain tumor MRI data
"""

# ============================================================
# CELL 1: Install and Import Required Libraries
# ============================================================

import subprocess
import sys

# Install required packages
def install_packages():
    packages = ['nibabel', 'numpy', 'matplotlib', 'scikit-image', 'plotly']
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])

install_packages()

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from glob import glob
from datetime import datetime

print("✅ All libraries imported successfully!")

# ============================================================
# CELL 2: Define Data Paths and Load Functions
# ============================================================

# Your dataset paths - Update these paths to match your actual folder structure
OTHER_NEOPLASMS_PATH = r"C:\Users\M S I\Downloads\2NDYR_RESEARCH\BraTS-Africa\51_OtherNeoplasms"
GLIOMA_PATH = r"C:\Users\M S I\Downloads\2NDYR_RESEARCH\BraTS-Africa\95_Glioma"

def get_all_cases(base_path):
    """Get all patient case folders from a directory"""
    path = Path(base_path)
    if not path.exists():
        print(f"⚠️  Path does not exist: {base_path}")
        return []
    return [d for d in path.iterdir() if d.is_dir()]

def load_nifti(file_path):
    """Load a NIFTI file and return the image data"""
    img = nib.load(str(file_path))
    return img.get_fdata()

def find_modalities(case_folder):
    """Find all MRI modalities and segmentation in a case folder
    BraTS typically has: T1, T1ce, T2, FLAIR, and seg(segmentation)
    """
    files = {}
    for f in Path(case_folder).glob("*.nii.gz"):
        fname = f.name.lower()
        if "t1ce" in fname:
            files['t1ce'] = f
        elif "t1" in fname and "t1ce" not in fname:
            files['t1'] = f
        elif "t2" in fname and "flair" not in fname:
            files['t2'] = f
        elif "flair" in fname:
            files['flair'] = f
        elif "seg" in fname:
            files['seg'] = f
    return files

# List all available cases
print("=== Other Neoplasms Cases ===")
neoplasms_cases = get_all_cases(OTHER_NEOPLASMS_PATH)
print(f"Found {len(neoplasms_cases)} cases")

print("\n=== Glioma Cases ===")
glioma_cases = get_all_cases(GLIOMA_PATH)
print(f"Found {len(glioma_cases)} cases")

# ============================================================
# CELL 3: 2D Slice Visualization for Diagnosis
# ============================================================

def visualize_2d_slices(case_folder, slice_idx=None, save_path=None):
    """
    Visualize 2D slices from all modalities for diagnosis
    """
    modalities = find_modalities(case_folder)
    
    if not modalities:
        print(f"No NIFTI files found in {case_folder}")
        return None
    
    # Load all available modalities
    data = {}
    for name, path in modalities.items():
        data[name] = load_nifti(path)
        print(f"Loaded {name}: shape = {data[name].shape}")
    
    # Get middle slice if not specified
    if slice_idx is None:
        sample = list(data.values())[0]
        slice_idx = sample.shape[2] // 2
    
    # Create visualization
    n_modalities = len(data)
    fig, axes = plt.subplots(1, n_modalities, figsize=(4*n_modalities, 4))
    
    if n_modalities == 1:
        axes = [axes]
    
    for ax, (name, img) in zip(axes, data.items()):
        if name == 'seg':
            # Show segmentation with colormap
            ax.imshow(img[:, :, slice_idx].T, cmap='nipy_spectral', origin='lower')
        else:
            ax.imshow(img[:, :, slice_idx].T, cmap='gray', origin='lower')
        ax.set_title(f"{name.upper()} - Slice {slice_idx}")
        ax.axis('off')
    
    plt.suptitle(f"Case: {Path(case_folder).name}", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()
    
    return data

# Example: Visualize first case from each category
if glioma_cases:
    print("\n--- Glioma Sample ---")
    glioma_data = visualize_2d_slices(glioma_cases[0])

if neoplasms_cases:
    print("\n--- Other Neoplasms Sample ---")
    neoplasms_data = visualize_2d_slices(neoplasms_cases[0])

# ============================================================
# CELL 4: Multi-Slice Grid Visualization
# ============================================================

def visualize_slice_grid(case_folder, n_slices=9, save_path=None):
    """
    Visualize multiple slices in a grid layout
    """
    modalities = find_modalities(case_folder)
    
    if 'flair' not in modalities:
        print("FLAIR modality not found, using first available modality")
        modality_key = list(modalities.keys())[0]
    else:
        modality_key = 'flair'
    
    data = load_nifti(modalities[modality_key])
    total_slices = data.shape[2]
    
    # Select evenly spaced slices
    slice_indices = np.linspace(total_slices * 0.2, total_slices * 0.8, n_slices).astype(int)
    
    # Create grid
    rows = int(np.ceil(np.sqrt(n_slices)))
    cols = int(np.ceil(n_slices / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten()
    
    for idx, (ax, slice_idx) in enumerate(zip(axes, slice_indices)):
        ax.imshow(data[:, :, slice_idx].T, cmap='gray', origin='lower')
        ax.set_title(f"Slice {slice_idx}")
        ax.axis('off')
    
    # Hide empty subplots
    for ax in axes[len(slice_indices):]:
        ax.axis('off')
    
    plt.suptitle(f"Case: {Path(case_folder).name} - {modality_key.upper()}", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

# Example usage
if glioma_cases:
    visualize_slice_grid(glioma_cases[0])

# ============================================================
# CELL 5: 3D Tumor Visualization
# ============================================================

import plotly.graph_objects as go
from skimage import measure

def visualize_3d_tumor(case_folder, threshold=0.5, save_html=None):
    """
    Convert segmentation to 3D surface visualization
    Tumor labels in BraTS:
    - 1: Necrotic/Non-enhancing tumor core
    - 2: Peritumoral edema
    - 3 or 4: Enhancing tumor
    """
    modalities = find_modalities(case_folder)
    
    if 'seg' not in modalities:
        print("No segmentation file found!")
        return None
    
    seg_data = load_nifti(modalities['seg'])
    print(f"Segmentation shape: {seg_data.shape}")
    print(f"Unique labels: {np.unique(seg_data)}")
    
    # Create 3D visualization for each tumor region
    colors = {
        1: 'red',      # Necrotic core
        2: 'green',    # Edema
        4: 'blue'      # Enhancing tumor (label can be 3 or 4)
    }
    
    labels_names = {
        1: 'Necrotic Core',
        2: 'Peritumoral Edema', 
        4: 'Enhancing Tumor'
    }
    
    fig = go.Figure()
    
    for label, color in colors.items():
        if label in seg_data:
            # Create binary mask for this label
            mask = (seg_data == label).astype(np.float32)
            
            if mask.sum() == 0:
                continue
            
            try:
                # Generate 3D surface mesh
                verts, faces, _, _ = measure.marching_cubes(mask, level=threshold)
                
                # Add surface to figure
                fig.add_trace(go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=0.5,
                    name=labels_names.get(label, f'Label {label}')
                ))
            except Exception as e:
                print(f"Could not generate mesh for label {label}: {e}")
    
    fig.update_layout(
        title=f"3D Tumor Visualization: {Path(case_folder).name}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=800,
        height=600
    )
    
    if save_html:
        fig.write_html(save_html)
        print(f"Saved 3D visualization to: {save_html}")
    
    fig.show()
    return seg_data

# Visualize 3D tumor from first glioma case
if glioma_cases:
    print("Generating 3D tumor visualization...")
    seg = visualize_3d_tumor(glioma_cases[0])

# ============================================================
# CELL 6: Diagnostic Report Generator
# ============================================================

def determine_brain_region(x, y, z, shape):
    """
    Approximate brain region based on tumor centroid location
    This is a simplified mapping - real clinical use requires atlas registration
    """
    # Normalize coordinates to 0-1 range
    nx, ny, nz = x/shape[0], y/shape[1], z/shape[2]
    
    # Determine hemisphere
    hemisphere = "Left" if nx < 0.5 else "Right"
    
    # Determine anterior/posterior
    ap = "Frontal" if ny < 0.4 else ("Parietal" if ny < 0.6 else "Occipital")
    
    # Determine superior/inferior
    if nz < 0.33:
        vertical = "Inferior"
    elif nz < 0.66:
        vertical = "Central"
    else:
        vertical = "Superior"
    
    # Special regions
    if 0.4 < nx < 0.6 and 0.3 < ny < 0.5 and 0.3 < nz < 0.6:
        return "Deep Brain / Thalamic Region"
    if ny > 0.7 and nz < 0.4:
        return f"{hemisphere} Temporal Lobe"
    if ny < 0.3 and 0.4 < nz < 0.7:
        return f"{hemisphere} Frontal Lobe"
    
    return f"{hemisphere} {ap} Lobe ({vertical})"

def generate_diagnostic_report(case_folder):
    """
    Generate a comprehensive diagnostic report for a brain tumor case
    """
    modalities = find_modalities(case_folder)
    case_name = Path(case_folder).name
    
    # Load segmentation data
    if 'seg' not in modalities:
        print("No segmentation file found!")
        return None
    
    seg_data = load_nifti(modalities['seg'])
    
    # Calculate tumor volumes (in voxels and approximate mm³)
    # BraTS images are typically 1mm³ per voxel
    voxel_volume_mm3 = 1.0  # Adjust based on actual voxel spacing
    
    # Tumor region volumes
    necrotic_voxels = np.sum(seg_data == 1)
    edema_voxels = np.sum(seg_data == 2)
    enhancing_voxels = np.sum(seg_data == 3) + np.sum(seg_data == 4)
    
    total_tumor_voxels = necrotic_voxels + edema_voxels + enhancing_voxels
    
    # Find tumor location (centroid)
    tumor_mask = seg_data > 0
    if tumor_mask.sum() > 0:
        coords = np.array(np.where(tumor_mask))
        centroid = coords.mean(axis=1)
        
        # Determine brain region based on centroid
        x, y, z = centroid
        region = determine_brain_region(x, y, z, seg_data.shape)
    else:
        centroid = np.array([0, 0, 0])
        region = "Unknown"
    
    # Calculate tumor extent (bounding box)
    if tumor_mask.sum() > 0:
        coords = np.array(np.where(tumor_mask))
        min_coords = coords.min(axis=1)
        max_coords = coords.max(axis=1)
        extent = max_coords - min_coords
    else:
        extent = np.array([0, 0, 0])
    
    # Generate report
    report = {
        'case_id': case_name,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tumor_type': 'Glioma' if '95_Glioma' in str(case_folder) else 'Other Neoplasm',
        'location': {
            'brain_region': region,
            'centroid_voxel': tuple(centroid.astype(int)),
            'extent_mm': tuple(extent)
        },
        'volumes': {
            'necrotic_core_mm3': necrotic_voxels * voxel_volume_mm3,
            'peritumoral_edema_mm3': edema_voxels * voxel_volume_mm3,
            'enhancing_tumor_mm3': enhancing_voxels * voxel_volume_mm3,
            'total_tumor_mm3': total_tumor_voxels * voxel_volume_mm3
        },
        'available_modalities': list(modalities.keys())
    }
    
    return report

def print_diagnostic_report(report):
    """
    Print formatted diagnostic report
    """
    if report is None:
        return None
    
    print("=" * 60)
    print("        BRAIN TUMOR DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"\nCase ID: {report['case_id']}")
    print(f"Report Date: {report['date']}")
    print(f"Tumor Classification: {report['tumor_type']}")
    
    print("\n" + "-" * 40)
    print("TUMOR LOCATION")
    print("-" * 40)
    print(f"Brain Region: {report['location']['brain_region']}")
    print(f"Centroid (voxel): {report['location']['centroid_voxel']}")
    print(f"Tumor Extent (mm): {report['location']['extent_mm']}")
    
    print("\n" + "-" * 40)
    print("TUMOR VOLUME ANALYSIS")
    print("-" * 40)
    vol = report['volumes']
    print(f"Necrotic Core:       {vol['necrotic_core_mm3']:,.0f} mm³ ({vol['necrotic_core_mm3']/1000:.2f} cm³)")
    print(f"Peritumoral Edema:   {vol['peritumoral_edema_mm3']:,.0f} mm³ ({vol['peritumoral_edema_mm3']/1000:.2f} cm³)")
    print(f"Enhancing Tumor:     {vol['enhancing_tumor_mm3']:,.0f} mm³ ({vol['enhancing_tumor_mm3']/1000:.2f} cm³)")
    print(f"TOTAL TUMOR VOLUME:  {vol['total_tumor_mm3']:,.0f} mm³ ({vol['total_tumor_mm3']/1000:.2f} cm³)")
    
    print("\n" + "-" * 40)
    print("AVAILABLE IMAGING MODALITIES")
    print("-" * 40)
    print(", ".join([m.upper() for m in report['available_modalities']]))
    
    # Clinical interpretation
    print("\n" + "-" * 40)
    print("CLINICAL NOTES")
    print("-" * 40)
    
    total_vol = vol['total_tumor_mm3']
    if total_vol > 50000:
        size_assessment = "Large tumor (>50 cm³) - significant mass effect likely"
    elif total_vol > 20000:
        size_assessment = "Medium-sized tumor (20-50 cm³)"
    else:
        size_assessment = "Small tumor (<20 cm³)"
    
    print(f"Size Assessment: {size_assessment}")
    
    # Enhancing ratio
    if vol['total_tumor_mm3'] > 0:
        enhancing_ratio = vol['enhancing_tumor_mm3'] / vol['total_tumor_mm3'] * 100
        print(f"Enhancing Component: {enhancing_ratio:.1f}% of total tumor")
        
        if enhancing_ratio > 50:
            print("⚠️  High enhancement suggests aggressive tumor behavior")
        elif enhancing_ratio < 10:
            print("ℹ️  Low enhancement - may indicate lower grade or treatment response")
    
    print("\n" + "=" * 60)
    print("Note: This report is for research purposes only.")
    print("Clinical decisions require radiologist review.")
    print("=" * 60)
    
    return report

# Generate report for first glioma case
if glioma_cases:
    print("\n📋 GENERATING DIAGNOSTIC REPORT...\n")
    report = generate_diagnostic_report(glioma_cases[0])
    print_diagnostic_report(report)

# ============================================================
# CELL 7: Batch Analysis - Analyze All Cases
# ============================================================

def analyze_all_cases(cases_list, tumor_type):
    """
    Analyze all cases and generate summary statistics
    """
    results = []
    
    for case_folder in cases_list:
        try:
            report = generate_diagnostic_report(case_folder)
            if report:
                report['tumor_type'] = tumor_type
                results.append(report)
        except Exception as e:
            print(f"Error processing {case_folder.name}: {e}")
    
    return results

def print_summary_statistics(results):
    """
    Print summary statistics for analyzed cases
    """
    if not results:
        print("No results to analyze")
        return
    
    print("\n" + "=" * 60)
    print("        DATASET SUMMARY STATISTICS")
    print("=" * 60)
    
    # Extract volumes
    total_volumes = [r['volumes']['total_tumor_mm3'] for r in results]
    necrotic_volumes = [r['volumes']['necrotic_core_mm3'] for r in results]
    edema_volumes = [r['volumes']['peritumoral_edema_mm3'] for r in results]
    enhancing_volumes = [r['volumes']['enhancing_tumor_mm3'] for r in results]
    
    print(f"\nTotal Cases Analyzed: {len(results)}")
    print(f"\nTumor Volume Statistics (mm³):")
    print(f"  Total Tumor Volume:")
    print(f"    Mean: {np.mean(total_volumes):,.0f} mm³")
    print(f"    Std:  {np.std(total_volumes):,.0f} mm³")
    print(f"    Min:  {np.min(total_volumes):,.0f} mm³")
    print(f"    Max:  {np.max(total_volumes):,.0f} mm³")
    
    print(f"\n  Necrotic Core:")
    print(f"    Mean: {np.mean(necrotic_volumes):,.0f} mm³")
    
    print(f"\n  Peritumoral Edema:")
    print(f"    Mean: {np.mean(edema_volumes):,.0f} mm³")
    
    print(f"\n  Enhancing Tumor:")
    print(f"    Mean: {np.mean(enhancing_volumes):,.0f} mm³")
    
    # Brain region distribution
    regions = [r['location']['brain_region'] for r in results]
    region_counts = {}
    for region in regions:
        region_counts[region] = region_counts.get(region, 0) + 1
    
    print(f"\nBrain Region Distribution:")
    for region, count in sorted(region_counts.items(), key=lambda x: -x[1]):
        print(f"  {region}: {count} cases ({count/len(results)*100:.1f}%)")

# Uncomment to run batch analysis (may take time)
# print("\nAnalyzing Glioma cases...")
# glioma_results = analyze_all_cases(glioma_cases[:5], "Glioma")  # First 5 cases
# print_summary_statistics(glioma_results)

# ============================================================
# CELL 8: Volume Comparison Visualization
# ============================================================

def plot_volume_comparison(results):
    """
    Create volume comparison visualizations
    """
    if not results:
        print("No results to plot")
        return
    
    case_ids = [r['case_id'][-7:] for r in results]  # Shortened case IDs
    necrotic = [r['volumes']['necrotic_core_mm3']/1000 for r in results]
    edema = [r['volumes']['peritumoral_edema_mm3']/1000 for r in results]
    enhancing = [r['volumes']['enhancing_tumor_mm3']/1000 for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Stacked bar chart
    ax1 = axes[0]
    x = np.arange(len(case_ids))
    width = 0.8
    
    ax1.bar(x, necrotic, width, label='Necrotic Core', color='red', alpha=0.8)
    ax1.bar(x, edema, width, bottom=necrotic, label='Edema', color='green', alpha=0.8)
    ax1.bar(x, enhancing, width, bottom=np.array(necrotic)+np.array(edema), 
            label='Enhancing', color='blue', alpha=0.8)
    
    ax1.set_xlabel('Case ID')
    ax1.set_ylabel('Volume (cm³)')
    ax1.set_title('Tumor Volume Components by Case')
    ax1.set_xticks(x)
    ax1.set_xticklabels(case_ids, rotation=45, ha='right')
    ax1.legend()
    
    # Pie chart for average composition
    ax2 = axes[1]
    avg_volumes = [np.mean(necrotic), np.mean(edema), np.mean(enhancing)]
    labels = ['Necrotic Core', 'Peritumoral Edema', 'Enhancing Tumor']
    colors = ['red', 'green', 'blue']
    
    ax2.pie(avg_volumes, labels=labels, colors=colors, autopct='%1.1f%%',
            explode=(0.05, 0.05, 0.05), shadow=True)
    ax2.set_title('Average Tumor Composition')
    
    plt.tight_layout()
    plt.savefig('tumor_volume_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: tumor_volume_comparison.png")

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("BraTS-Africa Brain Tumor Analysis - Complete")
    print("="*60)
    
    # Quick analysis of first few cases
    if glioma_cases and len(glioma_cases) >= 3:
        print("\n🔬 Quick Analysis of First 3 Glioma Cases:")
        quick_results = analyze_all_cases(glioma_cases[:3], "Glioma")
        print_summary_statistics(quick_results)
        
        # Create volume comparison
        if quick_results:
            plot_volume_comparison(quick_results)
    
    print("\n✅ Analysis complete!")
    print("Files created:")
    print("  - tumor_volume_comparison.png (if cases were analyzed)")
