import napari
import nibabel as nib
import numpy as np
import os
import csv
import pandas as pd

# Build path
def scan_datasets(folder_path, output_csv):
    results = []

    # Read *.nii or *.nii.gz
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                path = os.path.join(root, file)
                print(f'Processing {path}')
                try:
                    # Load in current image 
                    img = nib.load(path)
                    # Voxel size in x-, y-, z-direction
                    voxel_size = img.header.get_zooms()
                    # Number of voxels
                    shape = img.shape
                    # Coordinate system
                    Coord_system = nib.aff2axcodes(img.affine)
                    
                    # Append results
                    results.append({
                        "dataset": file,
                        "path": path,
                        "voxel_size": voxel_size,
                        "shape": shape,
                        "Coord_system": Coord_system,
                    })

                except Exception as e:
                    print(f"Error reading {file}: {e}")

    # Write CSV
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to: {output_csv}")

if __name__ == "__main__":
    
    import argparse

    # Instantiate parser
    parser = argparse.ArgumentParser(
        #description="Evaluate voxel resolution of NIfTI datasets"
    )
    # Input folder
    parser.add_argument("input_folder", type=str,
                        #help="Folder with .nii / .nii.gz files"
                       )
    # Output folder
    parser.add_argument("--output", type=str, default="voxel_evaluation.csv",
                        #help="Output CSV file"
                       )

    args = parser.parse_args()

    # Scan dataset
    scan_datasets(args.input_folder, args.output)