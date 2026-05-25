"""
Helper script to create a lightweight ZIP archive of the thesis project
for easy transfer (e.g., via AnyDesk).

This script excludes the heavy folders:
- data/NWPU-Crowd/density_maps (~122 GB)
- data/NWPU-Crowd/images_part1-5 (~20 GB)
- Intermediate checkpoints (saves only the 'best' checkpoints and the latest MobileCount epoch)

Run it with:
    python create_transfer_zip.py
"""

import os
import zipfile
import re

def get_latest_checkpoint(checkpoints_dir):
    """Find the latest mobilecount_epoch_*.pth checkpoint to allow resuming."""
    if not os.path.exists(checkpoints_dir):
        return None
    
    max_epoch = -1
    latest_file = None
    
    pattern = re.compile(r'mobilecount_epoch_(\d+)\.pth')
    for filename in os.listdir(checkpoints_dir):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_file = filename
                
    return latest_file

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(project_dir, "thesis_transfer.zip")
    checkpoints_dir = os.path.join(project_dir, "checkpoints")
    
    # Identify the latest checkpoint to include
    latest_mc_checkpoint = get_latest_checkpoint(checkpoints_dir)
    print(f"Project directory: {project_dir}")
    print(f"Target zip path:   {zip_path}")
    if latest_mc_checkpoint:
        print(f"Found latest MobileCount epoch checkpoint to include: {latest_mc_checkpoint}")
    else:
        print("No MobileCount epoch checkpoints found.")
        
    print("\nCreating zip archive...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(project_dir):
            # Exclude directories
            # 1. Skip virtual environment, git, and vscode folders
            if any(part in root.split(os.sep) for part in ['.git', '.venv', '.vscode', '__pycache__']):
                continue
                
            # 2. Strict exclusion for data folder
            parts = root.split(os.sep)
            if 'data' in parts:
                idx = parts.index('data')
                rel_data_parts = parts[idx:]
                # Only keep:
                # - data
                # - data/NWPU-Crowd
                # - data/NWPU-Crowd/mats
                # - data/NWPU-Crowd/splits
                if len(rel_data_parts) > 1:
                    if rel_data_parts[1] != 'NWPU-Crowd':
                        continue # Excludes data/density_maps, data/raw, data/ShanghaiTech
                    if len(rel_data_parts) > 2:
                        if rel_data_parts[2] not in ['mats', 'splits']:
                            continue # Excludes data/NWPU-Crowd/images_part1-5, data/NWPU-Crowd/density_maps
                        
            # Walk and write files
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_dir)
                
                # Check for the ZIP file itself to avoid self-reference recursion
                if file == "thesis_transfer.zip":
                    continue
                    
                # Exclude checkpoints folder intermediate files
                if 'checkpoints' in parts:
                    # Only include:
                    # - File names containing "best"
                    # - The latest mobilecount_epoch_*.pth file
                    # - Any other file that isn't an intermediate epoch checkpoint
                    is_best = "best" in file.lower()
                    is_latest = (latest_mc_checkpoint and file == latest_mc_checkpoint)
                    is_epoch_file = ("epoch_" in file.lower() and file.endswith(".pth"))
                    
                    if is_epoch_file and not is_latest:
                        continue # Skip intermediate checkpoints
                
                # Write file to zip
                zip_file.write(file_path, rel_path)
                
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nSuccess! Archive created at: {zip_path}")
    print(f"Total archive size: {zip_size_mb:.2f} MB")
    print("\nThis zip contains:")
    print(" - All python source code & folder structures")
    print(" - Core best weights (best_model, best_model_nwpu_sparse, mobilecount_best)")
    print(f" - Latest training epoch for resuming ({latest_mc_checkpoint if latest_mc_checkpoint else 'None'})")
    print(" - NWPU-Crowd annotations (.mat files) & splits (excluding raw images & density maps)")

if __name__ == "__main__":
    main()
