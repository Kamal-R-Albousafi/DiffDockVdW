import pandas as pd
import shutil
import os
import subprocess

# Configuration
csv_file = 'VS_DD_inactive1_2026_1_21/molecules/AA_csvs/top_confidences.csv'  # Path to your CSV
source_dir = 'VS_DD_inactive1_2026_1_21/molecules/'      # Where the CID folders currently live
target_container = 'VS_DD_inactive1_2026_1_21/Conf_Decoys1'

# If u want them to be sorted by molecule, put = True; otherwise do false if u need to click them quickly (Chimera)
organize = False

# Create ONLY the main container
os.makedirs(target_container, exist_ok=True)

df = pd.read_csv(csv_file)
decoys = df[df['above-1.0'] == True]['CID'].tolist()

for cid in decoys:
    source_path = os.path.join(source_dir, cid)
    if organize:
        destination_path_sdf = os.path.join(target_container, "sdf", cid)
        destination_path_mol2 = os.path.join(target_container, "mol2", cid)
    else:
        destination_path_sdf = os.path.join(target_container, "sdf")
        destination_path_mol2 = os.path.join(target_container, "mol2")
    
    if os.path.isdir(source_path):
        # 1. Create mol2 inside the SOURCE folder first
        # mol2_path = os.path.join(source_path, "mol2")
        # os.makedirs(mol2_path, exist_ok=True)
        os.makedirs(destination_path_sdf, exist_ok=True)
        os.makedirs(destination_path_mol2, exist_ok=True)
        
        # 2. Run obabel conversion in the source
        for file in os.listdir(source_path):
            if file.endswith(".sdf"):
                input_file = os.path.join(source_path, file)
                # if not organize_mol2:
                #     mol2_path = os.path.join(source_dir, "mol2")
                output_file = os.path.join(destination_path_mol2, file.replace(".sdf", ".mol2"))
                subprocess.run(["obabel", input_file, "-O", output_file], check=True)

        # 3. Copy the entire folder (including the new mol2 subfolder)
        # print(f"Copying {cid} to {target_container}...")
        
        # shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        shutil.copytree(source_path, destination_path_sdf, dirs_exist_ok=True)
    else:
        print(f"Warning: {cid} not found.")