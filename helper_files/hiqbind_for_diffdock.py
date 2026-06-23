import os
import json
import shutil
from pathlib import Path

# Placeholder paths
raw_data_path = Path('/path/to/your/raw_data').resolve() # path to your HIQBind complexes
refined_base_path = Path('/path/you/want/your/refined_data').resolve() # desired output location

# This can be adjusted to be finer or less restrictive
resolution_threshold = 2.5


subsets = ['pdbbind_opt_poly', 'pdbbind_opt_sm']

# Function to obtain resolution of complex
def get_resolution(json_path):
    if not json_path.exists():
        return None
    try:
        with open(json_path, 'r') as f:
            full_data = json.load(f)
        refine_list = full_data.get('data', {}).get('entry', {}).get('refine', [])
        if refine_list and len(refine_list) > 0:
            res = refine_list[0].get('ls_d_res_high')
            return float(res) if res is not None else None
        return None
    except Exception:
        return None


# Main function
refined_base_path.mkdir(parents=True, exist_ok=True)
for subset in subsets:
    subset_full_path = raw_data_path / subset
    if not subset_full_path.exists():
        print(f"Skipping {subset}: Path not found.")
        continue
    
    print(f"Processing Subset: {subset} --------")
    
    # Iterate through PDB ID folders
    for complex_parent in subset_full_path.iterdir():
        if not complex_parent.is_dir():
            continue
        pdb_id = complex_parent.name[:4]

        # 1. Immediate skips (done or err tags in each folder, we skip the err complexes)
        if (complex_parent / 'err').exists():
            continue
            
        # 2. Resolution check (at parent level)
        json_path = complex_parent / 'rcsb_data.json'
        resolution = get_resolution(json_path)
        
        if resolution is None or resolution > resolution_threshold:
            continue
        
        # 3. Obtain refined data from clean complexes
        pdb_id = complex_parent.name[:4]
        target_dir = refined_base_path / pdb_id
        
        found_data = False
        # Many complexes will have multiple ligands. We only use one ligand per complex. If you desire to use every ligand available,
        # simply remove the break at line 77 below
        for sub_item in complex_parent.iterdir():
            if sub_item.is_dir() and sub_item.name.startswith(pdb_id):
                prot = list(sub_item.glob('*protein_refined.pdb'))
                lig = list(sub_item.glob('*ligand_refined.sdf'))
                
                if prot and lig:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(prot[0], target_dir / f"{pdb_id}_protein_processed.pdb")
                    shutil.copy2(lig[0], target_dir / f"{pdb_id}_ligand.sdf")
                    found_data = True
                    break # Delete this line if you want more than 1 ligand per complex
        
        if found_data:
            print(f"Exported {pdb_id} (Res: {resolution})")

print(f"\nRefinement complete. Cleaned data is in: {refined_base_path}")