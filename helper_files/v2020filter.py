from pathlib import Path

# 0. File names -- replace these with your file locations if applicable
keep_list_file = Path('helper_files/timesplit_no_overlap_master') 
original_index = Path('helper_files/INDEX_general_PL.2020') # NOTE: this file is not provided, it must be downloaded from the official PDBBind-plus website 
output_index = Path('helper_files/INDEX_filtered_no_overlap.2020')

# 1. Load IDs to keep
print("Loading the no overlap split ids...")
with open(keep_list_file, 'r') as f:
    keep_ids = {line.strip().lower() for line in f if line.strip()}

print(f"{len(keep_ids)} complexes identified.")

# 2. Process the index file, retaining only the rows that have a PDB ID in the no ligand overlap splits
filtered_count = 0
with open(original_index, 'r') as f_in, open(output_index, 'w') as f_out:
    for line in f_in:
        # Preserve header comments (#) and empty lines
        if line.startswith('#') or not line.strip():
            f_out.write(line)
            continue
        
        # PDB ID is in first column
        parts = line.split()
        if not parts:
            continue
        # Select the pdb_id and send it to lowercase, matching the structure provided by Corso et al.
        pdb_id = parts[0].lower()
        
        # Check if this PDB ID exists in our keep set
        if pdb_id in keep_ids:
            f_out.write(line)
            filtered_count += 1

print(f"Filtering complete.")
print(f"New index created with {filtered_count} complexes.")