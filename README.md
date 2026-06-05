# DiffDock-VdW
DiffDock-VdW is a feature augmentation of DiffDock that updates the preprocessing algorithm to include VdW features at the atom-node level. We provide the code and some examples of how to make use of DiffDockHPC to leverage High Performance Computing clusters and slurm. Furthermore, DiffDock-VdW is a fork of [DiffDockHPC](https://github.com/Jnelen/DiffDockHPC) which provides users HPC DiffDock functionality through Singularity and Slurm. Further, DiffDockHPC is itself a fork of [DiffDock](https://github.com/gcorso/DiffDock).

## The Organization of the Information Below
1. Requirements
2. Running DiffDock-VdW using our VdW models
3. Reproducing our procedure (Data preprocessing with HiQBind -> Model Training -> Model Validation)
4. Examples of srun commands (inference and train/evaluation)

## Requirements
### Software Requirements:
* Singularity 
* Slurm (There is a --no_slurm mode, but using Slurm is highly recommended)

### Hardware Requirements
* **Inference:** Run-time execution is lightweight and compatible with standard modern GPUs.
* **Training Replication:** Reproducing the full ablation study requires a minimum of **80GB VRAM** (e.g., NVIDIA A100 80GB).

## Running our DiffDock-VdW models
### Installation instructions:
1. Clone the repository and navigate to it
    ```
    git clone https://github.com/Kamal-R-Albousafi/DiffDockVdW
    ```
   ```
   cd DiffDockVdW
   ```
   
2. **Install the Singularity image**. There are multiple methods of doing this. From [DiffDockHPC](https://github.com/Jnelen/DiffDockHPC): Run a test example to automatically download the Singularity image (~3 GB) and to generate the necessary cache look-up tables for SO(2) and SO(3) distributions. (This only needs to happen once and usually takes around 15 minutes).  
   The `--no_slurm` flag is optional here, but makes it easier to track the progress.   
   ```
   python inferenceVS.py -p data/1a0q/1a0q_protein_processed.pdb -l data/1a0q/ -out TEST -j 1 --no_slurm
   ```  
   Or if you have access to a GPU, you can also add the -gpu tag like this:  
   ```
   python inferenceVS.py -p data/1a0q/1a0q_protein_processed.pdb -l data/1a0q/ -out TEST -j 1 -gpu --no_slurm
   ```  
[DiffDockHPC](https://github.com/Jnelen/DiffDockHPC) additionally provides a method to manually download the Singularity image:
   ```
   wget --no-check-certificate -r "https://drive.usercontent.google.com/download?id=1TsbuhNWA74AHfIbKV5uh2lmEnD99VlCD&confirm=t" -O singularity/DiffDockHPC.sif
   ```
   
   Likewise, you can build the singularity image yourself using:
   ```
   singularity build singularity/DiffDockHPC.sif singularity/DiffDockHPC.def
   ```

   Optionally, if you intend on running many jobs in quick succession (i.e. debugging or preliminary jobs), you can sandbox the singularity image (There are examples of jobs with using both the sif and the sandbox in the examples section below) :
   ```
   singularity build --sandbox singularity/DiffDockHPC DiffDockHPC.sif
   ```

3. Note: When training the models, you must bind the batchnorm fix from the mye3nn folder to the singularity image. The srun examples below demonstrate this fix in greater detail.

4. Download and unzip our models 


### Options

The main file to use is `inferenceVS.py`. It has the following options/flags:  

- `-p`, `-r`, `--protein_path`: 
  Path to the protein/receptor `.pdb` file.

- `-l`, `--ligand`: 
  The path to the directory of (separate) `mol2`/`sdf` ligand files.

- `--protein_ligand_csv`: 
  The path to a protein_ligand_csv file. Format and header should be like the following: complex_name,protein_path,ligand_description.
  
- `-o`, `--out`, `--out_dir`: 
  Directory where the output structures will be saved to.

- `-j`, `--jobs`: 
  Number of jobs to use.

- `-qu`, `--queue`: 
  On which node to launch the slurm jobs. The default value is the default queue for the user. Might need to be specified if there is no default queue configured.

- `-m`, `--mem`: 
  How much memory to use for each job. The default value is `4GB`.

- `-gpu`, `--gpu`: 
  Use GPU resources. This will accelerate docking calculations if a compatible GPU is available.

- `-c`, `--cores`: 
  How many cores to use for each job. The default value is `1` when used with the GPU option enabled, otherwise it defaults to `4` cores.

- `-n`, `--num_outputs`: 
  How many structures to output per compound. The default value is `1`.

- `--remove_hs`: 
  Remove the hydrogens in the final output structures.
  
- `--no_slurm`: 
  Don't use slurm to handle the resources. This will run all samples on 1 GPU. Other Slurm arguments such as the amount memory, time limit, ... will also be ignored. The amount of CPU cores will still be set.

- `--config`: 
  Path to the config file you want to use. Defaults to `default_inference_args.yaml`

- `-h`, `--help`: 
  Show the help message and exit.

## License
MIT
