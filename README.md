# DiffDock-VdW
DiffDock-VdW is a feature augmentation of DiffDock that updates the preprocessing algorithm to include VdW features at the atom-node level. We provide the code and some examples of how to make use of DiffDockHPC to leverage High Performance Computing clusters and slurm. Furthermore, DiffDock-VdW is a fork of [DiffDockHPC](https://github.com/Jnelen/DiffDockHPC) which provides users HPC DiffDock functionality through Singularity and Slurm. Further, DiffDockHPC is itself a fork of [DiffDock](https://github.com/gcorso/DiffDock).

## The Organization of the Information Below
1. Requirements
2. Running DiffDock-VdW using our VdW models
3. Reproducing our procedure (Data preprocessing with HiQBind -> Model Training -> Model Validation)
4. Examples of srun commands (inference and train/evaluation)

## 1. Requirements
### Software Requirements:
* Singularity 
* Slurm* (Though not required, our examples will make use of it)

### Hardware Requirements
* **Inference:** Run-time execution is lightweight and compatible with standard modern GPUs.
* **Training Replication:** Reproducing the full ablation study requires a minimum of **80GB VRAM** (e.g., NVIDIA A100 80GB).

## 2. Running our DiffDock-VdW models
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

<!--
   Optionally, if you intend on running many jobs in quick succession (i.e. debugging or preliminary jobs), you can sandbox the singularity image (There are examples of jobs with using both the sif and the sandbox in the examples section below) :
   ```
   singularity build --sandbox singularity/DiffDockHPC DiffDockHPC.sif
   ```
-->

3. Note: When training the models (or running inference without inferenceVS), you must bind the batchnorm fix from the mye3nn folder to the singularity image. The srun examples below demonstrate this fix in greater detail.

4. Download and unzip our model weights: [Download weights](https://github.com/Kamal-R-Albousafi/DiffDockVdW/releases/download/v1.1.3/diffdockvdw_models.tar.gz)

   You can also download them using wget to place them directly onto your compute cluster:
   ```
   wget https://github.com/Kamal-R-Albousafi/DiffDockVdW/releases/download/v1.1.3/diffdockvdw_models.tar.gz
   ```
5. You are now ready to run inference. Placing the confidence_model and score_model folders into your DiffDockVdW directory will allow inference to be run with the following command (within a sbatch file):

```
# Example inference command
cd DiffDockVdW
SIF=singularity/DiffDockHPC.sif
BIND_DIR=$PWD
INTERNAL_PATH="/opt/conda/envs/DiffDockHPC/lib/python3.9/site-packages/e3nn/nn/_batchnorm.py"
srun singularity run --nv \
    --bind $BIND_DIR:$BIND_DIR,/scratch:/scratch\
    --bind $PWD/mye3nn/fixed_batchnorm.py:$INTERNAL_PATH \
    $SIF \
    python inference.py \
        --protein_ligand_csv data/your_protein_ligand.csv \
        --model_dir score_model \
        --ckpt best_ema_inference_epoch_model.pt \
        --confidence_model_dir confidence_model \
        --confidence_ckpt best_model_epoch75.pt \
        --out_dir $RES_DIR/activators \
        --samples_per_complex 20 \
        --inference_steps 20 \
        --actual_steps 19 \
        --batch_size 20
```
Furthermore, the additional inference options can be found around line 60 in inference.py. As an additional note, if you are running inference with a model you trained with a different combination of vdw features, model_parameters.yml will track which ones you used; therefore, no vdw flags need to be used to run inference.py.

## 3. Reproducing our Results
### Contents
1. Obtaining the HiQBind-corrected PDBBindv2020 dataset
2. Training the score model
3. Training the confidence model
4. Evaluation

### Obtaining the HiQBind-corrected PDBBindv2020 dataset
1. Obtain the subsetted INDEX_general_PLSubset.2020 file which we provide as helper_files/INDEX_filtered_no_overlap.2020 
2. Perform steps 2a and 3 (under the ''Alternatively, for processing PDBBind, use these codes instead'') from [HiQBind's How to reconstruct HiQBind and Optimized PDBBind Section](https://github.com/THGLab/HiQBind#how-to-reconstruct-hiqbind-and-optimized-pdbbind), making sure to replace ''INDEX_general_PLSubset.2020'' with ''INDEX_filtered_no_overlap.2020''
3.  We provide a helper file helper_files/hiqbind_for_diffdock.py that will help convert the output of these steps into a DiffDock-ready format.
<!-- INDEX_general_PLSubset.2020 file from the official PDBBind website and subset it so that it only includes the complexes from data/splits/timesplit_no_lig_overlap_train, data/splits/timesplit_no_lig_overlap_val, and data/splits/timesplit_test -->
### Training the score model
There are a lot of fine details here, but below are a few key details that should ease the process.
1. utils/parsing.py includes all of the training arguments. While many of these are not too important, a couple of them are vital and easy to miss. Here a few notable ones:
    a. `--vdw_base`, `--vdw_curv`, `--vdw_vol`: These are store_true flags that tell the model which vdw features to use. Any combination may be used, including 0 of them
    b. `--dropout`: This is the neuron dropout that prevents overfitting; its default is 0, but it is highly recommended to use a value of at least 0.1
    c. Continue list

### Training the confidence model
Similar to the score model, there are a lot of key training arguments that can easily be glossed over.
1. The train arguments for training the confidence model can be found at line 25 of confidence_train.py
2. Key notes: the confidence model can actually use an entirely different set of configurations and number of vdw features than the trained score model. The only time you will get errors in this manner is if you make a change to how data should be pre-processed but still used the old data cache. Furthermore, the flags `--vdw_base`, `--vdw_curv`, `--vdw_vol` are once again present to allow for any combination of vdw features
3. Once again, confidence_dropout has a default of 0.0 and is highly recommended to be at least 0.1


## License
MIT
