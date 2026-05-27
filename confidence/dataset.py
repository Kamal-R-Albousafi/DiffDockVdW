import itertools
import math
import os
import pickle
import random
from argparse import Namespace
from functools import partial
import copy

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets.pdbbind import PDBBind
from utils.diffusion_utils import get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model, read_strings_from_txt
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.molecules_utils import get_symmetry_rmsd
from rdkit.Chem import RemoveAllHs
import sys
sys.stdout.reconfigure(line_buffering=True)

class ListDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

def get_cache_path(args, split):
    cache_path = args.cache_path
    if not args.no_torsion:
        cache_path += '_torsion'
    if args.all_atoms:
        cache_path += '_allatoms'
    args.esm_embeddings_path = None # for some reason the yaml file did not save this
    split_path = args.split_train if split == 'train' else args.split_val
    # KRA Edited: resolved bug--cache paths have a leading dataset identifier and a number (from datasets.pdbbind.py)
    dataset_name = ''
    if args.dataset == 'pdbbind': 
        dataset_name = 'PDBBind3'
        # Hard coded: # TODO fix
        protein_file = 'protein_processed'
        protein_path_list = ligand_descriptions = None
        matching_tries = None
        keep_local_structures = False
        fixed_knn_radius_graph = True
        knn_only_graph = True
        use_old_wrong_embedding_order = False
        cache_path = os.path.join(cache_path, f'{dataset_name}_limit{args.limit_complexes}'
                                                        f'_INDEX{os.path.splitext(os.path.basename(split_path))[0]}'
                                                        f'_maxLigSize{args.max_lig_size}_H{int(not args.remove_hs)}'
                                                        f'_recRad{args.receptor_radius}_recMax{args.c_alpha_max_neighbors}'
                                                        f'_chainCutoff{args.chain_cutoff if args.chain_cutoff is None else int(args.chain_cutoff)}'
                                            + ('' if not args.all_atoms else f'_atomRad{args.atom_radius}_atomMax{args.atom_max_neighbors}')
                                            + ('' if args.no_torsion or args.num_conformers == 1 else f'_confs{args.num_conformers}')
                                            + ('' if args.esm_embeddings_path is None else f'_esmEmbeddings')
                                            + '_full'
                                            + ('' if not keep_local_structures else f'_keptLocalStruct')
                                            + ('' if protein_path_list is None or ligand_descriptions is None else str(binascii.crc32(''.join(ligand_descriptions + protein_path_list).encode())))
                                            + ('' if protein_file == "protein_processed" else '_' + protein_file)
                                            + ('' if not fixed_knn_radius_graph else (f'_fixedKNN' if not knn_only_graph else '_fixedKNNonly'))
                                            + ('' if not args.include_miscellaneous_atoms else '_miscAtoms')
                                            + ('' if use_old_wrong_embedding_order else '_chainOrd')
                                            + ('' if args.matching_tries == 1 else f'_tries{matching_tries}')
                                            + ('' if not args.vdw_base else '_vdwbase')
                                            + ('' if not args.vdw_curv else '_vdwcurv')
                                            + ('' if not args.vdw_vol else '_vdwvol'))
    # else if args.dataset == 'moad': dataset_name = 'MOAD12':
    #     # TODO
    else:
         cache_path = os.path.join(cache_path, f'{dataset_name}_limit{args.limit_complexes}_INDEX{os.path.splitext(os.path.basename(split_path))[0]}_maxLigSize{args.max_lig_size}_H{int(not args.remove_hs)}_recRad{args.receptor_radius}_recMax{args.c_alpha_max_neighbors}'
                                       + ('' if not args.all_atoms else f'_atomRad{args.atom_radius}_atomMax{args.atom_max_neighbors}')
                                       + ('' if args.no_torsion or args.num_conformers == 1 else
                                           f'_confs{args.num_conformers}')
                              + ('' if args.esm_embeddings_path is None else f'_esmEmbeddings'))
    
    return cache_path

def get_args_and_cache_path(original_model_dir, split):
    with open(f'{original_model_dir}/model_parameters.yml') as f:
        model_args = Namespace(**yaml.full_load(f))
    return model_args, get_cache_path(model_args,split)



class ConfidenceDataset(Dataset):
    def __init__(self, cache_path, original_model_dir, split, device, limit_complexes,
                 inference_steps, samples_per_complex, all_atoms,
                 args, model_ckpt, balance=False, use_original_model_cache=True, rmsd_classification_cutoff=2,
                 cache_ids_to_combine=None, cache_creation_id=None):

        super(ConfidenceDataset, self).__init__()

        self.device = device
        self.inference_steps = inference_steps
        self.limit_complexes = limit_complexes
        self.all_atoms = all_atoms
        self.original_model_dir = original_model_dir
        self.balance = balance
        self.use_original_model_cache = use_original_model_cache
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.cache_ids_to_combine = cache_ids_to_combine
        self.cache_creation_id = cache_creation_id
        self.samples_per_complex = samples_per_complex
        self.model_ckpt = model_ckpt
        # KRA: updated path logic for the cache loader
        if split == 'train':
            self.split_path = args.split_train
        elif split == 'val':
            self.split_path = args.split_val
        else:
            print('ERROR: no split detected, returning null for split path')
            self.split_path = ''

        self.original_model_args, original_model_cache = get_args_and_cache_path(original_model_dir, split)
        self.complex_graphs_cache = original_model_cache if self.use_original_model_cache else get_cache_path(args, split)

        # check if the docked positions have already been computed, if not run the preprocessing (docking every complex)
        self.full_cache_path = os.path.join(cache_path, f'model_{os.path.splitext(os.path.basename(original_model_dir))[0]}'
                                            f'_split_{split}_limit_{limit_complexes}')

        if (not os.path.exists(os.path.join(self.full_cache_path, "ligand_positions.pkl")) and self.cache_creation_id is None) or \
                (not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_id{self.cache_creation_id}.pkl")) and self.cache_creation_id is not None):
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.preprocessing(original_model_cache)

        # load the graphs that the confidence model will use
        print('Using the cached complex graphs of the original model args' if self.use_original_model_cache else 'Not using the cached complex graphs of the original model args. Instead the complex graphs are used that are at the location given by the dataset parameters given to confidence_train.py')
        print(self.complex_graphs_cache)
        if not os.path.exists(os.path.join(self.complex_graphs_cache, "heterographs.pkl")) and not os.path.exists(os.path.join(self.complex_graphs_cache, "heterographs0.pkl")):
            print(f'HAPPENING | Complex graphs path does not exist yet: {os.path.join(self.complex_graphs_cache, "heterographs.pkl")}. For that reason, we are now creating the dataset.')
            PDBBind(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes,
                    receptor_radius=args.receptor_radius,
                    cache_path=args.cache_path, split_path=args.split_val if split == 'val' else args.split_train,
                    remove_hs=args.remove_hs, max_lig_size=None,
                    c_alpha_max_neighbors=args.c_alpha_max_neighbors,
                    matching=not args.no_torsion, keep_original=True,
                    popsize=args.matching_popsize,
                    maxiter=args.matching_maxiter,
                    all_atoms=args.all_atoms,
                    atom_radius=args.atom_radius,
                    atom_max_neighbors=args.atom_max_neighbors,
                    esm_embeddings_path=args.esm_embeddings_path,
                    require_ligand=True)

        print(f'HAPPENING | Loading complex graphs from: {os.path.join(self.complex_graphs_cache, "heterographs.pkl")}')
        # KRA Edited: more logic to aid with heterographX.pkl files
        if os.path.exists(os.path.join(original_model_cache, f"heterographs.pkl")):
            with open(os.path.join(original_model_cache, "heterographs.pkl"), 'rb') as f:
                complex_graphs = pickle.load(f)
        else:
            complex_names_all = read_strings_from_txt(self.split_path)
            if self.limit_complexes is not None and self.limit_complexes != 0:
                complex_names_all = complex_names_all[:self.limit_complexes]
            complex_graphs_all = []
            for i in range(len(complex_names_all) // 1000 + 1):
                with open(os.path.join(original_model_cache, f"heterographs{i}.pkl"), 'rb') as f:
                    print(i)
                    l = pickle.load(f)
                    complex_graphs_all.extend(l)
            complex_graphs = complex_graphs_all
        self.complex_graph_dict = {d.name: d for d in complex_graphs}

        if self.cache_ids_to_combine is None:
            print(f'HAPPENING | Loading positions and rmsds from: {os.path.join(self.full_cache_path, "ligand_positions.pkl")}')
            with open(os.path.join(self.full_cache_path, "ligand_positions.pkl"), 'rb') as f:
                self.full_ligand_positions, self.rmsds = pickle.load(f)
            if os.path.exists(os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl")):
                with open(os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl"), 'rb') as f:
                    generated_rmsd_complex_names = pickle.load(f)
            else:
                print('HAPPENING | The path, ', os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl"),
                      ' does not exist. \n => We assume that means that we are using a ligand_positions.pkl where the '
                      'code was not saving the complex names for them yet. We now instead use the complex names of '
                      'the dataset that the original model used to create the ligand positions and RMSDs.')
                with open(os.path.join(original_model_cache, "heterographs.pkl"), 'rb') as f:
                    original_model_complex_graphs = pickle.load(f)
                    generated_rmsd_complex_names = [d.name for d in original_model_complex_graphs]
            assert (len(self.rmsds) == len(generated_rmsd_complex_names))
        else:
            all_rmsds_unsorted, all_full_ligand_positions_unsorted, all_names_unsorted = [], [], []
            for idx, cache_id in enumerate(self.cache_ids_to_combine):
                print(f'HAPPENING | Loading positions and rmsds from cache_id from the path: {os.path.join(self.full_cache_path, "ligand_positions_"+ str(cache_id)+ ".pkl")}')
                if not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl")): raise Exception(f'The generated ligand positions with cache_id do not exist: {cache_id}') # be careful with changing this error message since it is sometimes cought in a try catch
                with open(os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl"), 'rb') as f:
                    full_ligand_positions, rmsds = pickle.load(f)
                with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order_id{cache_id}.pkl"), 'rb') as f:
                    names_unsorted = pickle.load(f)
                all_names_unsorted.append(names_unsorted)
                all_rmsds_unsorted.append(rmsds)
                all_full_ligand_positions_unsorted.append(full_ligand_positions)
            names_order = list(set(sum(all_names_unsorted, [])))
            all_rmsds, all_full_ligand_positions, all_names = [], [], []
            for idx, (rmsds_unsorted, full_ligand_positions_unsorted, names_unsorted) in enumerate(zip(all_rmsds_unsorted,all_full_ligand_positions_unsorted, all_names_unsorted)):
                name_to_pos_dict = {name: (rmsd, pos) for name, rmsd, pos in zip(names_unsorted, full_ligand_positions_unsorted, rmsds_unsorted) }
                intermediate_rmsds = [name_to_pos_dict[name][1] for name in names_order]
                all_rmsds.append((intermediate_rmsds))
                intermediate_pos = [name_to_pos_dict[name][0] for name in names_order]
                all_full_ligand_positions.append((intermediate_pos))
            self.full_ligand_positions, self.rmsds = [], []
            for positions_tuple in list(zip(*all_full_ligand_positions)):
                self.full_ligand_positions.append(np.concatenate(positions_tuple, axis=0))
            for positions_tuple in list(zip(*all_rmsds)):
                self.rmsds.append(np.concatenate(positions_tuple, axis=0))
            generated_rmsd_complex_names = names_order
        print('Number of complex graphs: ', len(self.complex_graph_dict))
        print('Number of RMSDs and positions for the complex graphs: ', len(self.full_ligand_positions))

        self.all_samples_per_complex = samples_per_complex * (1 if self.cache_ids_to_combine is None else len(self.cache_ids_to_combine))

        self.positions_rmsds_dict = {name: (pos, rmsd) for name, pos, rmsd in zip (generated_rmsd_complex_names, self.full_ligand_positions, self.rmsds)}
        self.dataset_names = list(set(self.positions_rmsds_dict.keys()) & set(self.complex_graph_dict.keys()))
        if limit_complexes > 0:
            self.dataset_names = self.dataset_names[:limit_complexes]

    def len(self):
        return len(self.dataset_names)

    def get(self, idx):
        complex_graph = copy.deepcopy(self.complex_graph_dict[self.dataset_names[idx]])
        positions, rmsds = self.positions_rmsds_dict[self.dataset_names[idx]]

        if self.balance:
            if isinstance(self.rmsd_classification_cutoff, list): raise ValueError("a list for --rmsd_classification_cutoff can only be used without --balance")
            label = random.randint(0, 1)
            success = rmsds < self.rmsd_classification_cutoff
            n_success = np.count_nonzero(success)
            if label == 0 and n_success != self.all_samples_per_complex:
                # sample negative complex
                sample = random.randint(0, self.all_samples_per_complex - n_success - 1)
                lig_pos = positions[~success][sample]
                complex_graph['ligand'].pos = torch.from_numpy(lig_pos)
            else:
                # sample positive complex
                if n_success > 0: # if no successfull sample returns the matched complex
                    sample = random.randint(0, n_success - 1)
                    lig_pos = positions[success][sample]
                    complex_graph['ligand'].pos = torch.from_numpy(lig_pos)
            complex_graph.y = torch.tensor(label).float()
        else:
            # KRA edited: to follow the new logic of keeping high percentage successful samples
            sample = random.randint(0, len(positions) - 1)
            complex_graph['ligand'].pos = torch.from_numpy(positions[sample])
            complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff).float().unsqueeze(0)
            if isinstance(self.rmsd_classification_cutoff, list):
                complex_graph.y_binned = torch.tensor(np.logical_and(rmsds[sample] < self.rmsd_classification_cutoff + [math.inf],rmsds[sample] >= [0] + self.rmsd_classification_cutoff), dtype=torch.float).unsqueeze(0)
                complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff[0]).unsqueeze(0).float()
            complex_graph.rmsd = torch.tensor(rmsds[sample]).unsqueeze(0).float()

        complex_graph['ligand'].node_t = {'tr': 0 * torch.ones(complex_graph['ligand'].num_nodes),
                                          'rot': 0 * torch.ones(complex_graph['ligand'].num_nodes),
                                          'tor': 0 * torch.ones(complex_graph['ligand'].num_nodes)}
        complex_graph['receptor'].node_t = {'tr': 0 * torch.ones(complex_graph['receptor'].num_nodes),
                                            'rot': 0 * torch.ones(complex_graph['receptor'].num_nodes),
                                            'tor': 0 * torch.ones(complex_graph['receptor'].num_nodes)}
        if self.all_atoms:
            complex_graph['atom'].node_t = {'tr': 0 * torch.ones(complex_graph['atom'].num_nodes),
                                            'rot': 0 * torch.ones(complex_graph['atom'].num_nodes),
                                            'tor': 0 * torch.ones(complex_graph['atom'].num_nodes)}
        complex_graph.complex_t = {'tr': 0 * torch.ones(1), 'rot': 0 * torch.ones(1), 'tor': 0 * torch.ones(1)}
        return complex_graph

    def preprocessing(self, original_model_cache):
        t_to_sigma = partial(t_to_sigma_compl, args=self.original_model_args)

        model = get_model(self.original_model_args, self.device, t_to_sigma=t_to_sigma, no_parallel=True)
        state_dict = torch.load(f'{self.original_model_dir}/{self.model_ckpt}', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()
        
        # KRA - resolved bug: get_t_schedule missing sigma_schedule='expbeta' argument
        tr_schedule = get_t_schedule(sigma_schedule='expbeta', inference_steps=self.original_model_args.inference_steps,
                                inf_sched_alpha=1, inf_sched_beta=1)
        rot_schedule = tr_schedule
        tor_schedule = tr_schedule
        print('common t schedule', tr_schedule)

        # KRA - resolved bug: there may be indices on heterographs in ran in parallel during caching; edited version of collect_all_complexes copied over from datasets.pdbbind
        print('HAPPENING | loading cached complexes of the original model to create the confidence dataset RMSDs and predicted positions. Doing that from: ', os.path.join(self.complex_graphs_cache, "heterographs.pkl"))
        # Updated to have similar logic as datasets.pdbbind.py
        if os.path.exists(os.path.join(original_model_cache, f"heterographs.pkl")):
            with open(os.path.join(original_model_cache, "heterographs.pkl"), 'rb') as f:
                complex_graphs = pickle.load(f)
        else:
            complex_names_all = read_strings_from_txt(self.split_path)
            if self.limit_complexes is not None and self.limit_complexes != 0:
                complex_names_all = complex_names_all[:self.limit_complexes]
            complex_graphs_all, rdkit_ligands_all = [], []
            for i in range(len(complex_names_all) // 1000 + 1):
                with open(os.path.join(original_model_cache, f"heterographs{i}.pkl"), 'rb') as f:
                    print(i)
                    l = pickle.load(f)
                    complex_graphs_all.extend(l)
                with open(os.path.join(original_model_cache, f"rdkit_ligands{i}.pkl"), 'rb') as f:
                    rdkit_ligands_all.extend(pickle.load(f))
            complex_graphs = complex_graphs_all
        dataset = ListDataset(complex_graphs)
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        rmsds, min_rmsds, full_ligand_positions, names = [], [], [], []
        # KRA Edited: grab ligands as well, aligned with the loader
        for idx, (orig_complex_graph, rdkit_ligand) in tqdm(enumerate(zip(loader, rdkit_ligands_all))):
        # MAJOR KRA EDIT:
        # 1. New approach to sampling: if it fails to get mostly clean samples 10 times, discard the entire complex from dataset
        # 2. Compute symmetry corrected RMSD if available; the code defaulted to naive approach before
            predictions_list = None
            failed_convergence_counter = 0
            while predictions_list is None:
                try:
                    # This computes a new random position each fail--might help some samples find convergence
                    data_list = [copy.deepcopy(orig_complex_graph) for _ in range(self.samples_per_complex)]
                    randomize_position(data_list, self.original_model_args.no_torsion, False, self.original_model_args.tr_sigma_max)
                    predictions_list, confidences = sampling(data_list=data_list, model=model, inference_steps=self.inference_steps,
                                                             tr_schedule=tr_schedule, rot_schedule=rot_schedule, tor_schedule=tor_schedule,
                                                             device=self.device, t_to_sigma=t_to_sigma, model_args=self.original_model_args)
                    # KRA edited: check for bad samples via mask
                    ligand_pos_check = np.asarray([cg['ligand'].pos.cpu().numpy() for cg in predictions_list])
                    bad_mask = (
                        np.isnan(ligand_pos_check).any(axis=(1,2)) |
                        np.isinf(ligand_pos_check).any(axis=(1,2)) |
                        (np.abs(ligand_pos_check).max(axis=(1,2)) > 1000)
                    )
                    n_bad = bad_mask.sum()
                    if n_bad == self.samples_per_complex:
                        raise ValueError(f'{self.samples_per_complex} / {self.samples_per_complex} poses degenerate - retrying with new position')
                    elif 0 < n_bad and (n_bad / self.samples_per_complex) <= 0.15:
                        print(f'| WARNING: {n_bad}/{self.samples_per_complex} ({100*n_bad/self.samples_per_complex:.1f}%) poses degenerate - filtering and continuing')
                        predictions_list = [p for p, good in zip(predictions_list, ~bad_mask) if good]
                    elif 0 < n_bad:
                        raise ValueError(f'{n_bad}/{self.samples_per_complex} poses degenerate (>15%) - retrying')
                except Exception as e:
                    # if 'failed to converge' in str(e):
                    failed_convergence_counter += 1
                    predictions_list = None
                    if failed_convergence_counter > 10:
                        print('| WARNING: failed on complex 10 times - skipping the complex')
                        break
                    print(f'| WARNING: failed on complex - trying again with a new sample, {e}')
                    # else:
                    #     raise e
            # if failed_convergence_counter > 10: predictions_list = data_list
            if failed_convergence_counter > 10:
                # rmsds.append([100] * self.samples_per_complex)
                continue
            if self.original_model_args.no_torsion:
                orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

            filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

            if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

            ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
            orig_ligand_pos = np.expand_dims(orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(), axis=0)

            # KRA Edit 2: attempt symmetric rmsd (adapted from utils/training.py)
            mol = RemoveAllHs(rdkit_ligand)
            try:
                rmsd = np.array(get_symmetry_rmsd(mol, orig_ligand_pos[0], [l for l in ligand_pos]))
            except Exception as e:
                print(f'| WARNING: symmetry RMSD failed for {orig_complex_graph.name[0]}, using naive: {e}')
                rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
            # rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
            # Extra DEBUG:
            if idx < 50:
                n_atoms_graph = orig_ligand_pos.shape[1]
                n_atoms_ligand = RemoveAllHs(rdkit_ligand).GetNumAtoms() if rdkit_ligand is not None else None
                print(f"[{idx}] name={orig_complex_graph.name[0]}, "
                    f"graph_atoms={n_atoms_graph}, rdkit_atoms={n_atoms_ligand}, "
                    f"rmsd_min={rmsd.min():.3f}, rmsd_max={rmsd.max():.3f}")
            rmsds.append(rmsd)
            min_rmsds.append(rmsd.min())
            # Needs to get changed
            full_ligand_positions.append(np.asarray([complex_graph['ligand'].pos.cpu().numpy() for complex_graph in predictions_list]))
            names.append(orig_complex_graph.name[0])
            assert(len(orig_complex_graph.name) == 1) # I just put this assert here because of the above line where I assumed that the list is always only lenght 1. Just in case it isn't maybe check what the names in there are.
        with open(os.path.join(self.full_cache_path, f"ligand_positions{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
            pickle.dump((full_ligand_positions, rmsds), f)
        with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
            pickle.dump((names), f)
        
        # KRA Edited: final metrics
        rmsds_flat = np.concatenate(rmsds)
        min_rmsds = np.array(min_rmsds)
        original_set_length = len(read_strings_from_txt(self.split_path))
        print(f'complexes kept:  {len(names)} / {original_set_length}')
        print(f'rmsds_lt2:       {100 * (rmsds_flat < 2).mean():.2f}%')
        print(f'rmsds_lt5:       {100 * (rmsds_flat < 5).mean():.2f}%')
        print(f'min_rmsds_lt2:   {100 * (min_rmsds < 2).mean():.2f}%')
        print(f'min_rmsds_lt5:   {100 * (min_rmsds < 5).mean():.2f}%')



