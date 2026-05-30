import math
import random
from itertools import combinations

import numpy as np
import pandas as pd
import scipy
import torch
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import BaseScheduler
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Checkbox, Choice, NumberInput, Slider, StaticText
from mesa.visualization.modules import CanvasGrid
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model_setup import ExpDataset, block_receptors


def check_threshold(tensor1, tensor2, threshold):
    diff1 = tensor1 - tensor2
    diff2 = tensor2 - tensor1
    greater_than_threshold = torch.greater(diff1, threshold)
    less_than_threshold = torch.greater(diff2, threshold)
    return greater_than_threshold, less_than_threshold


_MODEL_CACHE = {}


def _get_nn_device(device=None):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_distance_device(device=None):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _load_model_cached(model_or_path, device=None):
    device = _get_nn_device() if device is None else torch.device(device)
    if hasattr(model_or_path, "parameters"):
        model = model_or_path
        if next(model.parameters()).device != device:
            model = model.to(device)
        return model
    cache_key = (model_or_path, str(device))
    if cache_key not in _MODEL_CACHE:
        model = torch.load(model_or_path, map_location=device)
        model = model.to(device)
        model.eval()
        _MODEL_CACHE[cache_key] = model
    return _MODEL_CACHE[cache_key]


def gene_exp_preds(model_path, lig_uni, rec_uni, inputs, all_genes, threshold=0.5, update=0.01):
    device = _get_nn_device()
    model = _load_model_cached(model_path, device=device)
    dataset2 = ExpDataset(inputs, all_genes)
    dataloader2 = DataLoader(dataset=dataset2, shuffle=False, batch_size=len(all_genes))

    for x_train, y_train in dataloader2:
        x_train = x_train.to(device, non_blocking=device.type == "cuda")
        y_train = y_train.to(device, non_blocking=device.type == "cuda")
        y_pred = model(x_train)
        for i, cell1 in enumerate(y_pred):
            t1 = y_train[i]
            greater_than_threshold, less_than_threshold = check_threshold(t1, cell1, threshold)
            up_idx = torch.where(greater_than_threshold)[0]
            down_idx = torch.where(less_than_threshold)[0]
            if len(up_idx) != 0:
                all_genes[i][up_idx] += update
            if len(down_idx) != 0:
                all_genes[i][down_idx] -= update

    all_genes = np.clip(all_genes, 0, None)
    return all_genes


def get_protein_choices(rec_uni):
    proteins = sorted(list(rec_uni.keys()))
    proteins.insert(0, False)
    return proteins


def get_lr_choices(rec_uni, lig_uni):
    pairs = set()
    for lig, rlist in lig_uni.items():
        for rec in rlist:
            pairs.add(f"{lig}-{rec}")
    for rec, llist in rec_uni.items():
        for lig in llist:
            pairs.add(f"{lig}-{rec}")
    pairs = sorted(pairs)
    pairs.insert(0, False)
    return pairs


def get_cluster_choices(adata):
    clusters = list(set(adata.obs["cell_type"]))
    clusters.insert(0, False)
    return clusters


def parameters(adata, lig_uni, rec_uni, rates, clusters, pairs, proteins):
    model_params = {
        "N": NumberInput("Number of agents", adata.shape[0]),
        "adata": adata,
        "lig_uni": lig_uni,
        "rec_uni": rec_uni,
        "rates": rates,
        "max_steps": Slider("Max number of steps", 1, 1, 100, 5),
        "delta": Slider("Delta", 1, 0, 10, 0.1),
        "tau": Slider("Tau", 2, 1, 10, 1),
        "rec_block": Choice("Block Receptor", value=False, choices=proteins),
        "protein_choice": Choice("Receptor", value=proteins[0], choices=proteins),
        "lr_choice": Choice("L-R Pair", value=pairs[0], choices=pairs),
        "dist": Checkbox("Distance", value=True),
        "sender": Choice("Sender", value=clusters[0], choices=clusters),
        "receiver": Choice("Receiver", value=clusters[0], choices=clusters),
        "noise": Slider("Gaussian Noise Percentage", 5, 0, 50, 1),
        "text": StaticText("This is a descriptive textbox"),
    }
    return model_params


def get_interactions(model):
    df1 = pd.DataFrame.from_dict(model.results, orient="index").transpose()
    df1 = df1.rename({0: "Sum"})
    return df1


def get_lr_interactions(model):
    return pd.DataFrame.from_dict(model.results2, orient="index").fillna(0).transpose()


def get_lr_interactions2(model):
    return model.results2


def get_avg_dist(model):
    return model.avg_dist


def _euclidean_distance(p1, p2):
    arr1 = np.asarray(p1, dtype=float)
    arr2 = np.asarray(p2, dtype=float)
    return float(np.linalg.norm(arr2 - arr1))


def _torch_pairwise_distances(coords, device=None):
    device = _get_distance_device(device)
    coord_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    distances = torch.cdist(coord_tensor, coord_tensor)
    return distances


class CellAgent(Agent):
    def __init__(self, unique_id, model, clust, exp, batch):
        super().__init__(unique_id, model)
        self.id = unique_id
        self.mobile = True
        self.cluster = clust
        self.expression = exp
        self.slice = batch
        # Visualization Tracking Fields
        self.num_r = 0
        self.iqr = "very_low"

    def Message(self):
        rates2 = {}
        rec_list = [
            (key, self.expression[key])
            for key in self.expression.keys()
            if key in self.model.rec_uni.keys() and self.expression[key] > 0.0
        ]
        if len(rec_list) != 0:
            # Receptor Receiving Rate
            unzipped_rec = list(zip(*rec_list))
            rr = np.array(unzipped_rec[1]) / np.max(np.array(unzipped_rec[1]))
            if rr.min() < 0.3:
                rr = (np.array(unzipped_rec[1]) / np.max(np.array(unzipped_rec[1])) * 0.7) + 0.3
            for i, val in enumerate(unzipped_rec[0]):
                rates2[val] = self.model.rates[val] * rr[i]

        # Ligand Expression
        clust1 = self.cluster
        cell_ligands = self.model.ligs
        if self.model.dist_param != 0:
            c = self.model.distances[self]

        for clust2 in cell_ligands.keys():
            ckey = str(clust2) + "_" + str(clust1)
            if ckey in self.model.output.keys():
                cnum = self.model.output[ckey]
                cdict = self.model.output2[ckey]
            else:
                cnum = 0
                cdict = {}

            lig_exp = cell_ligands[clust2] # ligand expression for sending cluster
            if self.model.dist_param != 0:
                dist2 = c[clust2]
            if len(rec_list) != 0:
                for rec in unzipped_rec[0]:    # each receptor
                    poss_ligs = self.model.rec_uni[rec]  # possible liganbds for each receptor
                    for lig in poss_ligs:
                        exp = lig_exp[lig]   # expression of each ligand
                        # Calculate ligand score
                        if self.model.dist_param != 0:
                            if not (isinstance(self.model.delta, int) or isinstance(self.model.delta, float)):
                                lig_delta = self.model.delta[lig]
                            else:
                                lig_delta = self.model.delta
                            new_dist2 = [1 / (d ** (self.model.dist_param * lig_delta)) if d != 0 else 0 for d in dist2]
                            b_length = np.count_nonzero(new_dist2)
                            lig_score = np.multiply(exp, new_dist2).sum() / b_length if b_length != 0 else 0
                        else:
                            lig_score = exp

                        if lig_score != 0:
                            lr_str = lig + "-" + rec
                            # Multiply rates
                            final = lig_score * rates2[rec]
                            if (
                                (self.model.protein_choice is False or self.model.protein_choice == rec)
                                and (self.model.sender == clust2 or self.model.sender is False)
                                and (self.model.receiver == clust1 or self.model.receiver is False)
                                and (self.model.lr_choice is False or self.model.lr_choice == lr_str)
                            ):
                                self.num_r += final

                            cnum = cnum + final
                            # Sum of specific LR pairs between clusters
                            if lr_str in cdict.keys():
                                lr_val = cdict[lr_str]
                                lr_val.append(final)
                                cdict[lr_str] = lr_val
                            else:
                                cdict[lr_str] = [final]

            # Save interactions
            self.model.output[ckey] = cnum
            self.model.output2[ckey] = cdict\
            
        # For animation
        if self.model.receiver == clust1 or self.model.receiver is False:
            self.model.rec_score.append(self.num_r)

    def step(self):
        self.Message()


class CellModel(Model):
    """
    CellAgentChat Parameters Help

    Number of Agents: The total count of cells utilized in the simulation.
    Max number of steps: The duration of the simulation expressed in iterations.
    Delta: Influences the degree of cell-to-cell distance (default=1). For long-range mode, a delta value less than 1 is used, while for short-range mode, a delta value greater than 1 is employed.
    Tau: Represents the degree of freedom for distance (default=2).
    Block Receptor: Specifies the receptor to be obstructed.
    Receptor Track: Designates the receptor to be tracked by the animation. Only interactions involving the selected receptor will be displayed on the screen.
    L-R Pair: Identifies the L-R pair to be tracked by the animation. Only interactions involving the chosen L-R pair will be shown on the screen.
    Distance: Distance mode setting.
    Sender: Refers to the cell type responsible for sending ligands. Only interactions where the ligands originate from the specified sending cell type will be displayed on the screen.
    Receiver: Refers to the cell type responsible for receiving ligands. Only interactions where ligands are received by the specified receiving cell type will be shown on the screen.
    Gaussian Noise Percentage: The percentage of Gaussian noise added to ligand expression to enhance the dynamism of the simulations (default=5%).
    """

    def __init__(
        self,
        N,
        adata,
        lig_uni,
        rec_uni,
        rates,
        max_steps,
        dist,
        delta=1,
        tau=2,
        diff=1,
        rec_block=False,
        protein_choice=False,
        lr_choice=False,
        sender=False,
        receiver=False,
        permutations=False,
        net=None,
    ):
        # Mandatory Inputs
        self.num_agents = N
        self.adata = adata
        self.lig_uni = lig_uni
        self.rec_uni = rec_uni
        self.rates = rates.copy() if isinstance(rates, dict) else rates
        self.max_steps = max_steps
        self.dist_on = dist

        # Other mandatory fields
        self.running = True
        self.genes = [gene for gene in adata.var_names if gene in rec_uni or gene in lig_uni]
        self.grid = MultiGrid(101, 101, True)
        self.schedule = BaseScheduler(self)
        self.curr_step = 0
        
        # Optional fields
        self.delta = delta
        self.diffusion = diff
        self.protein_choice = protein_choice
        self.lr_choice = lr_choice
        self.dist_param = tau
        self.rec_block = rec_block
        self.sender = sender
        self.receiver = receiver
        self.permutations = permutations
        self.net = net
        self.avg_dist = 0

        # Agent messaging outputs and results
        self.output = {}
        self.output2 = {}
        self.output3 = {}
        self.output4 = {}
        self.results = {}
        self.results2 = {}

        # Helper fields
        self.ligs = 0
        self.distances = 0
        self.clusters = {}
        self.rec_score = []
        self.tokenizer = {}

        # Agent related fields
        self._rate_override_applied = False
        self._gene_indices = [adata.var_names.get_loc(gene) for gene in self.genes]
        self._rec_indices = [i for i, gene in enumerate(self.adata.var_names) if gene in self.rec_uni]
        self._lig_indices = [i for i, gene in enumerate(self.adata.var_names) if gene in self.lig_uni]
        self._raw_matrix = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
        self.all_genes = np.array(self._raw_matrix, copy=True)
        self._gene_subset = self._raw_matrix[:, self._gene_indices] if self._gene_indices else np.empty((N, 0))
        self._obs_names = list(adata.obs_names)
        self._cell_types = [str(val) for val in adata.obs["cell_type"]]
        self._batches = [str(val) for val in adata.obs["Batch"]]
        self._grid_coords = self._get_grid_coordinate_array()
        self._distance_coords = self._get_distance_coordinate_array()
        self._agent_distance_coords = {}
        self._distance_device = _get_distance_device()

        for i, clust in enumerate(sorted(set(self._cell_types))):
            self.tokenizer[str(clust)] = i

        # Create agents
        agent_iter = tqdm(range(self.num_agents), desc="Initializing agents", unit="agent", leave=True)
        for i in agent_iter:
            clust = self._cell_types[i]       # Cluster 
            batch = self._batches[i]          # Batch
            exp = dict(zip(self.genes, self._gene_subset[i].tolist()))      # Expression
            a = CellAgent(i, self, clust, exp, batch)
            self.schedule.add(a)
            if self._distance_coords is not None:
                self._agent_distance_coords[a] = np.asarray(self._distance_coords[i], dtype=float)

            # Coordinates
            if self.dist_on and self._grid_coords is not None:
                grid_x, grid_y = self._project_to_grid(self._grid_coords[i])
                self.grid.place_agent(a, (grid_x, grid_y))
            else:
                x = random.randint(0, 100)
                y = random.randint(0, 100)
                self.grid.place_agent(a, (x, y))
                self.dist_param = 0

            # Add cells to clusters
            self.clusters.setdefault(clust, []).append(a)

    def _get_coordinate_array(self, coordinate_sets):
        for columns in coordinate_sets:
            if all(col in self.adata.obs.columns for col in columns):
                return np.asarray(self.adata.obs[columns].to_numpy(), dtype=float)
        return None

    def _get_grid_coordinate_array(self):
        # Use the display/grid coordinates for Mesa placement.
        return self._get_coordinate_array((["x", "y", "z"], ["x", "y"], ["x_true", "y_true", "z_true"], ["x_true", "y_true"]))

    def _get_distance_coordinate_array(self):
        # Keep raw coordinates for distance-based signaling and CCI calculations.
        return self._get_coordinate_array((["x_raw", "y_raw", "z_raw"], ["x_raw", "y_raw"], ["x_true", "y_true", "z_true"], ["x_true", "y_true"], ["x", "y", "z"], ["x", "y"]))

    def _project_to_grid(self, coord):
        arr = np.asarray(coord, dtype=float)
        return int(arr[0]), int(arr[1])
    
    #Function to modify database   
    def modify_db(self):
        if self.rec_block is not False and not self._rate_override_applied:
            block_targets = self.rec_block if isinstance(self.rec_block, (list, tuple, set)) else [self.rec_block]
            for rec in block_targets:
                if rec in self.rates:
                    self.rates[rec] = 0
            self._rate_override_applied = True

    def get_clusters(self):
        return self.clusters

    def get_dist(self):
        #{cell1: {cluster1: [1/dist1, ..., 1/distn]}}
        new_dict = {}
        clusters = self.get_clusters()
        for cell in self.schedule.agents:
            adict = {}
            for clust, cells in clusters.items():
                distances = []
                for cell2 in cells:
                    if cell.slice == cell2.slice:
                        d = _euclidean_distance(self._agent_distance_coords[cell], self._agent_distance_coords[cell2])
                        if d == 0:
                            d = 1
                    else:
                        d = 0
                    distances.append(d)
                adict[clust] = distances
            new_dict[cell] = adict
        return new_dict

    def get_normalized_dist(self):
        clusters = self.get_clusters()
        cell_ids = [cell.id for cell in self.schedule.agents]
        slices = np.array([cell.slice for cell in self.schedule.agents])
        if self._distance_coords is None:
            raise ValueError("Distance mode requires raw x/y or raw x/y/z coordinates in adata.obs")

        coords = self._distance_coords[cell_ids]
        distances = _torch_pairwise_distances(coords, device=self._distance_device)
        diag_idx = torch.arange(distances.shape[0], device=distances.device)
        distances[diag_idx, diag_idx] = 1
        slice_mask = torch.tensor(slices[:, None] == slices[None, :], dtype=distances.dtype, device=distances.device)
        distances = distances * slice_mask
        positive = distances[distances > 0]
        if positive.numel() > 0:
            min_d = positive.min()
            max_d = distances.max()
            if (max_d - min_d).item() > 0:
                distances = ((distances - min_d) / (max_d - min_d)) * 100
            else:
                distances = torch.zeros_like(distances)
        else:
            distances = torch.zeros_like(distances)
        distances = distances.detach().cpu().numpy()
        distance_df = pd.DataFrame(distances, index=cell_ids, columns=cell_ids)

        new_dict = {}
        for cell in self.schedule.agents:
            cell_id = cell.id
            adict = {}
            for clust, cells in clusters.items():
                cell2_ids = [c.id for c in cells]
                adict[clust] = distance_df.loc[cell_id, cell2_ids].tolist()
            new_dict[cell] = adict
        return new_dict

    def calc_ligands(self):
        new_dict = {}
        clusters = self.get_clusters()
        for clust, cells in clusters.items():
            ligd = {}
            for lig in self.lig_uni.keys():
                expl = [cell.expression[lig] for cell in cells]
                if self.dist_param == 0:
                    expl = sum(expl) / len(expl)
                ligd[lig] = expl
            new_dict[clust] = ligd
        return new_dict

    def dist(self, p1, p2):
        return _euclidean_distance(p1, p2)

    def calc_dist(self):
        if self._distance_coords is not None:
            points = self._distance_coords
        else:
            points = [cell.pos for cell in self.schedule.agents]
        distances2 = [self.dist(p1, p2) for p1, p2 in combinations(points, 2)]
        return sum(distances2) / len(distances2)

    def calc_normalized_dist(self):
        if self._distance_coords is None:
            raise ValueError("Distance mode requires raw x/y or raw x/y/z coordinates in adata.obs")
        distances = _torch_pairwise_distances(self._distance_coords, device=self._distance_device)
        tri_idx = torch.triu_indices(distances.shape[0], distances.shape[1], offset=1, device=distances.device)
        distances2 = distances[tri_idx[0], tri_idx[1]]
        if distances2.numel() == 0:
            return 0.0
        dmin = distances2.min()
        dmax = distances2.max()
        if (dmax - dmin).item() > 0:
            distances2_normalized = ((distances2 - dmin) / (dmax - dmin)) * 100
        else:
            distances2_normalized = torch.zeros_like(distances2)
        return float(distances2_normalized.mean().item())

    def step(self, show_agent_progress=True, agent_progress_position=0, agent_progress=None):
        self.rec_score = []
        if self.dist_param != 0 and self.curr_step == 0:
            self.avg_dist = self.calc_normalized_dist()
            self.distances = self.get_normalized_dist()
            print("Average Distance: " + str(self.avg_dist))
        self.ligs = self.calc_ligands()
        self.modify_db()
        if agent_progress is not None or show_agent_progress:
            if agent_progress is None:
                with tqdm(
                    total=self.schedule.get_agent_count(),
                    desc=f"Agents in step {self.curr_step + 1}",
                    unit="agent",
                    leave=True,
                    position=agent_progress_position,
                    dynamic_ncols=True,
                ) as agent_iter:
                    for agent in self.schedule.agent_buffer(shuffled=False):
                        agent.step()
                        agent_iter.update(1)
            else:
                for agent in self.schedule.agent_buffer(shuffled=False):
                    agent.step()
                    agent_progress.update(1)
            self.schedule.steps += 1
            self.schedule.time += 1
        else:
            self.schedule.step()

        for cpair in self.output.keys():
            splitted = cpair.split("_")
            c2 = splitted[1]
            cdict = self.output2[cpair]
            cdict2 = self.output4.get(cpair, {})
            num = len(self.clusters[c2])
            for lr in cdict.keys():
                val = sum(cdict[lr]) / num
                prev = self.output4[cpair][lr] if self.curr_step != 0 and cpair in self.output4 and lr in self.output4[cpair] else 0
                prev2 = self.output3[cpair] if self.curr_step != 0 and cpair in self.output3 else 0
                cdict[lr] = (val + prev) / (self.curr_step + 1)
                cdict2[lr] = val + prev
            self.output4[cpair] = cdict2
            prev2 = self.output3.get(cpair, 0)
            self.output3[cpair] = sum(list(self.output2[cpair].values())) + prev2
            self.results2[cpair] = cdict
            self.results[cpair] = (sum(list(self.output2[cpair].values())) + prev2) / (self.curr_step + 1)

        self.output = {}
        self.output2 = {}

        if not self.permutations:
            score2 = [x / (self.curr_step + 1) for x in self.rec_score]
            score2 = sorted(score2)
            if len(score2) != 0:
                b_size = max(1, len(score2) // 6)
                high = min(len(score2) - 1, b_size * 5)
                med_high = min(len(score2) - 1, b_size * 4)
                med_low = min(len(score2) - 1, b_size * 3)
                low = min(len(score2) - 1, b_size * 2)
                very_low = min(len(score2) - 1, b_size * 1)
                for agent in self.schedule.agents:
                    received = agent.num_r / (self.curr_step + 1)
                    if agent.cluster == self.receiver or self.receiver is False:
                        if received <= score2[very_low]:
                            agent.iqr = "very_low"
                        elif received < score2[low]:
                            agent.iqr = "low"
                        elif received < score2[med_low]:
                            agent.iqr = "med_low"
                        elif received < score2[med_high]:
                            agent.iqr = "med_high"
                        elif received < score2[high]:
                            agent.iqr = "high"
                        else:
                            agent.iqr = "very_high"
                    else:
                        agent.iqr = "very_low"

        self.curr_step += 1
        if self.max_steps != 1 and self.net is not None:
            input_recs = self.all_genes[:, self._rec_indices]
            input_ligs = self.all_genes[:, self._lig_indices]
            lig_averages = np.mean(input_ligs, axis=0)
            repeated_averages = np.repeat(lig_averages.reshape(1, -1), input_recs.shape[0], axis=0)
            inputs = np.concatenate((input_recs, repeated_averages), axis=1)
            self.all_genes = gene_exp_preds(self.net, self.lig_uni, self.rec_uni, inputs, self.all_genes, threshold=0.5, update=0.01)

            for i, cell in enumerate(self.schedule.agents):
                exp = dict(zip(self.genes, self.all_genes[i, self._gene_indices].tolist()))
                cell.expression = exp

        if self.max_steps == self.curr_step:
            if self.rec_block is not False:
                targets = self.rec_block if isinstance(self.rec_block, (list, tuple, set)) else [self.rec_block]
                block_receptors(self.adata, targets, self.rec_uni, self.lig_uni, self.net, device=_get_nn_device())
            self.running = False


def agent_portrayal(agent):
    colours = ["violet", "brown", "pink", "red", "green", "orange", "grey", "teal", "navy", "blue", "magenta", "cyan", "yellow", "black", "maroon"]
    portrayal = {
        "Shape": "circle",
        "Layer": 0,
        "Cell id": str(agent.unique_id),
        "r": 0.25,
        "Filled": True,
        "Cell Type": str(agent.cluster),
    }

    token = agent.model.tokenizer[str(agent.cluster)]
    if token < len(colours):
        portrayal["Color"] = colours[token]

    step = 1 if agent.model.curr_step == 0 else agent.model.curr_step
    if agent.iqr == "very_low":
        portrayal["r"] = 0.16
    elif agent.iqr == "low":
        portrayal["r"] = 0.33
    elif agent.iqr == "med_low":
        portrayal["r"] = 0.50
    elif agent.iqr == "med_high":
        portrayal["r"] = 0.66
    elif agent.iqr == "high":
        portrayal["r"] = 0.83
    else:
        portrayal["r"] = 1.0

    portrayal["Receiving_score"] = agent.num_r / step
    return portrayal


def visualization(adata, model_params, dist_on=True, port=8521):
    model = CellModel
    grid2 = CanvasGrid(agent_portrayal, 90, 130, 700, 700)
    server = ModularServer(model, [grid2], "CellAgentChat", model_params)
    server.port = port
    server.launch()
