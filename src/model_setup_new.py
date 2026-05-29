import copy
import csv
import math
import os
from importlib import resources

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import sparselinear as sl
import torch
from scipy.stats import gamma, norm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import ot
from output_paths_new import artifact_dir, artifact_path


def _get_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _get_model_device(model, device=None):
    if device is not None:
        return _get_device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return _get_device()


def _move_model_to_device(model, device=None):
    resolved_device = _get_model_device(model, device=device)
    return model.to(resolved_device), resolved_device


def _module_path(*parts):
    return os.path.join(os.path.dirname(__file__), *parts)


def _packaged_data_path(*parts):
    return str(resources.files("cellagentchat_data").joinpath(*parts))


def _resolve_data_file(path):
    if os.path.isabs(path) or os.path.exists(path):
        return path

    local_path = _module_path(path)
    if os.path.exists(local_path):
        return local_path

    normalized = path.replace("\\", "/")
    package_parts = tuple(part for part in normalized.split("/") if part)
    packaged_path = _packaged_data_path(*package_parts)
    if os.path.exists(packaged_path):
        return packaged_path

    return path


def _dense_matrix(adata):
    return adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)


def _build_io_matrices(adata, lig_uni, rec_uni):
    rec_keys = list(rec_uni.keys())
    lig_keys = list(lig_uni.keys())
    if scipy.sparse.issparse(adata.X):
        input_recs = adata[:, rec_keys].X.toarray()
        input_ligs = adata[:, lig_keys].X.toarray()
    else:
        input_recs = np.array(adata[:, rec_keys].X)
        input_ligs = np.array(adata[:, lig_keys].X)
    lig_averages = np.mean(input_ligs, axis=0)
    inputs = np.concatenate((input_recs, np.repeat(lig_averages.reshape(1, -1), input_recs.shape[0], axis=0)), axis=1)
    all_genes = _dense_matrix(adata)
    return inputs, all_genes, input_recs, input_ligs


def _loader_kwargs(device, shuffle=False, batch_size=256):
    use_cuda = device.type == "cuda"
    return {
        "shuffle": shuffle,
        "batch_size": batch_size,
        "pin_memory": use_cuda,
    }


def _resolve_output_file(path, kind, default_name):
    if path is None:
        return artifact_path(kind, default_name)
    if os.path.isabs(path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return path
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    if path == default_name:
        return artifact_path(kind, default_name)
    return path


def load_db(adata, file="human_lr_pair.tsv", sep="\t"):
    print("Loading Database...")
    file = _resolve_data_file(file)
    lig_uni = {}
    rec_uni = {}

    with open(file, "r") as csvfile:
        datareader = csv.reader(csvfile, delimiter=sep)
        next(datareader, None)
        genes = set(adata.var_names)
        for row in datareader:
            ligand = row[1]
            receptor = row[2]
            if ligand in genes and receptor in genes:
                lig_uni.setdefault(ligand, []).append(receptor)
                rec_uni.setdefault(receptor, []).append(ligand)

    lr_pairs = [f"{ligand}_{receptor}" for ligand, receptors in lig_uni.items() for receptor in receptors]
    print("Database Loaded!")
    return lig_uni, rec_uni, lr_pairs


def load_tf_db(species, adata, rec_uni):
    if species == "mouse":
        file1 = _resolve_data_file("databases/TF_TG_mouse.csv")
        file2 = _resolve_data_file("databases/KEGG_mouse.csv")
        file3 = _resolve_data_file("databases/REACTOME_mouse.csv")
    elif species == 'human':
        file1 = _resolve_data_file("databases/TF_TG_human.csv")
        file2 = _resolve_data_file("databases/KEGG_human.csv")
        file3 = _resolve_data_file("databases/REACTOME_human.csv")

    df = pd.read_csv(file1).drop(columns="Unnamed: 0", errors="ignore")
    df2 = pd.read_csv(file2).drop(columns="Unnamed: 0", errors="ignore")
    df3 = pd.read_csv(file3).drop(columns="Unnamed: 0", errors="ignore")

    def _explode_rec_tf(frame):
        receptors = list(frame["receptor"])
        tfs = list(frame["tf"])
        list1 = []
        list2 = []
        for i, rec in enumerate(receptors):
            for rec2 in rec.split(","):
                list1.append(rec2)
                list2.append(tfs[i])
        return pd.DataFrame({"Receptors": list1, "TF": list2})

    #Rec to TF dataframe
    rec_tf = pd.concat([_explode_rec_tf(df2), _explode_rec_tf(df3)]).drop_duplicates()
    genes = set(adata.var_names)

    #TF universe - TF to downstream genes
    tf_uni = {}
    for _, row in df.iterrows():
        tf = row["TF"]
        gene = row["Target Gene"]
        if tf in genes and gene in genes:
            tf_uni.setdefault(tf, [])
            if gene not in tf_uni[tf]:
                tf_uni[tf].append(gene)

    #Rec to TF universe  
    rec_tf_uni = {}
    for _, row in rec_tf.iterrows():
        rec = row["Receptors"]
        tf = row["TF"]
        if rec in rec_uni and tf in tf_uni:
            rec_tf_uni.setdefault(rec, [])
            if tf not in rec_tf_uni[rec]:
                rec_tf_uni[rec].append(tf)

    return tf_uni, rec_tf_uni


def create_masked_connections(adata, lig_uni, rec_uni, tf_uni, rec_tf_uni, lr_pairs):
    num_rec = len(rec_uni)
    num_lig = len(lig_uni)
    num_inputs = num_rec + num_lig
    input_names = list(rec_uni.keys()) + list(lig_uni.keys())
    inputs, all_genes, _, _ = _build_io_matrices(adata, lig_uni, rec_uni)

    num_lr_pairs = len(lr_pairs)
    num_tfs = len(tf_uni.keys())
    num_outputs = all_genes.shape[1]

    lr_index = dict(zip(lr_pairs, list(range(num_lr_pairs))))
    input_index = dict(zip(input_names, list(range(num_inputs))))
    tf_index = dict(zip(list(tf_uni.keys()), list(range(num_tfs))))
    gene_index = dict(zip(list(adata.var_names), list(range(num_outputs))))

    # Layer 1 Connections
    mask = np.zeros((num_lr_pairs, num_inputs))
    for pair, i in lr_index.items():
        lig, rec = pair.split("_")
        mask[i][input_index[lig]] = 1
        mask[i][input_index[rec]] = 1
    mask = torch.nonzero(torch.tensor(mask)).T

    # Layer 2 Connections
    mask2 = np.zeros((num_tfs, num_lr_pairs))
    for pair, j in lr_index.items():
        _, rec = pair.split("_")
        if rec in rec_tf_uni:
            for tf in rec_tf_uni[rec]:
                mask2[tf_index[tf]][j] = 1
        else:
            mask2[:, j] = np.ones(mask2.shape[0])
    mask2 = mask2.T

    # Layer 3 Connections
    mask3 = np.zeros((num_outputs, num_tfs))
    for tf, j in tf_index.items():
        for gene in tf_uni[tf]:
            mask3[gene_index[gene]][j] = 1

    # for any downstream gene with no connection make them fully connected with previous layer
    for i in range(num_outputs):
        if not np.any(mask3[i]):
            mask3[i] = np.ones(len(mask3[i]))
    mask3 = mask3.T
    return inputs, all_genes, num_inputs, num_lr_pairs, num_tfs, num_outputs, mask, mask2, mask3


def normalize(mat):
    smat = mat.transpose()
    for i, row in enumerate(smat):
        mmin = min(row)
        mmax = max(row)
        for j, val in enumerate(row):
            if mmin == mmax:
                val2 = 0
            else:
                val2 = (val - mmin) / (mmax - mmin)
            row[j] = val2
        smat[i] = row
    mat = smat.transpose()
    return mat


class CustomizedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                grad_weight = grad_weight * mask
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True):
        super().__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        return f"input_features={self.input_features}, output_features={self.output_features}, bias={self.bias is not None}"


class net(nn.Module):
    def __init__(self, num_inputs, num_lr_pairs, num_tfs, output_size, mask1, mask2, mask3):
        super().__init__()
        self.layers = nn.Sequential(
            sl.SparseLinear(in_features=num_inputs, out_features=num_lr_pairs, connectivity=mask1),
            nn.BatchNorm1d(num_lr_pairs),
            nn.ReLU(),
            CustomizedLinear(torch.tensor(mask2), bias=None),
            nn.BatchNorm1d(num_tfs),
            CustomizedLinear(torch.tensor(mask3), bias=None),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.layers(x)
        return output


class ExpDataset(Dataset):
    def __init__(self, x, y, device=None):
        tensor_device = _get_device(device) if device is not None else None
        self.x = torch.tensor(x, dtype=torch.float32, device=tensor_device)
        self.y = torch.tensor(y, dtype=torch.float32, device=tensor_device)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


def save_conversion_rate(conversion_rates, file="conversion_rates.txt"):
    file = _resolve_output_file(file, "conversion_rates", "conversion_rates.txt")
    with open(file, "w") as fp:
        for values in conversion_rates:
            fp.write(f"{values}\n")
    print("Done")


def load_conversion_rate(file="conversion_rates.txt"):
    file = _resolve_output_file(file, "conversion_rates", "conversion_rates.txt")
    conversion_rates = []
    with open(file, "r") as fp:
        for line in fp:
            conversion_rates.append(float(line[:-1]))
    print("Conversion rates loaded")
    return conversion_rates


def add_rates(conversion_rates, rec_uni):
    rates = dict(zip(list(rec_uni.keys()), conversion_rates))
    return rates


def load_model(path="model.pt", device=None):
    path = _resolve_output_file(path, "models", "model.pt")
    resolved_device = _get_device(device)
    model = torch.load(path, map_location=resolved_device, weights_only=False)
    return model.to(resolved_device)


def plot_results(dict1, adata, path="figures"):
    path = artifact_dir("perturbation") if path == "figures" else path
    os.makedirs(path, exist_ok=True)
    for rec in dict1.keys():
        print("Plotting results for receptor: " + rec)

        # Barplot 
        genes = dict1[rec][1]
        values = [x * 100 for x in dict1[rec][0]]
        genes2 = [y for _, y in sorted(zip(values, genes), reverse=True)]
        values.sort(reverse=True)
        plt.figure(figsize=(10, 7))
        plt.xticks(rotation=90)
        plt.xlabel("Target Genes")
        plt.ylabel("Percent Change in Gene Expression")
        plt.ylim((min(values), max(values) + 0.05))
        plt.bar(genes2, values)
        plt.savefig(path + "/" + rec + "_targets.pdf")
        plt.close()

        # Matrix Plot
        prev_figdir = sc.settings.figdir
        try:
            sc.settings.figdir = path
            sc.pl.matrixplot(
                adata,
                genes2,
                groupby="cell_type",
                cmap="viridis",
                standard_scale="var",
                save=rec + "_matrix.pdf",
            )
        finally:
            sc.settings.figdir = prev_figdir
    print("plots saved to directory /" + path)


def train(adata, lig_uni, rec_uni, tf_uni, rec_tf_uni, lr_pairs, path="model.pt", epochs=50, lr=0.1, device=None):
    print("Setting up model")
    path = _resolve_output_file(path, "models", "model.pt")
    device = _get_device(device)
    model_components = create_masked_connections(adata, lig_uni, rec_uni, tf_uni, rec_tf_uni, lr_pairs)
    inputs, all_genes, num_inputs, num_lr_pairs, num_tfs, num_outputs, mask, mask2, mask3 = model_components
    dataset = ExpDataset(inputs, all_genes, device=device)
    dataloader = DataLoader(dataset=dataset, **_loader_kwargs(device, shuffle=False, batch_size=256))
    model = net(num_inputs, num_lr_pairs, num_tfs, num_outputs, mask, mask2, mask3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print("Training model...")
    epoch_losses = []
    epoch_iter = tqdm(range(epochs), desc="Training epochs", unit="epoch")
    for _ in epoch_iter:
        last_cost = None
        for x_train, y_train in dataloader:
            x_train = x_train.to(device, non_blocking=device.type == "cuda")
            y_train = y_train.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad()
            y_pred = model(x_train)
            cost = criterion(y_pred, y_train)
            cost.backward()
            optimizer.step()
            last_cost = cost
        if last_cost is not None:
            epoch_iter.set_postfix(loss=f"{last_cost.item():.6f}")
            epoch_losses.append(last_cost.detach().cpu())
    model = model.to("cpu")
    torch.save(model, path)
    print("Training complete!")
    return inputs, all_genes


def perform_iteration(model, mat, C, criterion, batch=256, device=None):
    model, device = _move_model_to_device(model, device=device)
    dataset2 = ExpDataset(mat, C, device=device)
    dataloader2 = DataLoader(dataset=dataset2, **_loader_kwargs(device, shuffle=False, batch_size=batch))
    predictions = []
    total_loss = torch.zeros((), device=device)
    total_examples = 0
    for x_train, y_train in dataloader2:
        x_train = x_train.to(device, non_blocking=device.type == "cuda")
        y_train = y_train.to(device, non_blocking=device.type == "cuda")
        batch_pred = model(x_train)
        batch_cost = criterion(batch_pred, y_train)
        batch_size = x_train.shape[0]
        predictions.append(batch_pred)
        total_loss += batch_cost * batch_size
        total_examples += batch_size

    if not predictions:
        raise ValueError("perform_iteration received no data to iterate over")

    y_pred = torch.cat(predictions, dim=0)
    cost = total_loss / total_examples
    return y_pred, cost


def _shuffle_array(values, rng):
    shuffled = np.array(values, copy=True)
    rng.shuffle(shuffled)
    return shuffled


def _apply_receptor_perturbation(values, perc, zeros, rng):
    perturbed_values = _shuffle_array(values, rng)
    nonzero_values = perturbed_values[perturbed_values != 0.0]
    if perc == 100:
        return zeros
    if len(nonzero_values) != 0:
        scale = np.percentile(nonzero_values, 100 - perc)
        perturbed_values = perturbed_values * scale if scale < 1 else perturbed_values / scale
    return perturbed_values


def permutate_receptors(
    model,
    start,
    num,
    ocost,
    mat,
    C,
    criterion,
    loss=None,
    perc=100,
    n_shuffles=10,
    seed=None,
    device=None,
):
    if loss is None:
        loss = []
    model, device = _move_model_to_device(model, device=device)
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be at least 1")

    zeroed_values = np.zeros(len(mat))
    base_rng = np.random.default_rng(seed)
    for receptor_idx in tqdm(range(start, num), desc="Permuting receptors", unit="receptor"):
        receptor_losses = []
        for _ in range(n_shuffles):
            perturbed_matrix = copy.deepcopy(mat).transpose()
            perturbed_values = _apply_receptor_perturbation(perturbed_matrix[receptor_idx], perc, zeroed_values, base_rng)
            perturbed_matrix[receptor_idx] = perturbed_values
            perturbed_matrix = perturbed_matrix.transpose()
            _, perturbed_cost = perform_iteration(
                model, perturbed_matrix, C, criterion, batch=perturbed_matrix.shape[0], device=device
            )
            receptor_losses.append(abs(perturbed_cost - ocost))
        loss.append(torch.stack(receptor_losses).mean())
    return loss


def feature_selection(model, mat, C, rec_uni, start=0, perc=100, n_shuffles=10, seed=None, device=None):
    print("Performing feature selection to obtain conversion rates...")
    model, device = _move_model_to_device(model, device=device)
    criterion = nn.MSELoss()
    _, cost = perform_iteration(model, mat, C, criterion, batch=mat.shape[0], device=device)

    losses = permutate_receptors(
        model,
        start=start,
        num=len(rec_uni),
        mat=mat,
        C=C,
        ocost=cost,
        loss=[],
        criterion=criterion,
        perc=perc,
        n_shuffles=n_shuffles,
        seed=seed,
        device=device,
    )
    mean_loss = torch.mean(torch.stack(losses))
    normalized_losses = [loss_value / mean_loss for loss_value in losses]
    conversion_rates = []
    for normalized_loss in normalized_losses:
        if normalized_loss > 1:
            conversion_rates.append(1)
        elif normalized_loss < 0.4:
            conversion_rates.append(0.4)
        else:
            conversion_rates.append(normalized_loss.item())

    print("Complete")
    return conversion_rates


def get_target_genes(
    receptors,
    N,
    model,
    mat,
    C,
    rec_uni,
    adata,
    perc=100,
    threshold=50,
    n_shuffles=10,
    seed=None,
    device=None,
):
    target_gene_results = {}
    model, device = _move_model_to_device(model, device=device)
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be at least 1")

    baseline_predictions, _ = perform_iteration(model, mat, C, nn.MSELoss(), batch=mat.shape[0], device=device)
    receptor_names = list(rec_uni.keys())
    base_rng = np.random.default_rng(seed)
    for rec in tqdm(receptors, desc="Getting target genes", unit="receptor"):
        zeroed_values = np.zeros(N)
        shuffled_predictions = []
        for _ in range(n_shuffles):
            perturbed_matrix = copy.deepcopy(mat).transpose()
            for receptor_idx, receptor_name in enumerate(receptor_names):
                if receptor_name != rec:
                    continue
                perturbed_values = _apply_receptor_perturbation(
                    perturbed_matrix[receptor_idx], perc, zeroed_values, base_rng
                )
                perturbed_matrix[receptor_idx] = perturbed_values
                break
            perturbed_matrix = perturbed_matrix.transpose()
            shuffled_prediction, _ = perform_iteration(
                model, perturbed_matrix, C, nn.MSELoss(), batch=perturbed_matrix.shape[0], device=device
            )
            shuffled_predictions.append(shuffled_prediction)
        mean_perturbed_prediction = torch.stack(shuffled_predictions).mean(dim=0)
        prediction_diff = abs(baseline_predictions - mean_perturbed_prediction).detach().cpu().numpy()
        mean_gene_diff = prediction_diff.mean(axis=0)
        top_gene_indices = np.argpartition(mean_gene_diff, -threshold)[-threshold:]
        top_gene_scores = list(mean_gene_diff[top_gene_indices])
        top_gene_names = [gene for gene_idx, gene in enumerate(adata.var_names) if gene_idx in top_gene_indices]
        target_gene_results[rec] = (top_gene_scores, top_gene_names)
    return target_gene_results


def block_receptors(
    adata,
    receptors,
    rec_uni,
    lig_uni,
    net,
    perc=100,
    threshold=50,
    n_shuffles=10,
    seed=None,
    device=None,
):
    print("Blocked Receptor Analysis")
    device = _get_device(device)
    model = load_model(net, device=device)
    model.eval()
    inputs, all_genes, _, _ = _build_io_matrices(adata, lig_uni, rec_uni)
    print("Get original")
    target_gene_results = get_target_genes(
        receptors,
        len(adata.obs),
        model,
        inputs,
        all_genes,
        rec_uni,
        adata,
        perc=perc,
        threshold=threshold,
        n_shuffles=n_shuffles,
        seed=seed,
        device=device,
    )
    plot_results(target_gene_results, adata)
