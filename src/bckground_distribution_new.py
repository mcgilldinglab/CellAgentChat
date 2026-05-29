import math
import os

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as multi
from scipy.stats import gamma
from tqdm.auto import tqdm
from output_paths_new import artifact_dir, artifact_path


def _resolve_distribution_file(path):
    if path is None:
        return artifact_path("distributions", "distribution.csv")
    if os.path.isabs(path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return path
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    if path == "distribution.csv":
        return artifact_path("distributions", "distribution.csv")
    return path


def _clean_cluster_list(clusters):
    cluster_list = list(clusters)
    if cluster_list and cluster_list[0] == "Names":
        cluster_list = cluster_list[1:]
    return [cluster for cluster in cluster_list if cluster is not False]


def get_distribution(fin, dist=0, scaled=False, pseudotime=False, shape_factor=0.98, scale_factor=1.2):
    print("Getting background distribution for each ligand-receptor pair")
    fin3 = {}
    for key, alist in tqdm(fin.items(), desc="Fitting LR distributions", unit="pair"):
        if len(alist) == 0:
            continue
        if not pseudotime:
            alpha, loc, scale = gamma.fit(alist, floc=0)
            alpha_adj = alpha * shape_factor
            scale_adj = scale * scale_factor
            if scaled and dist > 0:
                fin3[key] = (alpha_adj / 2, 0, (scale_adj / dist) / 2)
            else:
                fin3[key] = (alpha_adj, 0, scale_adj)
        else:
            # Pseudotime
            alpha, loc, scale = gamma.fit(alist, floc=min(alist) - 0.00000001)
            fin3[key] = (alpha, loc, scale)
    return fin3


def save_distribution(fin, path="distribution.csv"):
    path = _resolve_distribution_file(path)
    pd.DataFrame(fin).to_csv(path, index=False)
    print("Distribution Saved")


def load_distribution(file="distribution.csv"):
    file = _resolve_distribution_file(file)
    distribution = pd.read_csv(file)
    fin = distribution.to_dict("list")
    print("Distribution loaded")
    return fin


def get_significant_lr_pairs(lr1, fin, cutoff=0.05):
    sig_lr = {}
    sig = 0
    unsig = 0
    pvalue = {}
    num_sig = {}
    pvalues2 = []
    valid_pairs = []

    for cluster, d1 in lr1.items():
        for key, val in d1.items():
            if key not in fin:
                continue
            alpha, loc, scale = fin[key]
            pval = 1 - gamma.cdf(val, a=alpha, loc=loc, scale=scale)
            if math.isnan(pval):
                pval = 1
            pvalues2.append(pval)
            valid_pairs.append((cluster, key))

    if not pvalues2:
        print("0 significant interactions")
        print("Percentage of significant interactions: 0.0")
        return sig_lr, pvalue, num_sig

    corr = multi.multipletests(pvalues2, cutoff, method="bonferroni")

    for idx, (cluster, key) in enumerate(valid_pairs):
        cluster_scores = lr1[cluster]
        sig_bucket = sig_lr.setdefault(cluster, {})
        pval_bucket = pvalue.setdefault(cluster, {})
        if corr[0][idx]:
            sig_bucket[key] = cluster_scores[key]
            pval_bucket[key] = corr[1][idx]
            sig += 1
            num_sig[cluster] = num_sig.get(cluster, 0) + 1
        else:
            unsig += 1

    sig_lr = {k: v for k, v in sig_lr.items() if v}
    pvalue = {k: v for k, v in pvalue.items() if v}

    print(f"{sig} significant interactions")
    print(f"Percentage of significant interactions: {sig / len(corr[0]) * 100}")
    return sig_lr, pvalue, num_sig


def pvalues_threshold(pvalue):
    pvalue2 = {}
    for key, dict1 in pvalue.items():
        dict2 = {}
        for key2, val in dict1.items():
            if val > 0.025:
                dict2[key2] = "Small"
            elif val > 0.01:
                dict2[key2] = "Smaller"
            else:
                dict2[key2] = "Smallest"
        pvalue2[key] = dict2
    return pvalue2


def get_interaction_matrix(clusters, num_sig):
    cluster_list = _clean_cluster_list(clusters)
    dict1 = {}
    for clus in cluster_list:
        dict2 = {}
        for clus2 in cluster_list:
            dict2[clus2] = num_sig.get(f"{clus}_{clus2}", 0)
        dict1[clus] = dict2
    return dict1


def save(
    sig_lr,
    pvalue,
    pvalue2,
    dict1,
    cluster_list,
    step,
    path="out",
    interaction_matrix="interaction_matrix.csv",
    sig_lr_pair="sig_lr_pairs.csv",
    pvalues_name="pvalues.csv",
    pvalues2_name="pvalues2.csv",
    cluster_names="cluster_names.csv",
):
    df1 = pd.DataFrame.from_dict(dict1, orient="index").transpose()
    df2 = pd.DataFrame.from_dict(sig_lr, orient="index").fillna(0).transpose()
    df3 = pd.DataFrame.from_dict(pvalue, orient="index").fillna(0).transpose()
    df4 = pd.DataFrame.from_dict(pvalue2, orient="index").fillna(0).transpose()

    cluster_list = _clean_cluster_list(cluster_list)
    df5 = pd.DataFrame(["Names"] + cluster_list)

    if not df2.empty:
        df2["total"] = df2.select_dtypes(np.number).gt(0).sum(axis=1)
        lrs = df2["total"].sort_values(ascending=False).head(25)
        l = list(lrs.index)
        df6 = df2.loc[l, :].drop(columns=["total"], errors="ignore")
        if not df6.empty:
            df6 = df6[df6.sum(0).sort_values(ascending=False)[:25].index]
        cols = df6.columns
        df7 = df3.loc[l, cols] if not df3.empty else pd.DataFrame(index=l, columns=cols)
        df8 = df4.loc[l, cols] if not df4.empty else pd.DataFrame(index=l, columns=cols)
    else:
        df6 = df2.copy()
        df7 = df3.copy()
        df8 = df4.copy()

    path = artifact_dir("cci") if path == "out" else path
    os.makedirs(path, exist_ok=True)

    df1.to_csv(os.path.join(path, f"{step}_{interaction_matrix}"))
    df6.to_csv(os.path.join(path, f"{step}_new_{sig_lr_pair}"))
    df7.to_csv(os.path.join(path, f"{step}_new_{pvalues_name}"))
    df8.to_csv(os.path.join(path, f"{step}_new_{pvalues2_name}"))
    df2.to_csv(os.path.join(path, f"{step}_{sig_lr_pair}"))
    df3.to_csv(os.path.join(path, f"{step}_{pvalues_name}"))
    df4.to_csv(os.path.join(path, f"{step}_{pvalues2_name}"))
    df5.to_csv(os.path.join(path, cluster_names))
    print("Saved files")
