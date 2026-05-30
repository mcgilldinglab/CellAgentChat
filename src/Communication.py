import os
import subprocess

import numpy as np
import pandas as pd

import abm as abm
from output_paths import artifact_dir
from bckground_distribution_new import (
    get_interaction_matrix,
    get_significant_lr_pairs,
    pvalues_threshold,
    save,
)


def _ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def _resolve_cci_dir(path):
    return artifact_dir("cci") if path == "out" else path


def _script_path(script_name):
    return os.path.join(os.path.dirname(__file__), script_name)


def _run_r_script(script_name, args):
    script = _script_path(script_name)
    if not os.path.exists(script):
        raise FileNotFoundError(f"Missing plotting script: {script}")
    cmd = ["Rscript", script] + args
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Failed to run {script_name}: {completed.stderr.strip() or completed.stdout.strip()}"
        )


def _plot_outputs(path, step, interaction_matrix, sig_lr_pair, pvalues2_name, cluster_names):
    _run_r_script(
        "heatmaps.R",
        [
            "--interaction_file",
            os.path.join(path, f"{step}_{interaction_matrix}"),
            "--cluster_names",
            os.path.join(path, cluster_names),
            "--out",
            os.path.join(path, f"heatmap_step_{step}.pdf"),
        ],
    )
    _run_r_script(
        "dotplot.R",
        [
            "--lr_file",
            os.path.join(path, f"{step}_new_{sig_lr_pair}"),
            "--pvalues",
            os.path.join(path, f"{step}_new_{pvalues2_name}"),
            "--out",
            os.path.join(path, f"dotplot_step_{step}.pdf"),
        ],
    )


def CCI(
    N,
    adata,
    lig_uni,
    rec_uni,
    rates,
    distribution,
    clusters,
    dist,
    delta=1,
    max_steps=1,
    tau=2,
    rec_block=False,
    plot=True,
    path="out",
    interaction_matrix="interaction_matrix.csv",
    sig_lr_pair="sig_lr_pairs.csv",
    pvalues_name="pvalues.csv",
    pvalues2_name="pvalues2.csv",
    cluster_names="cluster_names.csv",
    threshold=0.05,
    net=None,
):
    path = _resolve_cci_dir(path)
    _ensure_output_dir(path)
    model = abm.CellModel(
        N=N,
        adata=adata,
        lig_uni=lig_uni,
        rec_uni=rec_uni,
        rates=rates,
        max_steps=max_steps,
        dist=dist,
        delta=delta,
        tau=tau,
        rec_block=rec_block,
        net=net,
    )

    print("Calculating Interactions")
    for i in range(max_steps):
        model.step(show_agent_progress=True)

        print("Calculating Significant Interactions")
        lr1 = abm.get_lr_interactions2(model)
        sig_lr, pvalues, num_sig = get_significant_lr_pairs(lr1, distribution, threshold)
        pvalues2 = pvalues_threshold(pvalues)

        plot_clusters = clusters[1:] if clusters and clusters[0] == "Names" else list(clusters)
        dict1 = get_interaction_matrix(plot_clusters, num_sig)

        step = str(i + 1)
        print("Saving Files")
        save(
            sig_lr,
            pvalues,
            pvalues2,
            dict1,
            plot_clusters,
            i + 1,
            path=path,
            interaction_matrix=interaction_matrix,
            sig_lr_pair=sig_lr_pair,
            pvalues_name=pvalues_name,
            pvalues2_name=pvalues2_name,
            cluster_names=cluster_names,
        )

        if plot:
            print("Plotting results")
            _plot_outputs(path, step, interaction_matrix, sig_lr_pair, pvalues2_name, cluster_names)

    return model


def receiving_score(model, path="out"):
    path = _resolve_cci_dir(path)
    _ensure_output_dir(path)
    receiver = [agent.num_r for agent in model.schedule.agents]
    df = pd.DataFrame(
        {
            "Cell": list(model.adata.obs_names),
            "Cell Type": list(model.adata.obs["cell_type"]),
            "Receiving Score": receiver,
        }
    )
    out_file = os.path.join(path, "cell_receiving_scores.csv")
    df.to_csv(out_file)
    print(f"results saved to: {out_file}")
    return df


def plotting(
    path="out",
    step="1",
    interaction_matrix="interaction_matrix.csv",
    sig_lr_pair="sig_lr_pairs.csv",
    pvalues2_name="pvalues2.csv",
    cluster_names="cluster_names.csv",
):
    path = _resolve_cci_dir(path)
    print("Plotting results")
    _plot_outputs(path, str(step), interaction_matrix, sig_lr_pair, pvalues2_name, cluster_names)
    print("Plots saved")
