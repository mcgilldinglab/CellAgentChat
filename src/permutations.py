from collections import defaultdict

import anndata
import numpy as np
from tqdm.auto import tqdm

from abm import CellModel, get_lr_interactions2


def copy(adata):
    return anndata.AnnData(X=adata.X.copy(), obs=adata.obs.copy(), var=adata.var.copy())


def _accumulate_results(fin, res):
    for dic in res.values():
        for key, value in dic.items():
            fin[key].append(value)


def _average_num_scores(fin):
    if not fin:
        return 0
    val = sum(len(values) for values in fin.values())
    return val / len(fin)


def permutation_test(threshold, N, adata, lig_uni, rec_uni, rates, dist=False, tau=2, rec_block=False, seed=1234):
    np.random.seed(seed)

    distance = 1
    if dist:
        model3 = CellModel(
            N,
            adata,
            lig_uni,
            rec_uni,
            rates,
            dist=True,
            delta=1,
            max_steps=1,
            tau=tau,
            rec_block=rec_block,
            permutations=True,
        )
        distance = model3.calc_normalized_dist()

    fin = defaultdict(list)
    avg = 0
    i = 0
    progress = tqdm(desc="Permutations", unit="iter")
    while avg < threshold:
        progress.update(1)
        progress.set_postfix(avg_scores=f"{avg:.2f}")
        bdata = copy(adata)
        x = np.array(bdata.obs["cell_type"])
        np.random.shuffle(x)
        bdata.obs["cell_type"] = x
        model2 = CellModel(
            N,
            bdata,
            lig_uni,
            rec_uni,
            rates,
            dist=False,
            tau=tau,
            max_steps=1,
            rec_block=rec_block,
            permutations=True,
        )
        model2.step()

        res = get_lr_interactions2(model2)
        _accumulate_results(fin, res)
        avg = _average_num_scores(fin)
        i += 1
    progress.close()
    print(f"Average Number of LR Pair Scores:{avg}")
    return dict(fin), model2, distance


