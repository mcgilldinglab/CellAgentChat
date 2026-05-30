import os

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import seaborn as sns


MESA_TARGET_RANGE = 100.0


def _require_columns(df, required, name):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _load_adata(adata_or_path, copy=True):
    if isinstance(adata_or_path, anndata.AnnData):
        return adata_or_path.copy() if copy else adata_or_path
    if isinstance(adata_or_path, (str, os.PathLike)):
        adata = anndata.read_h5ad(adata_or_path)
        return adata.copy() if copy else adata
    raise TypeError("adata_or_path must be an AnnData object or a path to an .h5ad file")


def _normalize_obs_column(adata, source_label, target_label, required=False, default_value=None):
    if source_label is None:
        if required:
            raise ValueError(f"{target_label} requires a source label")
        if default_value is not None:
            adata.obs[target_label] = default_value
        return

    if source_label not in adata.obs.columns:
        raise ValueError(f"adata.obs is missing required column: {source_label}")

    values = adata.obs[source_label]
    if len(values) != adata.n_obs:
        raise ValueError(f"adata.obs['{source_label}'] has length {len(values)} but expected {adata.n_obs}")
    adata.obs[target_label] = values.to_numpy(copy=True)


def _get_coordinates_from_obsm(adata, coordinates_key):
    if coordinates_key is None:
        return None
    if coordinates_key not in adata.obsm:
        return None
    coords = np.asarray(adata.obsm[coordinates_key], dtype=float)
    if coords.ndim != 2:
        raise ValueError(f"adata.obsm['{coordinates_key}'] must be a 2D array")
    if coords.shape[0] != adata.n_obs:
        raise ValueError(
            f"adata.obsm['{coordinates_key}'] has {coords.shape[0]} rows but expected {adata.n_obs}"
        )
    if coords.shape[1] not in (2, 3):
        raise ValueError(
            f"adata.obsm['{coordinates_key}'] must have 2 or 3 columns, found {coords.shape[1]}"
        )
    return coords


def _scale_coordinates_for_mesa(coords, target_range=MESA_TARGET_RANGE):
    mins = np.min(coords, axis=0)
    shifted = coords - mins
    max_range = float(np.max(np.ptp(coords, axis=0))) if coords.size else 0.0
    if max_range <= 0:
        scaled = np.zeros_like(coords, dtype=float)
    else:
        scaled = shifted * (target_range / max_range)
    return scaled


def _assign_coordinate_columns(adata, coords, scale_for_mesa=True):
    if coords is None:
        return
    coord_names = ["x", "y"] if coords.shape[1] == 2 else ["x", "y", "z"]

    for idx, name in enumerate(coord_names):
        adata.obs[f"{name}_raw"] = coords[:, idx]

    scaled_coords = _scale_coordinates_for_mesa(coords) if scale_for_mesa else coords.copy()
    for idx, name in enumerate(coord_names):
        adata.obs[name] = scaled_coords[:, idx]


def setup_adata(
    adata_or_path,
    coordinates_key="spatial",
    batch_label=None,
    cell_type_label=None,
    scale_for_mesa=True,
    copy=True,
):
    if cell_type_label is None:
        raise ValueError("cell_type_label is mandatory")

    adata = _load_adata(adata_or_path, copy=copy)
    _normalize_obs_column(adata, cell_type_label, "cell_type", required=True)

    if batch_label is None:
        adata.obs["Batch"] = "0"
    else:
        _normalize_obs_column(adata, batch_label, "Batch", required=True)

    coords = _get_coordinates_from_obsm(adata, coordinates_key)
    _assign_coordinate_columns(adata, coords, scale_for_mesa=scale_for_mesa)
    return adata
