import os


_DEFAULT_ROOT_DIRNAME = "CellAgentChat_outputs"
_SUBDIRS = {
    "models": "models",
    "conversion_rates": "conversion_rates",
    "cci": "cci",
    "perturbation": "perturbation",
    "distributions": "distributions",
}


def _project_root():
    return os.path.dirname(os.path.dirname(__file__))


def artifact_root():
    path = os.path.join(_project_root(), _DEFAULT_ROOT_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def artifact_dir(kind):
    if kind not in _SUBDIRS:
        raise ValueError(f"Unknown artifact kind: {kind}")
    path = os.path.join(artifact_root(), _SUBDIRS[kind])
    os.makedirs(path, exist_ok=True)
    return path


def artifact_path(kind, filename):
    return os.path.join(artifact_dir(kind), filename)
