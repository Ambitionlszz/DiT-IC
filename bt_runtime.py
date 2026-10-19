"""Host-specific storage settings shared by BT training and launchers."""

from copy import deepcopy
import os
from pathlib import Path


def relocate_config(config):
    """Resolve storage overrides before configs are frozen in run manifests.

    Only the bundled V100 data prefix is replaced. Explicit custom dataset
    locations keep their meaning. No optimization or batch setting changes.
    """
    result = deepcopy(config)
    data_root = os.environ.get("BT_DATA_ROOT")
    if data_root:
        root = Path(data_root).expanduser().resolve()
        for split in ("train", "valid"):
            paths = []
            for path in result["data"].get(split, []):
                try:
                    relative = Path(path).relative_to("/data2/ls/data")
                except ValueError:
                    paths.append(path)
                else:
                    paths.append(str(root / relative))
            result["data"][split] = paths
    for variable, key in (("BT_SANA_PATH", "sana_path"), ("BT_TEXT_PATH", "text_path")):
        if os.environ.get(variable):
            result["model"][key] = str(Path(os.environ[variable]).expanduser().resolve())
    return result


def cache_root():
    return Path(os.environ.get("BT_CACHE_ROOT", str(Path.home() / ".cache" / "ditic"))).expanduser()


def configure_caches(env):
    root = cache_root()
    env.setdefault("TORCH_HOME", str(root / "torch"))
    env.setdefault("HF_HOME", str(root / "huggingface"))
