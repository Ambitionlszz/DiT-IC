from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from models.bt_self_flow.experiment import optimization_contract
from scripts.launch_bt_c_ablation import CAblationRun, resolved_configs


def test_two_c_runs_share_warmup_and_only_ablate_auxiliary_losses(tmp_path):
    configs = resolved_configs(tmp_path)
    common, keep, remove = (configs[name] for name in ("common", "c_joint", "c_no_latent"))
    assert common["train"]["steps"] == 50000
    assert common["train"]["batch_size"] == 32
    assert common["train"]["accumulation_steps"] == 1
    assert common["train"]["effective_batch"] == 64
    for config in (keep, remove):
        assert config["model"]["backend"] == "C"
        assert config["train"]["steps"] == 200000
        assert config["train"]["effective_batch"] == 8
        assert config["model"]["codec"] == common["model"]["codec"]
        assert config["train"]["rate_weight"] == common["train"]["rate_weight"] == 3.
        assert config["data"]["valid"] == ["/data2/ls/data/Kodak"]
        assert config["train"]["save_every"] == config["train"]["eval_every"] == 2000
    assert remove["image_loss"].pop("base_weight") == 0
    assert remove["image_loss"].pop("latent_weight") == 0
    assert keep["image_loss"].pop("base_weight") == .1
    assert keep["image_loss"].pop("latent_weight") == .1
    remove["train"]["output"] = keep["train"]["output"]
    assert remove == keep


def test_removed_losses_or_changed_rate_weight_cannot_silently_resume(tmp_path):
    config = resolved_configs(tmp_path)["c_joint"]
    for change in ("loss", "rate"):
        changed = deepcopy(config)
        if change == "loss":
            changed["image_loss"].update(base_weight=0., latent_weight=0.)
        else:
            changed["train"]["rate_weight"] = 1.
        assert optimization_contract(config) != optimization_contract(changed)


@pytest.mark.parametrize("fail_first", [False, True])
def test_real_children_share_initialization_and_peer_survives_failure(tmp_path, fail_first):
    (tmp_path/"logs").mkdir()
    shared = tmp_path/"backend_init.pt"
    shared.write_text("one shared initialization")
    run = CAblationRun(tmp_path, ["0", "1"], sys.executable)
    worker = '''
import os, sys, time
from pathlib import Path
directory, shared, fail = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
directory.mkdir(parents=True, exist_ok=True)
(directory/"initialization").write_text(shared)
(directory/"gpu").write_text(os.environ["CUDA_VISIBLE_DEVICES"])
if fail:
    sys.exit(2)
time.sleep(.1)
(directory/"complete.pt").write_text("complete")
'''
    run.command = lambda name, directory, initialization: [
        sys.executable, "-c", worker, str(directory), str(initialization),
        str(int(fail_first and name == "c_joint"))]
    if fail_first:
        with pytest.raises(RuntimeError, match="C experiment failures"):
            run.train_arms(shared)
    else:
        run.train_arms(shared)
    assert not run.children
    state = json.loads((tmp_path/"pipeline.json").read_text())
    for name, directory, gpu in (("c_joint", "C_aux", "0"), ("c_no_latent", "C_no_latent", "1")):
        path = tmp_path/directory/"joint"
        assert (path/"initialization").read_text() == str(shared)
        assert (path/"gpu").read_text() == gpu
        assert state["jobs"][name]["status"] == ("failed" if fail_first and name == "c_joint" else "completed")
    assert (tmp_path/"C_no_latent/joint/complete.pt").exists()
