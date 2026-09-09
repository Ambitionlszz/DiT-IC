import ast
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import SanaTransformer2DModel

from models.DiT_IC import merge_codec_subset
from models.sana_self_flow import (
    SelfFlowTeacher,
    TokenQualityConditioner,
    mix_dual_rate_tokens,
    quality_conditioned_sana_forward,
    restrict_mask_to_asymmetric_samples,
)
from self_flow_training import (
    all_token_representation_loss,
    resolve_self_flow_checkpoint_contract,
    sample_iid_uniform_pairs,
    sample_rate_anchored_pairs,
    sample_reflected_adjacent_pairs,
    select_post_burnin_best,
    self_flow_schedule,
    validate_self_flow_config,
)
from train_nogan_ddp import load_config


ROOT = Path(__file__).resolve().parents[1]


def tiny_sana():
    model = SanaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=3,
        num_cross_attention_heads=2,
        cross_attention_head_dim=8,
        cross_attention_dim=16,
        caption_channels=12,
        sample_size=4,
        patch_size=1,
    )
    return model.eval()


def test_reflected_adjacent_sampler_contract():
    generator = torch.Generator().manual_seed(1123)
    quality_a, quality_b = sample_reflected_adjacent_pairs(
        600_000, 6, generator
    )
    assert torch.all((quality_a - quality_b).abs() <= 1)
    same_fraction = (quality_a == quality_b).float().mean().item()
    assert abs(same_fraction - 1 / 6) < 0.003
    for values in (quality_a, quality_b):
        marginal = torch.bincount(values, minlength=6).float()
        marginal /= marginal.sum()
        assert torch.allclose(
            marginal, torch.full((6,), 1 / 6), atol=0.003, rtol=0
        )


def test_iid_uniform_sampler_covers_all_ordered_pairs():
    generator = torch.Generator().manual_seed(1123)
    quality_a, quality_b = sample_iid_uniform_pairs(
        720_000, 6, generator
    )
    pair_distribution = torch.bincount(
        quality_a * 6 + quality_b, minlength=36
    ).float()
    pair_distribution /= pair_distribution.sum()
    assert torch.allclose(
        pair_distribution, torch.full((36,), 1 / 36), atol=0.001, rtol=0
    )
    same_fraction = (quality_a == quality_b).float().mean().item()
    assert abs(same_fraction - 1 / 6) < 0.002
    for values in (quality_a, quality_b):
        marginal = torch.bincount(values, minlength=6).float()
        marginal /= marginal.sum()
        assert torch.allclose(
            marginal, torch.full((6,), 1 / 6), atol=0.002, rtol=0
        )


def test_rate_anchored_sampler_is_exact_and_preserves_subset_marginals():
    generator = torch.Generator().manual_seed(7719)
    quality_a, quality_b, asymmetric = sample_rate_anchored_pairs(
        720_000, 6, 0.5, generator
    )
    assert asymmetric.dtype == torch.bool
    assert asymmetric.sum().item() == 360_000
    anchors = ~asymmetric
    assert torch.equal(quality_a[anchors], quality_b[anchors])

    anchor_distribution = torch.bincount(
        quality_a[anchors], minlength=6
    ).float()
    anchor_distribution /= anchor_distribution.sum()
    assert torch.allclose(
        anchor_distribution, torch.full((6,), 1 / 6), atol=0.002, rtol=0
    )

    mixed_pair_distribution = torch.bincount(
        quality_a[asymmetric] * 6 + quality_b[asymmetric], minlength=36
    ).float()
    mixed_pair_distribution /= mixed_pair_distribution.sum()
    assert torch.allclose(
        mixed_pair_distribution,
        torch.full((36,), 1 / 36),
        atol=0.001,
        rtol=0,
    )


def test_codec_subset_merge_reuses_anchor_view_exactly():
    reference_mean = torch.arange(12, dtype=torch.float32).reshape(4, 3, 1, 1)
    reference_y = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1, 1)
    reference_z = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)
    reference = {
        "mean": reference_mean.clone(),
        "likelihoods": {"y": reference_y.clone(), "z": reference_z.clone()},
    }
    subset = {
        "mean": torch.full((2, 3, 1, 1), 101.0),
        "likelihoods": {
            "y": torch.full((2, 2, 1, 1), 102.0),
            "z": torch.full((2, 1, 1, 1), 103.0),
        },
    }
    asymmetric = torch.tensor([False, True, False, True])

    merged = merge_codec_subset(reference, subset, asymmetric)

    assert torch.equal(merged["mean"][~asymmetric], reference_mean[~asymmetric])
    assert torch.equal(merged["mean"][asymmetric], subset["mean"])
    for name, reference_value in (("y", reference_y), ("z", reference_z)):
        assert torch.equal(
            merged["likelihoods"][name][~asymmetric],
            reference_value[~asymmetric],
        )
        assert torch.equal(
            merged["likelihoods"][name][asymmetric],
            subset["likelihoods"][name],
        )
    assert torch.equal(reference["mean"], reference_mean)


def test_rate_anchor_mask_never_mixes_legal_samples():
    token_mask = torch.ones(4, 1, 2, 2, dtype=torch.bool)
    asymmetric = torch.tensor([False, True, False, True])
    restricted = restrict_mask_to_asymmetric_samples(token_mask, asymmetric)
    assert not restricted[~asymmetric].any()
    assert restricted[asymmetric].all()


def test_rate_anchored_checkpoint_contract_is_distinct():
    direct = resolve_self_flow_checkpoint_contract({})
    anchored = resolve_self_flow_checkpoint_contract(
        {"training_distribution": "rate_anchored"}
    )
    assert anchored["schema_version"] > direct["schema_version"]
    assert anchored["stage"] != direct["stage"]
    assert anchored["mechanism"] != direct["mechanism"]


def test_representation_gradient_excludes_legal_anchor_samples():
    student = torch.randn(4, 7, 5, requires_grad=True)
    teacher = torch.randn(2, 7, 5, requires_grad=True)
    asymmetric = torch.tensor([False, True, False, True])
    loss = all_token_representation_loss(student[asymmetric], teacher)
    loss.backward()
    assert torch.count_nonzero(student.grad[~asymmetric]).item() == 0
    assert torch.count_nonzero(student.grad[asymmetric]).item() > 0
    assert teacher.grad is None


def test_token_mask_selects_all_channels_and_quality_source():
    view_a = torch.zeros(2, 4, 2, 2)
    view_b = torch.ones_like(view_a)
    quality_a = torch.tensor([0, 4])
    quality_b = torch.tensor([1, 5])
    mask = torch.tensor(
        [
            [[[False, True], [True, False]]],
            [[[True, False], [False, True]]],
        ]
    )
    mixed, quality_map = mix_dual_rate_tokens(
        view_a, view_b, quality_a, quality_b, mask
    )
    assert torch.equal(mixed, mask.expand_as(mixed).to(mixed.dtype))
    assert torch.equal(
        quality_map,
        torch.where(
            mask[:, 0],
            quality_b[:, None, None],
            quality_a[:, None, None],
        ),
    )


def test_zero_quality_modulation_is_exact_stock_sana():
    torch.manual_seed(7)
    model = tiny_sana()
    conditioner = TokenQualityConditioner(6, 16, 3, embedding_dim=8)
    latent = torch.randn(2, 4, 4, 4)
    condition = torch.randn(2, 3, 12)
    timestep = torch.ones(2)
    quality_map = torch.randint(0, 6, (2, 4, 4))
    with torch.no_grad():
        stock = model(
            latent, condition, timestep, return_dict=False
        )[0]
        conditioned, _ = quality_conditioned_sana_forward(
            model,
            latent,
            condition,
            timestep,
            quality_map,
            conditioner,
            conditioning_scale=0.0,
        )
    assert torch.equal(stock, conditioned)


def test_token_quality_modulation_can_change_sana_output():
    torch.manual_seed(11)
    model = tiny_sana()
    conditioner = TokenQualityConditioner(6, 16, 3, embedding_dim=8)
    nn.init.normal_(conditioner.block_projection.weight, std=0.01)
    nn.init.normal_(conditioner.output_projection.weight, std=0.01)
    latent = torch.randn(1, 4, 4, 4)
    condition = torch.randn(1, 3, 12)
    timestep = torch.ones(1)
    with torch.no_grad():
        low, _ = quality_conditioned_sana_forward(
            model,
            latent,
            condition,
            timestep,
            torch.zeros(1, 4, 4, dtype=torch.long),
            conditioner,
        )
        high, _ = quality_conditioned_sana_forward(
            model,
            latent,
            condition,
            timestep,
            torch.full((1, 4, 4), 5, dtype=torch.long),
            conditioner,
        )
    assert not torch.allclose(low, high)


def test_representation_loss_uses_every_token_and_stops_teacher_gradient():
    student = torch.randn(2, 7, 5, requires_grad=True)
    teacher = torch.randn(2, 7, 5, requires_grad=True)
    loss = all_token_representation_loss(student, teacher)
    loss.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad).item() > 0
    assert teacher.grad is None


def test_teacher_is_truncated_frozen_and_serializes_only_ema_state():
    model = tiny_sana()
    text_projection = nn.Linear(7, 12)
    conditioner = TokenQualityConditioner(6, 16, 3, embedding_dim=8)
    teacher = SelfFlowTeacher(
        model, text_projection, conditioner, target_layer=1
    )
    assert len(teacher.transformer_blocks) == 2
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    assert all(
        "lora_" in name
        or name.startswith("text_projection.")
        or name.startswith("quality_conditioner.")
        for name in teacher.get_ema_state_dict()
    )


def test_full_student_decode_invokes_sana_once_and_ema_follows_optimizer_step():
    model_source = (ROOT / "models" / "DiT_IC.py").read_text()
    model_tree = ast.parse(model_source)
    codec_class = next(
        node
        for node in model_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Codec"
    )
    methods = {
        node.name: ast.get_source_segment(model_source, node)
        for node in codec_class.body
        if isinstance(node, ast.FunctionDef)
    }
    assert methods["_decode_self_flow_student"].count(
        "self._self_flow_sana_prediction("
    ) == 1

    trainer_source = (ROOT / "train_nogan_ddp.py").read_text()
    trainer_tree = ast.parse(trainer_source)
    train_function = next(
        node
        for node in trainer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "do_train"
    )
    train_source = ast.get_source_segment(trainer_source, train_function)
    optimizer_step = train_source.index("opt.step()")
    teacher_update = train_source.index(
        "self_flow_teacher.update_from_student(", optimizer_step
    )
    assert optimizer_step < teacher_update


def test_warmup_and_ramp_schedule_are_pre_registered():
    config = {
        "student_input": "mixed",
        "conditioning_enabled": True,
        "uniform_warmup_steps": 2000,
        "ramp_steps": 5000,
        "mask_ratio_target": 0.25,
    }
    assert self_flow_schedule(1999, config)["mask_ratio"] == 0
    assert self_flow_schedule(2000, config)["ramp_progress"] == 0
    midpoint = self_flow_schedule(4500, config)
    assert midpoint["mask_ratio"] == 0.125
    assert midpoint["conditioning_scale"] == 0.5
    final = self_flow_schedule(7000, config)
    assert final["mask_ratio"] == 0.25
    assert final["representation_scale"] == 1.0


def test_best_checkpoint_selection_ignores_pre_burnin_metrics():
    best_loss = float("inf")
    best_loss, is_best = select_post_burnin_best(
        0.2, best_loss, train_step=20_000, max_steps=100_000
    )
    assert best_loss == float("inf")
    assert not is_best

    best_loss, is_best = select_post_burnin_best(
        0.3, best_loss, train_step=72_000, max_steps=100_000
    )
    assert best_loss == 0.3
    assert is_best

    best_loss, is_best = select_post_burnin_best(
        0.4, best_loss, train_step=74_000, max_steps=100_000
    )
    assert best_loss == 0.3
    assert not is_best


def test_ramp_validation_uses_current_conditioning_scale():
    trainer_source = (ROOT / "train_nogan_ddp.py").read_text()
    assert "evaluation_conditioning_scale = (" in trainer_source
    assert "conditioning_scale=evaluation_conditioning_scale" in trainer_source
    model_source = (ROOT / "models" / "DiT_IC.py").read_text()
    assert "conditioning_scale=conditioning_scale" in model_source


def test_full_config_uses_iid_sampling_and_100k_budget():
    config = load_config(
        str(ROOT / "configs" / "dual_rate_self_flow_full.yaml")
    )
    assert config["model"]["params"]["self_flow_params"]["sampling"] == "iid_uniform"
    assert config["quality_control"]["sampling"] == "iid_uniform_per_sample"
    assert config["train"]["max_steps"] == 100_000
    assert config["optimizer"]["lr"] == 1e-4
    assert config["optimizer"]["dit_lr"] == 1e-4
    assert config["optimizer"]["condition_lr"] == 1e-4
    assert config["optimizer"]["vae_lr"] == 1e-5
    assert config["optimizer"]["step_lr"] == [60_000, 85_000, 95_000]
    assert config["optimizer"]["step_lr_gammas"] == [0.5, 0.5, 0.5]
    trainer_source = (ROOT / "train_nogan_ddp.py").read_text()
    assert '"step_lr_gammas", [0.5, 0.2, 0.1]' in trainer_source


def test_rate_anchored_config_only_changes_training_distribution():
    direct = load_config(
        str(ROOT / "configs" / "dual_rate_self_flow_full.yaml")
    )
    anchored = load_config(
        str(ROOT / "configs" / "rate_anchored_self_flow_full.yaml")
    )
    validated = validate_self_flow_config(anchored)
    assert validated["training_distribution"] == "rate_anchored"
    assert validated["legal_anchor_ratio"] == 0.5
    assert anchored["data"] == direct["data"]
    assert anchored["quality_control"] == direct["quality_control"]
    assert anchored["optimizer"] == direct["optimizer"]

    direct_params = dict(direct["model"]["params"])
    anchored_params = dict(anchored["model"]["params"])
    direct_self_flow = dict(direct_params.pop("self_flow_params"))
    anchored_self_flow = dict(anchored_params.pop("self_flow_params"))
    assert anchored_params == direct_params
    assert anchored_self_flow.pop("training_distribution") == "rate_anchored"
    assert anchored_self_flow.pop("legal_anchor_ratio") == 0.5
    assert anchored_self_flow == direct_self_flow

    direct_train = dict(direct["train"])
    anchored_train = dict(anchored["train"])
    assert anchored_train.pop("exp_name") == "rate_anchored_self_flow_full"
    direct_train.pop("exp_name")
    assert anchored_train == direct_train


def test_rate_anchored_trainer_uses_one_student_and_teacher_subbatch():
    trainer_source = (ROOT / "train_nogan_ddp.py").read_text()
    trainer_tree = ast.parse(trainer_source)
    train_function = next(
        node
        for node in trainer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "do_train"
    )
    train_source = ast.get_source_segment(trainer_source, train_function)
    assert train_source.count("output_image, self_flow_output = model(") == 1
    assert '"asymmetric_samples": asymmetric_samples' in train_source
    assert (
        'self_flow_output["teacher_latent"][representation_samples]'
        in train_source.replace("\n", "").replace(" ", "")
    )
    assert (
        'self_flow_output["student_feature"][representation_samples]'
        in train_source.replace("\n", "").replace(" ", "")
    )


def test_ablation_configs_share_every_non_mechanism_training_setting():
    names = (
        "full",
        "posttrain_control",
        "q_only",
        "mix_only",
        "distill_only",
        "same_rate_teacher",
    )
    configs = {
        name: load_config(
            str(ROOT / "configs" / f"dual_rate_self_flow_{name}.yaml")
        )
        for name in names
    }
    reference = configs["full"]
    invariant_paths = (
        ("data",),
        ("model", "params", "codec_path"),
        ("model", "params", "DiT_params"),
        ("model", "params", "vae_params"),
        ("model", "params", "codec_param"),
        ("model", "params", "loss_params"),
        ("quality_control",),
        ("optimizer",),
    )

    def resolve(config, path):
        value = config
        for key in path:
            value = value[key]
        return value

    for name, config in configs.items():
        for path in invariant_paths:
            assert resolve(config, path) == resolve(reference, path), (name, path)
        reference_train = dict(reference["train"])
        candidate_train = dict(config["train"])
        reference_train.pop("exp_name")
        candidate_train.pop("exp_name")
        assert candidate_train == reference_train
