import inspect

import torch

from navsim.agents.ImagineWorld.open_observer_wam import (
    OpenObserverWAM,
    ObserverBlindDynamics,
    action_control_errors,
    action_sensitivity_gap,
    build_world_targets,
    canonical_to_local,
    local_to_canonical,
    observer_subset_masks,
    occupancy_scores,
    open_observer_wam_loss,
    select_trajectory,
    warp_local_bev_to_canonical,
    world_metrics,
)
from navsim.planning.training.open_observer_dataset import (
    SPDOpenObserverDataset,
    vla_base_scene_split,
)


EXTENT = (-1.0, 1.0)


def _model() -> OpenObserverWAM:
    return OpenObserverWAM(
        dim=8,
        num_steps=4,
        grid_size=(3, 3),
        x_extent=EXTENT,
        y_extent=EXTENT,
        num_types=2,
        num_domains=2,
    )


def _batch(num_nodes: int = 3):
    torch.manual_seed(0)
    bev = torch.randn(2, num_nodes, 8, 3, 3)
    pose = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.2]],
         [[0.0, 0.0, 0.0], [-1.0, 0.2, -0.4], [0.4, -0.6, 0.3]]]
    )[:, :num_nodes]
    mask = torch.ones(2, num_nodes, dtype=torch.bool)
    agent_type = torch.zeros(2, num_nodes, dtype=torch.long)
    domain_type = torch.zeros(2, num_nodes, dtype=torch.long)
    action = torch.randn(2, 4, 3)
    return bev, pose, mask, agent_type, domain_type, action


def test_point_transforms_are_inverses_under_rotation_and_translation():
    pose = torch.tensor([1.5, -0.4, 0.7])
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [-0.2, 0.5]])
    canonical = local_to_canonical(points, pose)
    restored = canonical_to_local(canonical, pose)
    torch.testing.assert_close(restored, points, atol=1e-6, rtol=0)
    torch.testing.assert_close(local_to_canonical(points, torch.zeros(3)), points, atol=1e-6, rtol=0)
    rotated = local_to_canonical(torch.tensor([[1.0, 0.0]]), torch.tensor([0.0, 0.0, torch.pi / 2]))
    torch.testing.assert_close(rotated, torch.tensor([[0.0, 1.0]]), atol=1e-6, rtol=0)


def test_warp_places_local_origin_at_the_observer_pose():
    bev = torch.zeros(1, 1, 1, 3, 3)
    bev[0, 0, 0, 1, 1] = 1.0
    identity, visibility = warp_local_bev_to_canonical(bev, torch.zeros(1, 1, 3), EXTENT, EXTENT)
    torch.testing.assert_close(identity[0, 0, 0, 1, 1], torch.tensor(1.0))
    assert visibility[0, 0, 0, 1, 1] > 0.5

    shifted, shifted_visibility = warp_local_bev_to_canonical(
        bev, torch.tensor([[[1.0, 0.0, 0.0]]]), EXTENT, EXTENT
    )
    torch.testing.assert_close(shifted[0, 0, 0, 1, 2], torch.tensor(1.0))
    assert shifted_visibility[0, 0, 0, 1, 0] < 0.5

    front = torch.zeros(1, 1, 1, 3, 3)
    front[0, 0, 0, 1, 2] = 1.0
    rotated, _ = warp_local_bev_to_canonical(front, torch.tensor([[[0.0, 0.0, torch.pi / 2]]]), EXTENT, EXTENT)
    torch.testing.assert_close(rotated[0, 0, 0, 2, 1], torch.tensor(1.0), atol=1e-5, rtol=0)


def test_aggregation_is_invariant_to_observer_order():
    model = _model().eval()
    bev, pose, mask, agent_type, domain_type, action = _batch()
    mask[:, 2] = False
    output = model(bev, pose, mask, agent_type, domain_type, action)
    order = torch.tensor([2, 0, 1])
    permuted = model(
        bev[:, order],
        pose[:, order],
        mask[:, order],
        agent_type[:, order],
        domain_type[:, order],
        action,
    )
    for key in ("state", "occupancy", "future_occupancy", "future_flow", "refined_trajectory"):
        torch.testing.assert_close(permuted[key], output[key], atol=1e-5, rtol=1e-5)


def test_dynamics_interface_ignores_observer_identity():
    signature = inspect.signature(ObserverBlindDynamics.forward)
    assert set(signature.parameters) == {"self", "state", "ego_action", "map_context"}
    assert OpenObserverWAM(dim=8, num_steps=4, grid_size=(3, 3)).dynamics_inputs == (
        "occupancy",
        "flow",
        "ego_action",
        "map_context",
    )
    model = _model().eval()
    forbidden = ("type", "domain", "observer", "weather", "agent")
    assert all(all(word not in name for word in forbidden) for name, _ in model.dynamics.named_parameters())

    bev, pose, mask, agent_type, domain_type, action = _batch()
    state = model.observe(bev, pose, mask, agent_type, domain_type)["state"]
    first = model.dynamics(state, action)
    typed = agent_type.clone()
    typed[:, 1] = 1
    other_state = model.observe(bev, pose, mask, typed, domain_type)["state"]
    assert not torch.allclose(state, other_state)
    second = model.dynamics(state, action)
    torch.testing.assert_close(first["future_occupancy"], second["future_occupancy"])
    shuffled = model.dynamics(state, torch.flip(action, dims=(1,)))
    assert not torch.allclose(first["future_occupancy"], shuffled["future_occupancy"])


def test_forward_feeds_dynamics_the_structured_state_not_the_fused_latent():
    model = _model().eval()
    bev, pose, mask, agent_type, domain_type, action = _batch()
    seen = {}
    original = model.dynamics.forward

    def wrapped(state, ego_action, map_context=None):
        seen["state"] = state.detach().clone()
        return original(state, ego_action, map_context)

    model.dynamics.forward = wrapped
    output = model(bev, pose, mask, agent_type, domain_type, action, rollout_proposal=False)
    expected = model.canonical_state(output["occupancy"], output["flow"])
    torch.testing.assert_close(seen["state"], expected, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(seen["state"], output["state"])


def test_subset_loss_and_disagreement_backpropagate():
    model = _model()
    bev, pose, mask, agent_type, domain_type, action = _batch(num_nodes=2)
    query = torch.zeros(2, dtype=torch.long)
    full, subset = observer_subset_masks(mask, query, "query_only")
    assert full[:, 0].all() and subset[:, 0].all()
    assert subset[:, 1].count_nonzero() == 0
    primary = model(bev, pose, full, agent_type, domain_type, action)
    alternate = model(bev, pose, subset, agent_type, domain_type, action)
    trajectories = torch.zeros(2, 2, 4, 3)
    trajectories[:, 0, :, 0] = torch.linspace(0.2, 0.8, 4)
    target = build_world_targets(pose, trajectories, mask, (3, 3), EXTENT, EXTENT)
    target["ego_action"] = action
    losses = open_observer_wam_loss(primary, target, alternate)
    sum(losses.values()).backward()
    assert model.dynamics.action_encoder[0].weight.grad.abs().sum() > 0
    assert model.adapter.type_embed.weight.grad.abs().sum() > 0
    gap = action_sensitivity_gap(
        primary["future_occupancy"],
        torch.flip(primary["future_occupancy"], dims=(1,)),
        target["future_occupancy"],
    )
    assert torch.isfinite(gap)


def test_unobserved_cells_learn_higher_uncertainty():
    torch.manual_seed(1)
    model = _model()
    bev = torch.randn(1, 1, 8, 3, 3)
    pose = torch.tensor([[[1.0, 0.0, 0.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    agent_type = torch.zeros(1, 1, dtype=torch.long)
    domain_type = agent_type.clone()
    optimizer = torch.optim.Adam(model.aggregator.uncertainty_head.parameters(), lr=0.2)
    for _ in range(40):
        optimizer.zero_grad()
        observed = model.observe(bev, pose, mask, agent_type, domain_type)
        visible = observed["visibility"] > 0.5
        target = torch.where(visible, torch.full_like(observed["log_var"], -2.0), torch.full_like(observed["log_var"], 2.0))
        loss = torch.nn.functional.smooth_l1_loss(observed["log_var"], target)
        loss.backward()
        optimizer.step()
    observed = model.observe(bev, pose, mask, agent_type, domain_type)
    visible = observed["visibility"] > 0.5
    assert visible.any() and (~visible).any()
    assert observed["log_var"][~visible].mean() > observed["log_var"][visible].mean()


def test_base_scenes_keep_every_weather_replay_together():
    assert vla_base_scene_split(1) == "train"
    assert vla_base_scene_split(7) == "val"
    assert vla_base_scene_split(9) == "test"


def test_winner_take_all_uses_the_teacher_and_scores_at_inference():
    proposals = torch.zeros(1, 3, 2, 3)
    proposals[0, 1, :, 0] = 1.0
    teacher = proposals[0, 1].unsqueeze(0)
    logits = torch.tensor([[5.0, 0.0, 0.0]])
    _, teacher_index = select_trajectory(proposals, logits, teacher)
    _, score_index = select_trajectory(proposals, logits, None)
    assert int(teacher_index) == 1
    assert int(score_index) == 0


def test_oracle_rollout_ignores_the_observer_set_and_residual_starts_at_zero():
    model = _model().eval()
    bev, pose, mask, agent_type, domain_type, action = _batch()
    occupancy = torch.zeros(2, 1, 3, 3)
    occupancy[:, :, 1, 1] = 1
    flow = torch.zeros(2, 2, 3, 3)
    first = model(bev, pose, mask, agent_type, domain_type, action, gt_occupancy=occupancy, gt_flow=flow)
    second = model(bev + 3, pose.roll(1, 1), mask, agent_type, domain_type, action, gt_occupancy=occupancy, gt_flow=flow)
    torch.testing.assert_close(first["oracle_future_occupancy"], second["oracle_future_occupancy"])
    assert torch.count_nonzero(model.residual(first["state"], action)) == 0
    metrics = world_metrics(first, {
        "occupancy": occupancy,
        "future_occupancy": occupancy[:, None].expand(-1, 4, -1, -1, -1).contiguous(),
        "future_flow": flow[:, None].expand(-1, 4, -1, -1, -1).contiguous(),
        "ego_action": action,
    })
    assert torch.isfinite(metrics["future_iou_acc"])
    assert "oracle_future_iou_acc" in metrics


def test_actor_loss_reaches_the_actor_head():
    model = _model()
    bev, pose, mask, agent_type, domain_type, action = _batch(num_nodes=2)
    output = model(bev, pose, mask, agent_type, domain_type, action)
    target = build_world_targets(pose, torch.zeros(2, 2, 4, 3), mask, (3, 3), EXTENT, EXTENT)
    target["ego_action"] = action
    target["actor_box"] = torch.zeros(2, 16, 7)
    target["actor_box"][:, 0, 0] = 1
    target["actor_mask"] = torch.zeros(2, 16)
    target["actor_mask"][:, 0] = 1
    target["actor_class"] = torch.zeros(2, 16, dtype=torch.long)
    target["future_actor_box"] = torch.zeros(2, 4, 16, 7)
    target["future_actor_mask"] = target["actor_mask"][:, None].expand(-1, 4, -1).float()
    losses = open_observer_wam_loss(output, target, None)
    losses["actor_loss"].backward()
    assert model.actor_head.box[-1].weight.grad.abs().sum() > 0


def test_spd_sample_keeps_the_rsu_as_an_observer_only():
    root = "/mnt/nvme1n1/LCX2/nyc/data/V2X-Seq-SPD"
    train = SPDOpenObserverDataset(root, split="train", limit=1)
    assert len(train) == 1
    features, targets = train[0]
    assert int(features["agent_type"][1]) == 1
    assert bool(features["query_mask"][0]) and not bool(features["query_mask"][1])
    assert not bool(features["agent_mask"][2])
    assert tuple(features["future_time_s"].tolist()) == (0.5, 1.0, 1.5, 2.0, 2.5)
    assert int(targets["actor_role"][0]) == 1
    assert int((targets["actor_role"][targets["actor_mask"] > 0] == 1).sum()) == 1
    assert targets["trajectory"].shape[-2] == 5


def test_metric_names_match_correct_wrong_and_ego_only_edits():
    horizon = 2
    target_occ = torch.zeros(1, horizon, 1, 4, 4)
    target_occ[:, :, :, 1, 1] = 1
    target_occ[:, :, :, 2, 2] = 1
    known = torch.ones_like(target_occ)
    correct_logits = torch.where(target_occ > 0.5, torch.full_like(target_occ, 8.0), torch.full_like(target_occ, -8.0))
    wrong_logits = -correct_logits
    correct_recall, correct_iou = occupancy_scores(correct_logits[:, 0], target_occ[:, 0], known[:, 0])
    wrong_recall, wrong_iou = occupancy_scores(wrong_logits[:, 0], target_occ[:, 0], known[:, 0])
    assert correct_iou > 0.99 and wrong_iou < 0.01
    assert correct_recall > wrong_recall

    gt = torch.zeros(1, horizon, 2, 7)
    gt[:, :, 0, 0] = torch.tensor([0.0, 1.0])
    gt[:, :, 1, 0] = torch.tensor([4.0, 6.0])
    gt[:, :, :, 5] = 4.0
    gt[:, :, :, 6] = 2.0
    prediction = {
        "occupancy": correct_logits[:, 0],
        "future_occupancy": correct_logits,
        "future_flow": torch.zeros(1, horizon, 2, 4, 4),
        "actor_box": gt[:, 0],
        "future_actor_box": gt.clone(),
    }
    target = {
        "occupancy": target_occ[:, 0],
        "future_occupancy": target_occ,
        "future_flow": torch.zeros(1, horizon, 2, 4, 4),
        "occupancy_known": known[:, 0],
        "future_occupancy_known": known,
        "actor_box": gt[:, 0],
        "actor_mask": torch.tensor([[1.0, 1.0]]),
        "actor_role": torch.tensor([[1, 0]]),
        "future_actor_box": gt,
        "future_actor_mask": torch.ones(1, horizon, 2),
    }
    correct = world_metrics(prediction, target)
    assert correct["future_actor_ade_acc"] < 1e-5
    assert "plan_ade_acc" not in correct
    assert "actor_ade_acc" not in correct
    moved = {**prediction, "future_actor_box": gt.clone()}
    moved["future_actor_box"][:, :, 0, 0] += 5
    moved["future_actor_box"][:, :, 1, 0] += 3
    wrong = world_metrics(moved, target)
    assert wrong["future_actor_ade_acc"] > correct["future_actor_ade_acc"]
    ego_only = {**prediction, "future_actor_box": gt.clone()}
    ego_only["future_actor_box"][:, :, 0, 0] += 4
    ego_metrics = world_metrics(ego_only, target)
    assert ego_metrics["nonego_future_actor_ade_acc"] < 1e-5
    assert ego_metrics["future_actor_ade_acc"] > 1.0
    flow = torch.zeros(1, horizon, 2, 4, 4)
    flow[:, :, 0, 1, 1] = 2
    region = torch.zeros(1, horizon, 1, 4, 4)
    region[:, :, :, 1, 1] = 1
    controls = action_control_errors(flow, flow + 3, flow + 1, flow, region)
    assert controls["action_correct_nonego_epe_acc"] < controls["action_perturbed_nonego_epe_acc"]
    assert controls["action_perturbed_nonego_epe_acc"] < controls["action_removed_nonego_epe_acc"]


def test_actor_raster_follows_the_canonical_pose():
    pose = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    trajectories = torch.zeros(1, 2, 2, 3)
    valid = torch.tensor([[True, True]])
    target = build_world_targets(pose, trajectories, valid, (3, 3), EXTENT, EXTENT)
    assert target["occupancy"][0, 0, 1, 1] == 1
    assert target["occupancy"][0, 0, 1, 2] == 1
    assert target["future_occupancy"].shape == (1, 2, 1, 3, 3)
