import torch

from navsim.agents.ImagineWorld.joint_action_wam import (
    JointActionWAMCore,
    SE2TokenAligner,
    joint_action_wam_loss,
    transform_trajectories,
)


def _pairwise_pose(global_pose: torch.Tensor) -> torch.Tensor:
    """Build source-in-receiver poses for the test scenes."""
    receiver = global_pose[:, :, None]
    source = global_pose[:, None, :]
    delta = source[..., :2] - receiver[..., :2]
    yaw = receiver[..., 2]
    x = torch.cos(yaw) * delta[..., 0] + torch.sin(yaw) * delta[..., 1]
    y = -torch.sin(yaw) * delta[..., 0] + torch.cos(yaw) * delta[..., 1]
    heading = source[..., 2] - receiver[..., 2]
    return torch.stack((x, y, heading), dim=-1)


def test_transform_trajectories_uses_all_source_actions():
    trajectories = torch.tensor([[[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]])
    relative_pose = torch.tensor(
        [[[[0.0, 0.0, 0.0], [2.0, 0.0, torch.pi / 2]],
          [[0.0, -2.0, -torch.pi / 2], [0.0, 0.0, 0.0]]]]
    )
    transformed = transform_trajectories(trajectories, relative_pose)
    torch.testing.assert_close(transformed[0, 0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(
        transformed[0, 0, 1, 0],
        torch.tensor([1.0, 0.0, torch.pi / 2]),
        atol=1e-6,
        rtol=0,
    )


def test_se2_aligner_applies_metric_translation():
    aligner = SE2TokenAligner(grid_size=(3, 3), x_extent=(-1.0, 1.0), y_extent=(-1.0, 1.0))
    bev = torch.zeros(1, 2, 9, 1)
    bev[0, 1, 4, 0] = 1.0
    relative_pose = torch.zeros(1, 2, 2, 3)
    relative_pose[0, 0, 1, 0] = 1.0
    aligned = aligner(bev, relative_pose).reshape(1, 2, 2, 3, 3, 1)
    expected = torch.zeros(3, 3)
    expected[1, 2] = 1.0
    torch.testing.assert_close(aligned[0, 0, 1, :, :, 0], expected)


def test_core_is_agent_permutation_equivariant():
    torch.manual_seed(3)
    model = JointActionWAMCore(
        dim=16,
        status_dim=8,
        num_steps=4,
        num_proposals=3,
        num_heads=4,
        grid_size=(2, 2),
        x_extent=(-5.0, 5.0),
        y_extent=(-5.0, 5.0),
    ).eval()
    bev = torch.randn(2, 3, 4, 16)
    status = torch.randn(2, 3, 8)
    global_pose = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 1.0, 0.2], [-1.0, 3.0, -0.1]]] * 2
    )
    pose = _pairwise_pose(global_pose)
    mask = torch.ones(2, 3, dtype=torch.bool)
    active = torch.tensor([[True, True, False], [True, False, True]])
    agent_type = (~active).long()
    joint_action = torch.randn(2, 3, 4, 3) * active[..., None, None]

    output = model(bev, status, pose, mask, active, agent_type, joint_action)
    permutation = torch.tensor([2, 0, 1])
    permuted_output = model(
        bev[:, permutation],
        status[:, permutation],
        pose[:, permutation][:, :, permutation],
        mask[:, permutation],
        active[:, permutation],
        agent_type[:, permutation],
        joint_action[:, permutation],
    )
    for key in ("fused_bev", "future_bev", "refined_trajectory"):
        torch.testing.assert_close(permuted_output[key], output[key][:, permutation], atol=2e-5, rtol=2e-5)


def test_neighbor_action_changes_predicted_future():
    torch.manual_seed(7)
    model = JointActionWAMCore(
        dim=16,
        status_dim=8,
        num_steps=4,
        num_proposals=2,
        num_heads=4,
        grid_size=(2, 2),
        x_extent=(-5.0, 5.0),
        y_extent=(-5.0, 5.0),
    ).eval()
    bev = torch.randn(1, 2, 4, 16)
    status = torch.randn(1, 2, 8)
    pose = _pairwise_pose(torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]))
    mask = torch.ones(1, 2, dtype=torch.bool)
    active = torch.ones(1, 2, dtype=torch.bool)
    agent_type = torch.zeros(1, 2, dtype=torch.long)
    action_a = torch.zeros(1, 2, 4, 3)
    action_b = action_a.clone()
    action_b[0, 1, :, 0] = torch.arange(1, 5)

    future_a = model(bev, status, pose, mask, active, agent_type, action_a)["future_bev"]
    future_b = model(bev, status, pose, mask, active, agent_type, action_b)["future_bev"]
    assert not torch.allclose(future_a[:, 0], future_b[:, 0])


def test_passive_node_observes_but_is_not_planned_and_loss_backpropagates():
    torch.manual_seed(11)
    model = JointActionWAMCore(
        dim=8,
        status_dim=8,
        num_steps=3,
        num_proposals=2,
        num_heads=2,
        grid_size=(2, 2),
        x_extent=(-5.0, 5.0),
        y_extent=(-5.0, 5.0),
    )
    bev = torch.randn(1, 3, 4, 8)
    status = torch.randn(1, 3, 8)
    pose = _pairwise_pose(torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]]))
    mask = torch.ones(1, 3, dtype=torch.bool)
    active = torch.tensor([[True, True, False]])
    agent_type = torch.tensor([[0, 0, 1]])
    output = model(bev, status, pose, mask, active, agent_type)
    assert output["future_bev"][0, 2].abs().sum() > 0
    assert output["refined_trajectory"][0, 2].count_nonzero() == 0

    target = torch.randn(1, 3, 3, 3) * active[..., None, None]
    losses = joint_action_wam_loss(output, target, target_future_bev=torch.randn_like(output["future_bev"]))
    sum(losses.values()).backward()
    assert model.world_model.action_encoder[0].weight.grad.abs().sum() > 0
