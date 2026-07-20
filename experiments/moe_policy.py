"""Small-scale, reproducible MoE policy-gradient training for synthetic states.

This is a single-node PyTorch DDP demonstration.  It is intentionally not a
foundation-model, LLM-serving, multi-node, or expert-parallel training system.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.synthetic_experiment import _git_revision, load_profile


@dataclass(frozen=True)
class MoeTrainingConfig:
    seed: int = 20260706
    epochs: int = 12
    batch_size: int = 64
    learning_rate: float = 0.003
    entropy_coefficient: float = 0.02
    expert_count: int = 4
    hidden_size: int = 32
    training_examples: int = 768
    capacity_factor: float = 1.25


def _torch_modules():
    try:
        import torch
        from torch.distributed import ReduceOp
        from torch.nn.parallel import DistributedDataParallel
        from torch.utils.data import DataLoader, TensorDataset
        from torch.utils.data.distributed import DistributedSampler
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the MoE demonstration. Install the optional "
            "dependency with `pip install -r requirements-rl.txt`."
        ) from exc
    return torch, ReduceOp, DistributedDataParallel, DataLoader, TensorDataset, DistributedSampler


def build_synthetic_bandit(
    profile: Mapping[str, Any],
    config: MoeTrainingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic contextual-bandit task tied to synthetic profile scale."""

    rng = np.random.default_rng(config.seed)
    size = config.training_examples
    utilization = float(profile["fleet_utilization"])
    nr_probability = float(profile["mean_nr_probability"])
    slot_pressure = min(1.0, float(profile["policy_task_count"]) / 500.0)
    rotation_pressure = min(1.0, float(profile["rotation_count"]) / 200.0)

    states = np.column_stack([
        np.full(size, utilization),
        np.full(size, nr_probability),
        np.full(size, slot_pressure),
        np.full(size, rotation_pressure),
        rng.uniform(0.0, 1.0, size),
        rng.uniform(0.0, 1.0, size),
        rng.normal(0.0, 0.15, size),
        rng.normal(0.0, 0.15, size),
    ]).astype(np.float32)

    risk = 0.45 * states[:, 1] + 0.30 * states[:, 3] + 0.25 * states[:, 4]
    capacity = 0.50 * states[:, 0] + 0.30 * states[:, 2] + 0.20 * states[:, 5]
    noise = rng.normal(0.0, 0.025, (size, 3)).astype(np.float32)
    rewards = np.column_stack([
        0.52 - 0.35 * risk - 0.15 * capacity,
        0.60 + 0.14 * capacity - 0.20 * risk,
        0.57 + 0.30 * risk - 0.22 * capacity,
    ]).astype(np.float32) + noise
    return states, rewards


def _build_model(torch: Any, config: MoeTrainingConfig):
    class SoftMoePolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = torch.nn.Linear(8, config.expert_count)
            self.experts = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(8, config.hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(config.hidden_size, 3),
                )
                for _ in range(config.expert_count)
            ])

        def forward(self, states):
            routing = torch.softmax(self.router(states), dim=-1)
            expert_logits = torch.stack([expert(states) for expert in self.experts], dim=1)
            logits = (routing.unsqueeze(-1) * expert_logits).sum(dim=1)
            return logits, routing

    return SoftMoePolicy()


def _distributed_context(torch: Any) -> tuple[int, int, int, str, bool]:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    distributed = world_size > 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if distributed and not torch.distributed.is_initialized():
        init_method = os.getenv("TORCH_DISTRIBUTED_INIT_METHOD")
        kwargs = (
            {"init_method": init_method, "rank": rank, "world_size": world_size}
            if init_method
            else {}
        )
        torch.distributed.init_process_group(backend=backend, **kwargs)
    return world_size, rank, local_rank, backend, distributed


def _reduce(torch: Any, reduce_op: Any, values: list[float], device: Any, distributed: bool) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed:
        torch.distributed.all_reduce(tensor, op=reduce_op.SUM)
    return [float(value) for value in tensor.cpu().tolist()]


def _gradient_norm(torch: Any, model: Any) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().norm(2).item() ** 2)
    return squared ** 0.5


def train_moe_policy(
    profile: Mapping[str, Any],
    output_dir: Path,
    config: MoeTrainingConfig = MoeTrainingConfig(),
) -> dict[str, Any]:
    """Train a small soft-routed MoE policy and write portable evidence artifacts."""

    torch, reduce_op, ddp, data_loader, tensor_dataset, distributed_sampler = _torch_modules()
    world_size, rank, local_rank, backend, distributed = _distributed_context(torch)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    states, reward_table = build_synthetic_bandit(profile, config)
    dataset = tensor_dataset(torch.from_numpy(states), torch.from_numpy(reward_table))
    sampler = (
        distributed_sampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.seed,
            drop_last=True,
        )
        if distributed
        else None
    )
    loader = data_loader(dataset, batch_size=config.batch_size, shuffle=sampler is None, sampler=sampler)

    if sampler is None:
        shard_has_no_duplicates = True
        shard_size = len(dataset)
    else:
        shard_indices = list(iter(sampler))
        shard_size = len(shard_indices)
        gathered_indices: list[Any] = [None] * world_size
        torch.distributed.all_gather_object(gathered_indices, shard_indices)
        flattened = [index for shard in gathered_indices for index in shard]
        shard_has_no_duplicates = len(flattened) == len(set(flattened))

    model = _build_model(torch, config).to(device)
    learner = ddp(model, device_ids=[local_rank] if torch.cuda.is_available() else None) if distributed else model
    optimizer = torch.optim.Adam(learner.parameters(), lr=config.learning_rate)
    history: list[dict[str, Any]] = []

    for epoch in range(config.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        learner.train()
        totals = [0.0] * (6 + config.expert_count)
        for state_batch, reward_batch in loader:
            state_batch = state_batch.to(device)
            reward_batch = reward_batch.to(device)
            logits, routing = learner(state_batch)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample()
            rewards = reward_batch.gather(1, actions.unsqueeze(1)).squeeze(1)
            advantage = rewards - rewards.mean()
            entropy = distribution.entropy().mean()
            loss = -(advantage.detach() * distribution.log_prob(actions)).mean() - config.entropy_coefficient * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norm = _gradient_norm(torch, learner)
            optimizer.step()

            expected_load = routing.detach().sum(dim=0)
            capacity = config.capacity_factor * len(state_batch) / config.expert_count
            capacity_excess = torch.clamp(expected_load - capacity, min=0.0).sum()
            totals[0] += float(loss.item()) * len(state_batch)
            totals[1] += float(rewards.mean().item()) * len(state_batch)
            totals[2] += float(entropy.item()) * len(state_batch)
            totals[3] += grad_norm * len(state_batch)
            totals[4] += float(capacity_excess.item())
            totals[5] += len(state_batch)
            for expert_index, load in enumerate(expected_load.cpu().tolist()):
                totals[6 + expert_index] += float(load)

        totals = _reduce(torch, reduce_op, totals, device, distributed)
        sample_count = max(totals[5], 1.0)
        expert_loads = [round(value / sample_count, 4) for value in totals[6:]]
        history.append({
            "epoch": epoch + 1,
            "mean_loss": round(totals[0] / sample_count, 6),
            "mean_reward": round(totals[1] / sample_count, 6),
            "policy_entropy": round(totals[2] / sample_count, 6),
            "mean_gradient_norm": round(totals[3] / sample_count, 6),
            "capacity_excess_fraction": round(totals[4] / sample_count, 6),
            "expert_load_fraction": expert_loads,
        })

    learner.eval()
    with torch.no_grad():
        full_states = torch.from_numpy(states).to(device)
        full_rewards = torch.from_numpy(reward_table).to(device)
        logits, routing = learner(full_states)
        greedy_actions = logits.argmax(dim=1)
        validation_reward = full_rewards.gather(1, greedy_actions.unsqueeze(1)).mean()
        router_entropy = -(routing * torch.log(routing.clamp_min(1e-8))).sum(dim=1).mean()
        expert_load = routing.mean(dim=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "moe_policy_checkpoint.pt"
    artifact_path = output_dir / "moe_training.json"
    result = {
        "scope": {
            "label": "Single-node synthetic MoE policy-gradient training demonstration",
            "does_not_implement": [
                "LLM post-training or RLHF",
                "multi-node execution",
                "MoE expert parallelism or all-to-all token dispatch",
                "inference serving in an RL rollout loop",
            ],
        },
        "framework": "PyTorch DistributedDataParallel" if distributed else "PyTorch single-process",
        "distributed": {
            "backend": backend,
            "world_size": world_size,
            "rank_zero_artifact_writer": True,
            "data_sharding": "DistributedSampler" if distributed else "not_applicable",
            "global_metric_reduction": "torch.distributed.all_reduce" if distributed else "not_applicable",
        },
        "configuration": asdict(config),
        "simulator_version": _git_revision(),
        "history": history,
        "validation": {
            "greedy_policy_reward": round(float(validation_reward.item()), 6),
            "router_entropy": round(float(router_entropy.item()), 6),
            "expert_load_fraction": [round(float(value), 6) for value in expert_load.cpu().tolist()],
        },
        "debug_invariants": {
            "finite_metrics": all(np.isfinite(record["mean_loss"]) and np.isfinite(record["mean_reward"]) for record in history),
            "expert_load_sum": round(sum(float(value) for value in expert_load.cpu().tolist()), 6),
            "effective_global_batch_size": config.batch_size * world_size,
            "local_data_shard_size": shard_size,
            "distributed_data_shards_have_no_duplicates": shard_has_no_duplicates,
            "policy_checkpoint_version": _git_revision(),
        },
    }
    if rank == 0:
        torch.save({"model_state_dict": model.state_dict(), "config": asdict(config), "simulator_version": _git_revision()}, checkpoint_path)
        artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        kpis_path = output_dir / "kpis.csv"
        profile_path = output_dir / "synthetic_profile.json"
        if kpis_path.exists() and profile_path.exists():
            import pandas as pd

            from experiments.rl_llm_evaluation import build_evaluation_artifact, write_evaluation_artifact

            comparison_profile = json.loads(profile_path.read_text(encoding="utf-8"))
            evaluation = build_evaluation_artifact(
                output_dir.name,
                pd.read_csv(kpis_path),
                comparison_profile,
                _git_revision(),
                moe_training=result,
            )
            write_evaluation_artifact(output_dir, evaluation)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return result


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _local_ddp_worker(
    local_rank: int,
    world_size: int,
    profile: Mapping[str, Any],
    output_dir: str,
    config: MoeTrainingConfig,
    port: int,
) -> None:
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["TORCH_DISTRIBUTED_INIT_METHOD"] = f"tcp://127.0.0.1:{port}?use_libuv=0"
    train_moe_policy(profile, Path(output_dir), config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the synthetic MoE policy-gradient demonstration.")
    parser.add_argument("--scenario", default="default_run")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--output-dir", default="artifacts/experiments")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--ddp-local-processes", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    profile = load_profile(repo_root / "Data", args.scenario)
    comparison_dir = repo_root / args.output_dir / f"{args.scenario}_comparison_seed{args.seed}"
    config = MoeTrainingConfig(seed=args.seed, epochs=args.epochs)
    if args.ddp_local_processes > 1:
        if os.getenv("WORLD_SIZE"):
            raise ValueError("Use either torchrun or --ddp-local-processes, not both.")
        torch, *_ = _torch_modules()
        torch.multiprocessing.spawn(
            _local_ddp_worker,
            args=(args.ddp_local_processes, profile, str(comparison_dir), config, _free_local_port()),
            nprocs=args.ddp_local_processes,
            join=True,
        )
        result = json.loads((comparison_dir / "moe_training.json").read_text(encoding="utf-8"))
    else:
        result = train_moe_policy(profile, comparison_dir, config)
    if int(os.getenv("RANK", "0")) == 0:
        print(f"Wrote MoE training artifact to {comparison_dir / 'moe_training.json'}")
        print(f"Validation reward: {result['validation']['greedy_policy_reward']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
