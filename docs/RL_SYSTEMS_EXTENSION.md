# RL Systems Extension

## Purpose

This public synthetic-data extension makes several RL systems concepts inspectable
without representing them as deployed clinical or foundation-model infrastructure.

## Implemented

- **Verifiable reward audit:** every policy-comparison KPI row receives deterministic
  checks and a reward breakdown. A failed check remains visible rather than being
  hidden behind an aggregate score.
- **Rollout trace contract:** policy version, environment, action, outcome reference,
  and required serving fields are recorded so a future learner can reject stale data.
- **Small MoE policy:** a PyTorch soft router combines four MLP experts while training
  on a seeded synthetic contextual-bandit environment derived from simulator profile
  features.
- **Single-node DDP:** when launched with `--ddp-local-processes 2`, the trainer
  uses `DistributedSampler`, `DistributedDataParallel`, global metric reduction,
  and rank-zero checkpoints. The local launcher uses a TCP rendezvous compatible
  with Windows CPU PyTorch builds.
- **Diagnostics:** policy entropy, gradient norm, expected expert load, capacity
  excess, effective global batch size, and finite-metric checks are persisted.

## LLM-as-judge protocol

The `rl_llm_evaluation.json` artifact documents a grounded judge protocol. It is not
executed as a training reward. Before any judge score could be used, it must be
calibrated against held-out human-labelled examples and checked for disagreement
with the deterministic validators.

## Not implemented

- LLM rollout serving (for example vLLM or SGLang);
- RLHF/RLAIF or reward-model training;
- multi-node DDP/FSDP/DeepSpeed;
- MoE expert parallelism, token all-to-all dispatch, or large-language-model MoE
  training.

## Interview-safe wording

> I implemented a small, synthetic MoE policy-gradient experiment with PyTorch and
> a DDP path, including data sharding, global metric reduction, rank-zero
> checkpointing, reward verification, and router/gradient diagnostics. It is a
> single-node research demonstration, not distributed LLM post-training.
