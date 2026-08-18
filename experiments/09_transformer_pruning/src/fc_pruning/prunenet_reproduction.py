from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats


@dataclass
class PruneNetTrainingState:
    episode: int
    best_score: float
    best_model_state: dict
    episode_rows: list[dict]


def set_prunenet_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.inference_mode()
def singular_values(weight: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return the spectrum used by PruneNet's KS reward.

    The official code calls ``torch.svd`` and discards U/V. ``svdvals`` gives
    the same spectrum without materializing those two large matrices.
    """
    return torch.linalg.svdvals(weight.detach().float().to(device)).cpu()


def spectral_ks_reward(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    distance = stats.ks_2samp(reference.numpy(), candidate.numpy()).statistic
    return 99999.0 if distance == 0 else float(1.0 / distance)


def train_prunenet_policy(
    layers,
    action_model,
    compression_ratio: float,
    num_episodes: int,
    learning_rate: float,
    device: torch.device,
    checkpoint_callback,
    resume_payload: dict | None = None,
) -> PruneNetTrainingState:
    """Train the official PruneNet policy with layer-streamed model weights."""
    optimizer = torch.optim.AdamW(action_model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    start_episode = 0
    best_score = 0.0
    best_model_state = copy.deepcopy(action_model.state_dict())
    episode_rows: list[dict] = []
    if resume_payload:
        action_model.load_state_dict(resume_payload["action_model"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        scaler.load_state_dict(resume_payload["scaler"])
        start_episode = int(resume_payload["episode"]) + 1
        best_score = float(resume_payload["best_score"])
        best_model_state = resume_payload["best_model_state"]
        episode_rows = list(resume_payload.get("episode_rows", []))
        torch.set_rng_state(resume_payload["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state_all"])
        np.random.set_state(resume_payload["numpy_rng_state"])
        random.setstate(resume_payload["python_rng_state"])

    print("PruneNet reference spectra", flush=True)
    references = []
    for index, layer in enumerate(layers):
        print(f"PruneNet reference layer={index:02d}", flush=True)
        references.append(singular_values(layer.mlp.gate_proj.weight, device))

    action_model.train()
    for episode in range(start_episode, num_episodes):
        sampled_indices = []
        rewards = []
        total_reward = 0.0
        for index, layer in enumerate(layers):
            print(
                f"PruneNet episode={episode + 1:02d}/{num_episodes:02d} "
                f"sample layer={index:02d}",
                flush=True,
            )
            state = layer.mlp.gate_proj.weight.detach().float().to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                probabilities = action_model(state)
                keep = int((1.0 - compression_ratio) * state.shape[0])
                indices = torch.multinomial(
                    probabilities, keep, replacement=False
                ).sort().values
            candidate_spectrum = singular_values(state.index_select(0, indices), device)
            reward = spectral_ks_reward(references[index], candidate_spectrum)
            sampled_indices.append(indices.cpu())
            rewards.append(reward)
            total_reward += reward
            del state, probabilities, indices, candidate_spectrum
            torch.cuda.empty_cache()

        discounted = np.array(
            [0.99**index * reward for index, reward in enumerate(rewards)]
        )
        total_loss = 0.0
        for index, (layer, indices, reward) in enumerate(
            zip(layers, sampled_indices, discounted)
        ):
            print(
                f"PruneNet episode={episode + 1:02d}/{num_episodes:02d} "
                f"update layer={index:02d}",
                flush=True,
            )
            state = layer.mlp.gate_proj.weight.detach().float().to(device)
            action = indices.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                probabilities = action_model(state)
                loss = -torch.gather(
                    torch.log(probabilities.clamp_min(1e-12)), 0, action
                ).sum() * float(reward)
            total_loss += float(loss.detach())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.nn.utils.clip_grad_norm_(action_model.parameters(), 1.0)
            del state, action, probabilities, loss
            torch.cuda.empty_cache()

        if total_reward > best_score:
            best_score = total_reward
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in action_model.state_dict().items()
            }
        row = {
            "episode": episode + 1,
            "average_reward": total_reward / len(layers),
            "average_loss": total_loss / len(layers),
            "best_total_reward": best_score,
        }
        episode_rows.append(row)
        print(f"PruneNet episode result {row}", flush=True)
        checkpoint_callback(
            {
                "episode": episode,
                "action_model": action_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_score": best_score,
                "best_model_state": best_model_state,
                "episode_rows": episode_rows,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
            }
        )

    return PruneNetTrainingState(
        episode=num_episodes,
        best_score=best_score,
        best_model_state=best_model_state,
        episode_rows=episode_rows,
    )


@torch.no_grad()
def select_prunenet_channels(
    layers,
    action_model,
    compression_ratio: float,
    device: torch.device,
) -> list[torch.Tensor]:
    action_model.eval()
    selections = []
    for index, layer in enumerate(layers):
        print(f"PruneNet final selection layer={index:02d}", flush=True)
        state = layer.mlp.gate_proj.weight.detach().float().to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            probabilities = action_model(state)
            keep = int((1.0 - compression_ratio) * state.shape[0])
            indices = torch.multinomial(
                probabilities, keep, replacement=False
            ).sort().values
        selections.append(indices.cpu())
        del state, probabilities, indices
        torch.cuda.empty_cache()
    return selections
