"""Replay : rejouer des échantillons passés pour limiter l'oubli catastrophique.

À chaque pas, en plus de l'échantillon courant, on réapprend sur quelques
échantillons tirés d'un tampon (replay buffer) des pas précédents.
"""
import random
from collections import deque

import torch
import torch.nn as nn

LEARNING_RATE = 5e-3
SEED = 42


class ReplayBuffer:
    """Tampon circulaire d'échantillons passés (x de forme (2,), y de forme (1,))."""
    def _init_(self, capacity: int = 500, seed: int = SEED):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.buffer.append((x.detach().clone(), y.detach().clone()))

    def sample(self, k: int):
        """Tire k échantillons au hasard ; retourne (xs de forme (k,2), ys de forme (k,1))."""
        batch = self.rng.sample(list(self.buffer), min(k, len(self.buffer)))
        return (torch.stack([b[0] for b in batch]),
                torch.stack([b[1] for b in batch]))

    def _len_(self) -> int:
        return len(self.buffer)


def _detach(state):
    if state is None:
        return None
    if isinstance(state, tuple):              # LSTM : (h, c)
        return tuple(s.detach() for s in state)
    return state.detach()


def train_online_with_replay(model, X, y, lr: float = LEARNING_RATE,
                             buffer_size: int = 500, replay_k: int = 8):
    """Entraînement online avec replay. Retourne l'erreur courante à chaque pas.

    Approximation : les échantillons rejoués sont traités comme des séquences d'un
    seul pas, sans état caché (state = None).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(capacity=buffer_size)
    state, errors = None, []
    for t in range(len(X)):
        x_t = X[t].view(1, 1, -1)
        y_t = y[t].view(1, 1, 1)
        prediction, state = model(x_t, state)
        loss = loss_fn(prediction, y_t)
        errors.append(loss.item())            # erreur sur l'échantillon courant

        if len(buffer) >= replay_k:
            xs, ys = buffer.sample(replay_k)
            replayed, _ = model(xs.view(len(xs), 1, -1), None)
            loss = loss + loss_fn(replayed, ys.view(len(ys), 1, 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state = _detach(state)
        buffer.add(X[t], y[t])
    return errors