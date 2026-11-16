python
import torch
import torch.nn as nn

from src.replay import ReplayBuffer, train_online_with_replay


class TinyModel(nn.Module):
    """Modèle jouet à l'interface (x, état) -> (prédiction, état)."""
    def _init_(self):
        super()._init_()
        self.layer = nn.Linear(2, 1)

    def forward(self, x, state=None):
        return self.layer(x), None


def test_buffer_respects_capacity():
    buffer = ReplayBuffer(capacity=3)
    for i in range(5):
        buffer.add(torch.tensor([float(i), 0.0]), torch.tensor([float(i)]))
    assert len(buffer) == 3


def test_sample_shapes():
    buffer = ReplayBuffer(capacity=10)
    for i in range(4):
        buffer.add(torch.randn(2), torch.randn(1))
    xs, ys = buffer.sample(3)
    assert xs.shape == (3, 2) and ys.shape == (3, 1)

def test_training_returns_one_error_per_step():
    X, y = torch.randn(50, 2), torch.randn(50, 1)
    errors = train_online_with_replay(TinyModel(), X, y, replay_k=4)
    assert len(errors) == 50