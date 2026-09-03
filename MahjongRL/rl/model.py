"""Res2Net + LSTM policy/value network (PyTorch).

Res2Net (Gao et al., 2021, "Res2Net: A New Multi-scale Backbone Architecture")
replaces the single 3x3 conv inside a residual bottleneck with hierarchical
sub-groups: channels are split into `scale` groups and each group's conv
receives the previous group's output, giving multi-scale receptive fields
within one block. On the 4x34 mahjong grid this captures both local shapes
(adjacent tiles -> chows) and long-range suit/honor structure in one block.

The LSTM consumes the recent event history (who did what), giving the policy
memory of turn order, claim behavior, and opponent discard tempo - the same
signals AlphaJong's hand-crafted defense (ai_defense.js) reads explicitly.

Heads:
  policy: logits over the 41 flat actions (illegal ones masked to -inf)
  value:  scalar expected normalized score for the acting player
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import NUM_PLANES, HISTORY_LEN, EVENT_DIM
from hk_mahjong.actions import NUM_ACTIONS


class Res2NetBlock(nn.Module):
    """Bottleneck residual block with Res2Net hierarchical multi-scale convs."""

    def __init__(self, channels, scale=4, bottleneck=None):
        super().__init__()
        bottleneck = bottleneck or channels
        assert bottleneck % scale == 0
        self.scale = scale
        self.width = bottleneck // scale
        self.conv_in = nn.Conv2d(channels, bottleneck, 1, bias=False)
        self.bn_in = nn.BatchNorm2d(bottleneck)
        # one 3x3 conv per sub-group except the identity first group
        self.convs = nn.ModuleList(
            nn.Conv2d(self.width, self.width, 3, padding=1, bias=False)
            for _ in range(scale - 1))
        self.bns = nn.ModuleList(nn.BatchNorm2d(self.width)
                                 for _ in range(scale - 1))
        self.conv_out = nn.Conv2d(bottleneck, channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn_in(self.conv_in(x)))
        groups = torch.chunk(out, self.scale, dim=1)
        results = [groups[0]]                      # first group: identity
        prev = None
        for i in range(1, self.scale):
            g = groups[i] if prev is None else groups[i] + prev
            prev = F.relu(self.bns[i - 1](self.convs[i - 1](g)))
            results.append(prev)
        out = torch.cat(results, dim=1)
        out = self.bn_out(self.conv_out(out))
        return F.relu(out + x)                     # outer residual connection


class MahjongNet(nn.Module):
    def __init__(self, channels=64, num_blocks=4, scale=4,
                 lstm_hidden=128, lstm_layers=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(
            *[Res2NetBlock(channels, scale) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.lstm = nn.LSTM(EVENT_DIM, lstm_hidden, lstm_layers,
                            batch_first=True)

        fused = channels + lstm_hidden
        self.policy_head = nn.Sequential(
            nn.Linear(fused, 256), nn.ReLU(), nn.Linear(256, NUM_ACTIONS))
        self.value_head = nn.Sequential(
            nn.Linear(fused, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh())

    def forward(self, planes, history, mask=None):
        """planes (B,18,4,34), history (B,32,45), mask (B,41) bool or None."""
        s = self.stem(planes)
        s = self.blocks(s)
        s = self.pool(s).flatten(1)                # (B, channels)
        h, _ = self.lstm(history)
        h = h[:, -1, :]                            # last hidden state
        z = torch.cat([s, h], dim=1)
        logits = self.policy_head(z)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        value = self.value_head(z).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, planes, history, mask, greedy=False):
        """Sample (or argmax) an action. Returns (action, logprob, value)."""
        logits, value = self(planes, history, mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), value


def build_model(device="cpu", **kwargs):
    return MahjongNet(**kwargs).to(device)
