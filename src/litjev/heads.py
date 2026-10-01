"""A small decision head read beside lm_head at the answer boundary.

The backbone stays frozen. The head sees only readout hidden states and summary
statistics of the candidate distribution, never question or state text.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from litjev.slots import SLOT_FORMAT

FEATURE_FORMAT = "readout_hidden_plus_stats_v1"
QUESTION_TYPES = ("choice", "score", "noul")
STATS_DIM = 5 + len(QUESTION_TYPES)
OUTCOME_DIM = 4


def concentration(probabilities):
    count = len(probabilities)
    if count == 1:
        return 1.0
    return float(np.clip((count * sum(p * p for p in probabilities) - 1) / (count - 1), 0, 1))


def distribution_stats(probabilities, question_type):
    """Summary of the fast candidate distribution; scale-free so it transfers across K."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("probabilities must be a non-empty vector")
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"Unknown question type {question_type!r}")
    order = np.sort(p)[::-1]
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    normalized_entropy = entropy / math.log(len(p)) if len(p) > 1 else 0.0
    margin = float(order[0] - order[1]) if len(p) > 1 else 1.0
    one_hot = [1.0 if question_type == kind else 0.0 for kind in QUESTION_TYPES]
    return np.array(
        [float(order[0]), normalized_entropy, margin, concentration(p), math.log(len(p)), *one_hot],
        dtype=np.float32,
    )


def build_features(hidden, probabilities, question_type):
    """Concatenate readout hidden states (already selected layers) with distribution stats."""
    hidden = np.asarray(hidden, dtype=np.float32).reshape(-1)
    return np.concatenate([hidden, distribution_stats(probabilities, question_type)])


def parse_feature_layers(text):
    """Comma-separated hidden_states indices; -1 is the final normalized output."""
    layers = tuple(int(item) for item in text.split(",") if item.strip())
    if not layers:
        raise ValueError("At least one feature layer is required")
    return layers


@dataclass(frozen=True)
class HeadMetadata:
    model_id: str
    revision: str
    hidden_size: int
    feature_layers: tuple[int, ...]
    hidden_width: int = 256
    pca_dim: int = 0
    slot_format: str = SLOT_FORMAT
    feature_format: str = FEATURE_FORMAT
    training: dict = field(default_factory=dict)
    # Which readout produced the distribution this head's features were built
    # from. "coded" scores single-token letters; "content" scores each option's
    # own text, and its softmax over option scores is a different quantity on a
    # different scale. A head fitted to one and served under the other is reading
    # numbers that do not mean what it learned they meant, and nothing in its
    # output would say so. Defaults to coded, which is every head trained so far.
    readout: str = "coded"

    @property
    def hidden_dim(self):
        return self.hidden_size * len(self.feature_layers)

    @property
    def input_dim(self):
        return self.hidden_dim + STATS_DIM

    @property
    def model_dim(self):
        """Width the MLP actually sees: PCA components (if any) plus the stats."""
        return (self.pca_dim if self.pca_dim else self.hidden_dim) + STATS_DIM

    def check_serving(self, model_id, revision=None, readout=None):
        if self.model_id != model_id:
            raise ValueError("Decision head was trained for a different model")
        if revision is not None and self.revision != revision:
            raise ValueError("Decision head was trained for a different model revision")
        if self.slot_format != SLOT_FORMAT:
            raise ValueError("Decision head prompt format changed; retrain the head")
        if self.feature_format != FEATURE_FORMAT:
            raise ValueError("Decision head feature format changed; retrain the head")
        if readout is not None and self.readout != readout:
            raise ValueError(
                f"this head's features were built from the {self.readout!r} readout and it "
                f"is being served under {readout!r}. The two produce different quantities "
                f"on different scales -- one a distribution over letter codes, the other a "
                f"softmax over option scores -- so it would be reading numbers that do not "
                f"mean what it learned they meant, and nothing in its output would say so. "
                f"Train a head under {readout!r}, or serve this one under {self.readout!r}")


class DecisionHead(nn.Module):
    def __init__(self, metadata: HeadMetadata, dropout=0.1):
        super().__init__()
        self.metadata = metadata
        dim = metadata.input_dim
        if metadata.pca_dim < 0 or metadata.pca_dim > metadata.hidden_dim:
            raise ValueError("pca_dim must be between 0 and the hidden feature width")
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        if metadata.pca_dim:
            # Whitened principal directions of the standardized hidden features.
            self.register_buffer("projection", torch.zeros(metadata.hidden_dim, metadata.pca_dim))
        self.net = nn.Sequential(
            nn.Linear(metadata.model_dim, metadata.hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(metadata.hidden_width, OUTCOME_DIM),
        )

    def _reduce(self, standardized):
        if not self.metadata.pca_dim:
            return standardized
        split = self.metadata.hidden_dim
        components = standardized[:, :split] @ self.projection
        return torch.cat([components, standardized[:, split:]], dim=1)

    def forward(self, features):
        return self.net(self._reduce((features - self.mean) / self.std))

    def fit_normalizer(self, features):
        matrix = torch.as_tensor(np.asarray(features, dtype=np.float32))
        self.mean.copy_(matrix.mean(0))
        self.std.copy_(matrix.std(0).clamp_min(1e-6))
        if self.metadata.pca_dim:
            split = self.metadata.hidden_dim
            hidden = ((matrix - self.mean) / self.std)[:, :split]
            rank = min(self.metadata.pca_dim, len(hidden) - 1, split)
            if rank < 1:
                raise ValueError("Not enough examples to fit the PCA bottleneck")
            _, singular, right = torch.linalg.svd(hidden, full_matrices=False)
            scale = (singular[:rank] / math.sqrt(max(len(hidden) - 1, 1))).clamp_min(1e-6)
            self.projection.zero_()
            self.projection[:, :rank] = right[:rank].T / scale

    @torch.no_grad()
    def predict(self, features):
        """Outcome probabilities [N, 4] for a feature matrix [N, input_dim]."""
        self.eval()
        matrix = torch.as_tensor(np.asarray(features, dtype=np.float32)).reshape(
            -1, self.metadata.input_dim
        )
        return torch.softmax(self(matrix.to(self.mean.device)), dim=-1).cpu().numpy()

    def save(self, path):
        payload = {
            key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()
        }
        save_file(payload, str(path), metadata={"litjev": json.dumps(asdict(self.metadata))})

    @classmethod
    def load(cls, path, device="cpu"):
        with safe_open(str(Path(path)), framework="pt") as handle:
            raw = handle.metadata() or {}
        if "litjev" not in raw:
            raise ValueError("Not a LitJev decision head file")
        data = json.loads(raw["litjev"])
        data["feature_layers"] = tuple(data["feature_layers"])
        data.setdefault("pca_dim", 0)
        head = cls(HeadMetadata(**data))
        head.load_state_dict(load_file(str(path)))
        return head.to(device).eval()


def outcome_labels(fast_correct, slow_correct):
    """Map (fast, slow) correctness to the four outcome classes."""
    fast = np.asarray(fast_correct, dtype=bool)
    slow = np.asarray(slow_correct, dtype=bool)
    return np.where(fast, np.where(slow, 0, 1), np.where(slow, 2, 3)).astype(np.int64)


def train_head(
    metadata,
    features,
    labels,
    epochs=30,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=5e-2,
    dropout=0.3,
    validation_fraction=0.15,
    patience=6,
    seed=0,
    device="cpu",
):
    """Supervised four-class training with early stopping; the backbone is never touched.

    A slice of the training set is held out for validation; the returned head is the
    epoch with the lowest validation loss. History rows carry train and validation loss.
    """
    torch.manual_seed(seed)
    matrix = torch.as_tensor(np.asarray(features, dtype=np.float32))
    targets = torch.as_tensor(np.asarray(labels, dtype=np.int64))
    if matrix.ndim != 2 or matrix.shape[1] != metadata.input_dim:
        raise ValueError("features must be [N, input_dim]")
    if len(matrix) != len(targets):
        raise ValueError("features and labels length mismatch")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(matrix), generator=generator)
    n_val = int(len(matrix) * validation_fraction) if len(matrix) >= 8 else 0
    val_index, train_index = order[:n_val], order[n_val:]
    head = DecisionHead(metadata, dropout=dropout).to(device)
    head.fit_normalizer(matrix[train_index])
    counts = torch.bincount(targets[train_index], minlength=OUTCOME_DIM).float().clamp_min(1)
    class_weight = (counts.sum() / (OUTCOME_DIM * counts)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    history, best_state, best_loss, stale = [], None, float("inf"), 0
    for _ in range(epochs):
        head.train()
        shuffled = train_index[torch.randperm(len(train_index), generator=generator)]
        total = 0.0
        for start in range(0, len(shuffled), batch_size):
            index = shuffled[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(head(matrix[index].to(device)), targets[index].to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
        head.eval()
        row = {"train_loss": total / max(len(shuffled), 1)}
        if n_val:
            with torch.no_grad():
                val_loss = loss_fn(
                    head(matrix[val_index].to(device)), targets[val_index].to(device)
                ).item()
            row["val_loss"] = val_loss
            if val_loss < best_loss - 1e-4:
                best_loss, stale = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        if n_val and stale >= patience:
            break
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head, history


def auroc(scores, positives):
    """Rank-based AUROC with tie handling, no sklearn dependency."""
    scores = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(positives, dtype=bool)
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def coverage_accuracy_curve(gain, fast_correct, slow_correct, lambdas):
    """Escalation rate and accuracy per lambda, using the slow answer where escalated."""
    gain = np.asarray(gain, dtype=np.float64)
    fast = np.asarray(fast_correct, dtype=bool)
    slow = np.asarray(slow_correct, dtype=bool)
    rows = []
    for value in lambdas:
        escalate = gain > value
        rows.append(
            {
                "lambda": float(value),
                "escalation_rate": float(escalate.mean()),
                "accuracy": float(np.where(escalate, slow, fast).mean()),
            }
        )
    return rows
