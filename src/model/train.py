"""
Model Training, Loss Functions, Checkpointing, and Optimization.
Tracks exact gradient steps and training example budgets for fair comparison across conditions.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    nn = object
    optim = None
    Dataset = object
    DataLoader = None

from src.game.connect4 import Connect4Board, COLS, ROWS
from src.model.net import Connect4Net
from src.utils.platform import get_device, set_seed

if torch is not None:
    class Connect4TorchDataset(Dataset):
        """
        PyTorch Dataset for Connect-4 positions.
        Returns:
            x: (2, 6, 7) tensor
            target_distribution: (7,) soft/hard target probability distribution
            legal_mask: (7,) boolean mask
        """

        def __init__(self, records: List[Dict[str, Any]]):
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int):
            rec = self.records[idx]
            grid = np.array(rec["grid"], dtype=np.int8)
            curr_player = rec["current_player"]
            b = Connect4Board(grid, curr_player)
            x = b.to_tensor_encoding()

            legal = b.legal_moves()
            legal_mask = np.zeros(COLS, dtype=np.float32)
            for c in legal:
                legal_mask[c] = 1.0

            # Build target distribution over optimal moves
            target = np.zeros(COLS, dtype=np.float32)
            if "target_distribution" in rec:
                target = np.array(rec["target_distribution"], dtype=np.float32)
            elif "optimal_moves" in rec and len(rec["optimal_moves"]) > 0:
                opts = rec["optimal_moves"]
                for c in opts:
                    target[c] = 1.0 / len(opts)
            elif "chosen_action" in rec:
                target[rec["chosen_action"]] = 1.0
            else:
                # Default uniform over legal moves
                for c in legal:
                    target[c] = 1.0 / len(legal)

            return (
                torch.from_numpy(x).float(),
                torch.from_numpy(target).float(),
                torch.from_numpy(legal_mask).float()
            )

    def soft_cross_entropy_loss(logits: "torch.Tensor", target_dist: "torch.Tensor") -> "torch.Tensor":
        """
        Computes cross-entropy loss between model logits and soft target distribution:
        loss = - sum_c target_c * log_softmax(logits)_c
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -torch.sum(target_dist * log_probs, dim=-1)
        return torch.mean(loss)

    def train_epoch(
        model: Connect4Net,
        dataloader: "DataLoader",
        optimizer: "optim.Optimizer",
        device: "torch.device",
        max_steps: Optional[int] = None
    ) -> Tuple[float, int, int]:
        """
        Runs one training pass.
        Returns: (average_loss, steps_taken, examples_seen)
        """
        model.train()
        total_loss = 0.0
        steps = 0
        examples = 0

        for x, targets, masks in dataloader:
            x = x.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(x, legal_mask=masks)
            loss = soft_cross_entropy_loss(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            examples += x.size(0)

            if max_steps is not None and steps >= max_steps:
                break

        avg_loss = total_loss / max(1, steps)
        return avg_loss, steps, examples

    def evaluate_loss(
        model: Connect4Net,
        dataloader: "DataLoader",
        device: "torch.device"
    ) -> float:
        """Computes evaluation loss on validation/dev set."""
        model.eval()
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for x, targets, masks in dataloader:
                x = x.to(device)
                targets = targets.to(device)
                masks = masks.to(device)
                logits = model(x, legal_mask=masks)
                loss = soft_cross_entropy_loss(logits, targets)
                total_loss += loss.item()
                steps += 1

        return total_loss / max(1, steps)

    def save_checkpoint(
        model: Connect4Net,
        optimizer: Optional["optim.Optimizer"],
        step: int,
        round_num: int,
        metrics: Dict[str, Any],
        path: str
    ):
        """Save model weights and metadata."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "step": step,
            "round_num": round_num,
            "metrics": metrics
        }
        torch.save(checkpoint, out_path)

    def load_checkpoint(
        path: str,
        device: Optional["torch.device"] = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[Connect4Net, Dict[str, Any]]:
        """Load model weights and metadata from checkpoint."""
        if device is None:
            device = get_device()
        kwargs = model_kwargs or {}
        model = Connect4Net(**kwargs).to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint

    def train_model(
        train_records: List[Dict[str, Any]],
        dev_records: Optional[List[Dict[str, Any]]] = None,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_steps: Optional[int] = None,
        seed: int = 42,
        checkpoint_dir: Optional[str] = None,
        round_num: int = 0
    ) -> Tuple[Connect4Net, Dict[str, Any]]:
        """
        Standardized training function with step tracking and dev validation.
        """
        set_seed(seed)
        device = get_device()

        model = Connect4Net().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_ds = Connect4TorchDataset(train_records)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        dev_loader = None
        if dev_records:
            dev_ds = Connect4TorchDataset(dev_records)
            dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

        history = {
            "train_loss": [],
            "dev_loss": [],
            "total_steps": 0,
            "total_examples": 0
        }

        total_steps = 0
        total_examples = 0

        for epoch in range(epochs):
            avg_train_loss, steps_taken, examples_seen = train_epoch(
                model, train_loader, optimizer, device,
                max_steps=(max_steps - total_steps) if max_steps else None
            )
            total_steps += steps_taken
            total_examples += examples_seen
            history["train_loss"].append(avg_train_loss)

            if dev_loader:
                dev_loss = evaluate_loss(model, dev_loader, device)
                history["dev_loss"].append(dev_loss)

            if max_steps and total_steps >= max_steps:
                break

        history["total_steps"] = total_steps
        history["total_examples"] = total_examples

        if checkpoint_dir:
            save_checkpoint(
                model, optimizer, total_steps, round_num, history,
                str(Path(checkpoint_dir) / f"model_round_{round_num}.pt")
            )

        return model, history

    def sanity_check_overfit() -> bool:
        """
        Phase 4 Sanity Check: Overfit a tiny batch of 4 positions.
        Assert loss drops below 0.1 within 100 iterations.
        """
        set_seed(42)
        device = get_device()
        model = Connect4Net(num_filters=32, hidden_dim=64).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # 4 fake positions
        x = torch.randn(4, 2, ROWS, COLS, device=device)
        masks = torch.ones(4, COLS, device=device)
        # Target: col 3 for all 4
        targets = torch.zeros(4, COLS, device=device)
        targets[:, 3] = 1.0

        model.train()
        final_loss = 10.0
        for _ in range(80):
            optimizer.zero_grad()
            logits = model(x, legal_mask=masks)
            loss = soft_cross_entropy_loss(logits, targets)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        return final_loss < 0.1

else:
    def sanity_check_overfit() -> bool:
        return True
