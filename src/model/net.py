"""
Neural Network Architecture for Connect-4 Player.
Lightweight Convolutional Network with legal action masking
and soft output probability distribution generation for Task 3 distillation.
"""

from typing import Optional, List, Tuple
import numpy as np

# PyTorch import with graceful fallback handling
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = object
    F = None

from src.game.connect4 import Connect4Board, ROWS, COLS

if torch is not None:
    class Connect4Net(nn.Module):
        """
        Convolutional policy network for Connect-4.
        Input: (B, 2, 6, 7)
        Output: (B, 7) column logits with legal action masking.
        """

        def __init__(self, num_filters: int = 64, hidden_dim: int = 128):
            super().__init__()
            self.conv1 = nn.Conv2d(2, num_filters, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(num_filters)
            self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(num_filters)
            self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(num_filters)

            flatten_dim = num_filters * ROWS * COLS
            self.fc1 = nn.Linear(flatten_dim, hidden_dim)
            self.dropout = nn.Dropout(p=0.1)
            self.fc2 = nn.Linear(hidden_dim, COLS)

        def forward(self, x: "torch.Tensor", legal_mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
            """
            x: (B, 2, 6, 7)
            legal_mask: (B, 7) boolean or float tensor where 1 indicates legal, 0 illegal
            """
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = F.relu(self.bn3(self.conv3(out)))
            out = out.reshape(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = self.dropout(out)
            logits = self.fc2(out)

            if legal_mask is not None:
                # Mask illegal moves with -1e9 so their probability becomes 0 under softmax
                mask_bool = legal_mask.bool()
                large_neg = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
                logits = torch.where(mask_bool, logits, large_neg)

            return logits

        @torch.no_grad()
        def predict_distribution(self, board: Connect4Board, temperature: float = 1.0) -> np.ndarray:
            """
            Predict full probability distribution over the 7 columns for a given board.
            Distillation in Task 3 requires these exact soft outputs.
            Illegal moves are strictly assigned probability 0.0.
            """
            self.eval()
            device = next(self.parameters()).device
            tensor_np = board.to_tensor_encoding()
            x = torch.from_numpy(tensor_np).unsqueeze(0).to(device)

            legal = board.legal_moves()
            mask_np = np.zeros(COLS, dtype=np.float32)
            for c in legal:
                mask_np[c] = 1.0
            legal_mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)

            logits = self.forward(x, legal_mask=legal_mask)
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1).squeeze(0).cpu().numpy()
            else:
                # Greedy one-hot
                probs = np.zeros(COLS, dtype=np.float32)
                best_col = int(torch.argmax(logits, dim=-1).item())
                probs[best_col] = 1.0

            # Normalize across legal moves to protect numerical stability
            legal_sum = float(probs[legal].sum())
            if legal_sum > 0:
                probs[legal] = probs[legal] / legal_sum
            return probs

        @torch.no_grad()
        def select_action(self, board: Connect4Board, greedy: bool = True, temperature: float = 1.0) -> int:
            """Selects next column move for Connect4Board."""
            probs = self.predict_distribution(board, temperature=temperature if not greedy else 0.0)
            legal = board.legal_moves()
            if not legal:
                raise ValueError("No legal moves available.")
            if greedy:
                # Choose legal action with highest probability
                legal_probs = [(probs[c], c) for c in legal]
                return max(legal_probs, key=lambda p: p[0])[1]
            else:
                # Sample from legal probabilities
                p_legal = probs[legal]
                p_legal = p_legal / p_legal.sum()
                return int(np.random.choice(legal, p=p_legal))

else:
    class Connect4Net:
        """Stub when PyTorch is not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for Connect4Net.")
