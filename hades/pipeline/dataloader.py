from typing import Tuple
import numpy as np

import torch

from torch.utils.data import Dataset


class EvolutionMatrixLoader(Dataset):
    def __init__(self, evolution_matrix, labels) -> None:
        super().__init__()
        self.evolution_matrix = torch.tensor(evolution_matrix)
        self.labels = torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.evolution_matrix)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.evolution_matrix[idx], self.labels[idx]
        print(x.dim(), y.shape)
        return x, y


class EvolutionMatrixLoaderTest(Dataset):
    def __init__(self, evolution_matrix) -> None:
        super().__init__()
        self.evolution_matrix = evolution_matrix

    def __len__(self) -> int:
        return len(self.evolution_matrix)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.evolution_matrix[idx]
