from sys import stderr

import numpy as np


import torch
from tqdm import tqdm

from hades import features as available_features

from hades.pipeline import Pipeline
from hades.pipeline.dataloader import EvolutionMatrixLoader, EvolutionMatrixLoaderTest
from hades.pipeline.windowed_feature_pipeline import WindowedFeaturePipeline


class RNNPipeline(WindowedFeaturePipeline):
    NAME = "rnn_pipeline"

    def __init__(
        self,
        fs,
        window_length=100e-3,
        window_displacement=50e-3,
        history=3,
        hidden_size=100,
        num_layers=1,
        epochs=100,
        device="cpu",
    ):
        super().__init__(
            f"{self.NAME}_wl{window_length}_wd{window_displacement}_h{history}_emb{hidden_size}_l{num_layers}",
            fs,
            window_length,
            window_displacement,
            history,
        )
        self.name += f"_fts{len(self.features)}"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        if type(device) is int:
            device = f"cuda:{device}"
        self.device = torch.device(device)

    @property
    def features(self):
        return [
            available_features.fn_line_length,
            available_features.fn_area,
            available_features.fn_energy,
            available_features.fn_signed_area,
            available_features.fn_deviation,
        ]

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        dataset = EvolutionMatrixLoader(X, Y)

        self.model = torch.nn.RNN(X.shape[-1], self.hidden_size, self.num_layers)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}", file=stderr)
            pbar = tqdm(dataset)
            self.model.train()
            for x, y in pbar:
                y = y.to(self.device)
                x = x.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Loss: {loss.item():.4f}")
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        dataset = EvolutionMatrixLoaderTest(X)
        preds = []
        pbar = tqdm(dataset, desc=" Predicting ")
        self.model.eval()
        with torch.no_grad():
            for x in pbar:
                x = x.to(self.device)
                preds.append(self.model(X).cpu().numpy())
        return np.array(preds)
