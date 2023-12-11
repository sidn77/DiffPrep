import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from matplotlib import pyplot as plt
import json
import ast

# Set plotly renderer
rndr_type = "jupyterlab+png"
pio.renderers.default = rndr_type


class PCA(object):
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) -> None:  # 5 points
        X = X - np.mean(X, axis=0)
        self.U, self.S, self.V = np.linalg.svd(X, full_matrices=False)

    def transform(self, data: np.ndarray, K: int = 2) -> np.ndarray:  # 2 pts
        data = data - np.mean(data, axis=0)
        return (data @ self.get_V().T[:, :K])

    def transform_rv(
        self, data: np.ndarray, retained_variance: float = 0.99
    ) -> np.ndarray:  # 3 pts
        data = data - np.mean(data, axis=0)
        k = np.argwhere(np.cumsum(self.S ** 2) > (retained_variance * np.sum((self.S ** 2))))[0][0]
        return self.transform(data, K=k + 1)

    def get_V(self) -> np.ndarray:
        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) -> None:  # 5 pts
        # df = pd.DataFrame(X, columns=range(X.shape[1]))
        # df['y'] = y
        # fig = px.scatter_3d(df, x=df.columns[0], y=df.columns[1], z=df.columns[2], color='y', title=fig_title)
        # fig.show()
        self.fit(X)
        X = self.transform(X, K=2)
        df = pd.DataFrame(X, columns=range(X.shape[1]))
        df['y'] = y
        fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color='y', title=fig_title)
        fig.show()

def parse_weight_file(file_path):
    weights_dict = {}
    with open(file_path) as my_file:
        for line in my_file.readlines():
            split = line.split(":")
            print(split)
            x = ast.literal_eval(split[1].strip())
            weights_dict.update({split[0]: x})

    PCA().visualize(weights_dict[list(weights_dict.keys())[2]], list(weights_dict.keys())[2], "random")


if __name__ == '__main__':
    parse_weight_file('bestpipelines_pc.json')
