from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.models.face import FaceData


class ClusterVisualizer:

    def __init__(self, report_dir="report"):

        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)

    def visualize(self, faces):

        if len(faces) < 2:
            return None

        embeddings = np.array(
            [face.embedding for face in faces]
        )

        labels = np.array(
            [face.cluster_id for face in faces]
        )

        pca = PCA(n_components=2)

        points = pca.fit_transform(embeddings)

        plt.figure(figsize=(8, 6))

        unique_labels = np.unique(labels)

        for label in unique_labels:

            mask = labels == label

            plt.scatter(
                points[mask, 0],
                points[mask, 1],
                label=f"Person {label + 1}",
                s=120,
            )

        plt.title("Face Clustering Result")

        plt.xlabel("PCA Component 1")

        plt.ylabel("PCA Component 2")

        plt.legend()

        plt.grid(True)

        output_file = self.report_dir / "clusters.png"

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        return output_file