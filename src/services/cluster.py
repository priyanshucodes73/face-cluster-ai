from typing import List

import numpy as np
from sklearn.cluster import DBSCAN

from src.models.face import FaceData


class FaceClusterer:

    def __init__(
        self,
        eps: float = 0.45,
        min_samples: int = 1,
    ):
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="cosine",
        )

    def cluster(
        self,
        faces: List[FaceData],
    ) -> List[FaceData]:

        if len(faces) == 0:
            return []

        embeddings = np.array(
            [face.embedding for face in faces]
        )

        labels = self.model.fit_predict(
            embeddings
        )

        for face, label in zip(
            faces,
            labels,
        ):
            face.cluster_id = int(label)

        return faces