from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class FaceData:
    """
    Stores information about a detected face.
    """

    image_path: Path
    face_id: int

    bbox: np.ndarray
    landmarks: np.ndarray

    embedding: Optional[np.ndarray] = None

    cluster_id: int = -1

    confidence: float = 0.0

    person_name: str = ""

    processed: bool = False

    metadata: dict = field(default_factory=dict)

    def has_embedding(self) -> bool:
        return self.embedding is not None

    def mark_processed(self):
        self.processed = True

    def set_cluster(self, cluster: int):
        self.cluster_id = cluster

    def set_confidence(self, score: float):
        self.confidence = round(score, 4)

    def set_person(self, name: str):
        self.person_name = name

    def __repr__(self):
        return (
            f"FaceData("
            f"face_id={self.face_id}, "
            f"cluster={self.cluster_id}, "
            f"confidence={self.confidence:.2f})"
        )