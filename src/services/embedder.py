import numpy as np

from src.models.face import FaceData


class FaceEmbedder:

    @staticmethod
    def normalize(face: FaceData) -> FaceData:

        if face.embedding is None:
            return face

        norm = np.linalg.norm(face.embedding)

        if norm != 0:
            face.embedding = face.embedding / norm

        return face