import numpy as np

from src.models.face import FaceData


class ConfidenceCalculator:
    """
    Calculate confidence score based on cosine similarity.
    """

    @staticmethod
    def cosine_similarity(vec1, vec2):

        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        return float(np.dot(vec1, vec2))

    def assign(self, faces):

        clusters = {}

        for face in faces:

            clusters.setdefault(
                face.cluster_id,
                []
            ).append(face)

        for cluster_faces in clusters.values():

            embeddings = [
                face.embedding
                for face in cluster_faces
            ]

            centroid = np.mean(
                embeddings,
                axis=0,
            )

            centroid = centroid / np.linalg.norm(
                centroid
            )

            for face in cluster_faces:

                similarity = self.cosine_similarity(
                    face.embedding,
                    centroid,
                )

                confidence = max(
                    0,
                    min(
                        100,
                        similarity * 100,
                    ),
                )

                face.confidence = round(
                    confidence,
                    2,
                )

        return faces