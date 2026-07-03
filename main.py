from pathlib import Path

from src.services.detector import FaceDetector
from src.services.embedder import FaceEmbedder
from src.services.cluster import FaceClusterer

DATASET = Path("dataset/person_identification")


def main():

    detector = FaceDetector()
    embedder = FaceEmbedder()
    clusterer = FaceClusterer()

    faces = []

    for image in DATASET.glob("*.*"):

        detected = detector.detect(image)

        for face in detected:
            faces.append(embedder.normalize(face))

    clustered = clusterer.cluster(faces)

    print("\n========== RESULT ==========\n")

    for face in clustered:

        print(
            f"{face.image_path.name}"
        )

        print(
            f"Cluster : {face.cluster_id}"
        )

        print("---------------------")


if __name__ == "__main__":
    main()