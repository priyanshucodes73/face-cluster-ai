from pathlib import Path

from src.services.detector import FaceDetector
from src.services.embedder import FaceEmbedder
from src.services.cluster import FaceClusterer
from src.services.confidence import ConfidenceCalculator

DATASET = Path("dataset/person_identification")


def main():

    detector = FaceDetector()

    embedder = FaceEmbedder()

    clusterer = FaceClusterer()

    confidence = ConfidenceCalculator()

    faces = []

    # Process all images in dataset
    for image in sorted(DATASET.glob("*.*")):

        detected_faces = detector.detect(image)

        for face in detected_faces:
            faces.append(embedder.normalize(face))

    # Perform clustering
    clustered_faces = clusterer.cluster(faces)

    # Assign confidence scores
    clustered_faces = confidence.assign(clustered_faces)

    # Display results
    print("\n========== RESULT ==========\n")

    for face in clustered_faces:

        print(f"Image       : {face.image_path.name}")
        print(f"Cluster     : {face.cluster_id}")
        print(f"Confidence  : {face.confidence:.2f}%")
        print("-" * 40)


if __name__ == "__main__":
    main()