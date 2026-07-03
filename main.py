from pathlib import Path

from src.services.detector import FaceDetector
from src.services.embedder import FaceEmbedder
from src.services.cluster import FaceClusterer
from src.services.confidence import ConfidenceCalculator
from src.services.organizer import OutputOrganizer

DATASET = Path("dataset/person_identification")


def main():

    print("\n========== Face Clustering Pipeline ==========\n")

    detector = FaceDetector()
    embedder = FaceEmbedder()
    clusterer = FaceClusterer()
    confidence = ConfidenceCalculator()
    organizer = OutputOrganizer()

    faces = []

    # Process all images
    for image in sorted(DATASET.glob("*.*")):

        detected_faces = detector.detect(image)

        for face in detected_faces:
            normalized_face = embedder.normalize(face)
            faces.append(normalized_face)

    # Cluster faces
    clustered_faces = clusterer.cluster(faces)

    # Assign confidence
    clustered_faces = confidence.assign(clustered_faces)

    # Organize output folders
    organizer.organize(clustered_faces)

    # Print results
    print("\n========== RESULT ==========\n")

    for face in clustered_faces:

        print(f"Image       : {face.image_path.name}")
        print(f"Cluster     : Person_{face.cluster_id + 1}")
        print(f"Confidence  : {face.confidence:.2f}%")
        print("-" * 45)

    print("\n=============================================")
    print("✅ Face clustering completed successfully.")
    print("📁 Clustered images saved in: output/")
    print("=============================================\n")


if __name__ == "__main__":
    main()