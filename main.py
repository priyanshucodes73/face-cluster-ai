from pathlib import Path

from src.services.detector import FaceDetector
from src.services.embedder import FaceEmbedder
from src.services.cluster import FaceClusterer
from src.services.confidence import ConfidenceCalculator
from src.services.organizer import OutputOrganizer
from src.services.report import ReportGenerator
from src.services.visualizer import ClusterVisualizer

DATASET = Path("dataset/person_identification")


def main():

    print("\n========== FACE CLUSTERING PIPELINE ==========\n")

    detector = FaceDetector()
    embedder = FaceEmbedder()
    clusterer = FaceClusterer()
    confidence = ConfidenceCalculator()
    organizer = OutputOrganizer()
    report = ReportGenerator()
    visualizer = ClusterVisualizer()

    faces = []

    print("Step 1 : Detecting Faces...\n")

    for image in sorted(DATASET.glob("*.*")):

        detected_faces = detector.detect(image)

        for face in detected_faces:

            face = embedder.normalize(face)

            faces.append(face)

    print(f"Detected {len(faces)} face(s).\n")

    print("Step 2 : Clustering Faces...\n")

    clustered_faces = clusterer.cluster(faces)

    print("Step 3 : Calculating Confidence...\n")

    clustered_faces = confidence.assign(clustered_faces)

    print("Step 4 : Organizing Output...\n")

    organizer.organize(clustered_faces)

    print("Step 5 : Generating CSV Report...\n")

    report_path = report.generate(clustered_faces)

    print("Step 6 : Generating Cluster Visualization...\n")

    graph_path = visualizer.visualize(clustered_faces)

    print("\n============== RESULT ==============\n")

    for face in clustered_faces:

        print(f"Image       : {face.image_path.name}")
        print(f"Cluster     : Person_{face.cluster_id + 1}")
        print(f"Confidence  : {face.confidence:.2f}%")
        print("---------------------------------------")

    print("\n============================================")
    print("✅ Face clustering completed successfully.")
    print(f"📁 Output Folder : output/")
    print(f"📄 CSV Report    : {report_path}")

    if graph_path:
        print(f"📊 Cluster Graph : {graph_path}")

    print("============================================\n")


if __name__ == "__main__":
    main()