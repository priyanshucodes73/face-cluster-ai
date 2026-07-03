from pathlib import Path

from src.services.detector import FaceDetector
from src.services.embedder import FaceEmbedder
from src.services.cluster import FaceClusterer
from src.services.confidence import ConfidenceCalculator
from src.services.organizer import OutputOrganizer
from src.services.report import ReportGenerator
from src.services.visualizer import ClusterVisualizer
from src.utils.timer import Timer

DATASET = Path("dataset/person_identification")


def main():

    timer = Timer()

    print("\n" + "=" * 60)
    print("        FACE CLUSTERING PIPELINE")
    print("=" * 60)

    detector = FaceDetector()
    embedder = FaceEmbedder()
    clusterer = FaceClusterer()
    confidence = ConfidenceCalculator()
    organizer = OutputOrganizer()
    report = ReportGenerator()
    visualizer = ClusterVisualizer()

    faces = []

    print("\n[1/6] Detecting Faces...\n")

    for image in sorted(DATASET.glob("*.*")):

        detected_faces = detector.detect(image)

        for face in detected_faces:
            face = embedder.normalize(face)
            faces.append(face)

    print(f"✓ Total Faces Detected : {len(faces)}")

    print("\n[2/6] Clustering Faces...")

    clustered_faces = clusterer.cluster(faces)

    print("✓ Clustering Completed")

    print("\n[3/6] Calculating Confidence...")

    clustered_faces = confidence.assign(clustered_faces)

    print("✓ Confidence Assigned")

    print("\n[4/6] Organizing Images...")

    organizer.organize(clustered_faces)

    print("✓ Images Saved")

    print("\n[5/6] Generating CSV Report...")

    report_path = report.generate(clustered_faces)

    print("✓ Report Generated")

    print("\n[6/6] Creating Visualization...")

    graph_path = visualizer.visualize(clustered_faces)

    print("✓ Visualization Generated")

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    for face in clustered_faces:

        print(f"Image       : {face.image_path.name}")
        print(f"Cluster     : Person_{face.cluster_id + 1}")
        print(f"Confidence  : {face.confidence:.2f}%")
        print("-" * 60)

    total_images = len(clustered_faces)
    total_clusters = len(set(face.cluster_id for face in clustered_faces))

    average_confidence = (
        sum(face.confidence for face in clustered_faces) / total_images
        if total_images > 0
        else 0
    )

    elapsed = timer.stop()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Total Images          : {total_images}")
    print(f"Total Clusters        : {total_clusters}")
    print(f"Average Confidence    : {average_confidence:.2f}%")
    print(f"Processing Time       : {elapsed} sec")

    print("\nGenerated Files")
    print("-" * 60)
    print("✓ Output Folder : output/")
    print(f"✓ CSV Report    : {report_path}")

    if graph_path:
        print(f"✓ Cluster Graph : {graph_path}")

    print("\n" + "=" * 60)
    print("✓ FACE CLUSTERING COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()