from pathlib import Path
import csv

from src.models.face import FaceData


class ReportGenerator:

    def __init__(self, report_dir="report"):

        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)

    def generate(self, faces):

        report_file = self.report_dir / "report.csv"

        with open(
            report_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow(
                [
                    "Image",
                    "Cluster",
                    "Confidence",
                ]
            )

            for face in faces:

                writer.writerow(
                    [
                        face.image_path.name,
                        f"Person_{face.cluster_id + 1}",
                        f"{face.confidence:.2f}%",
                    ]
                )

        return report_file