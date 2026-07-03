from pathlib import Path
import shutil

from src.models.face import FaceData


class OutputOrganizer:

    def __init__(self, output_dir="output"):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True)

    def organize(self, faces):

        for face in faces:

            cluster_folder = (
                self.output_dir /
                f"Person_{face.cluster_id + 1}"
            )

            cluster_folder.mkdir(
                exist_ok=True
            )

            destination = (
                cluster_folder /
                face.image_path.name
            )

            shutil.copy2(
                face.image_path,
                destination,
            )

        return True