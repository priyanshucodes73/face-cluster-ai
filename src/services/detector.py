from pathlib import Path
from typing import List

import cv2
from insightface.app import FaceAnalysis

from src.models.face import FaceData
from src.core.logger import logger


class FaceDetector:
    """
    Detect the primary face from an image using InsightFace.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers=None,
    ):

        if providers is None:
            providers = ["CPUExecutionProvider"]

        logger.info("Loading InsightFace model...")

        self.app = FaceAnalysis(
            name=model_name,
            providers=providers,
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640),
        )

        logger.info("InsightFace model loaded successfully.")

    def detect(self, image_path: Path) -> List[FaceData]:

        image = cv2.imread(str(image_path))

        if image is None:
            logger.warning(f"Unable to read image: {image_path}")
            return []

        faces = self.app.get(image)

        if len(faces) == 0:
            logger.warning(f"No face detected in {image_path.name}")
            return []

        # Select the largest face
        best_face = max(
            faces,
            key=lambda face: (
                (face.bbox[2] - face.bbox[0])
                * (face.bbox[3] - face.bbox[1])
            ),
        )

        face_data = FaceData(
            image_path=image_path,
            face_id=0,
            bbox=best_face.bbox,
            landmarks=best_face.kps,
            embedding=best_face.embedding,
        )

        logger.info(f"Detected face in {image_path.name}")

        return [face_data]