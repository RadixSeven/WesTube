"""S3FD face detection implementation package."""

import logging
import time
from collections.abc import Sequence

import cv2
import numpy as np
import torch

from .box_utils import nms_
from .nets import S3FDNet

PATH_WEIGHT = "./detectors/s3fd/weights/sfd_face.pth"
img_mean = np.array([104.0, 117.0, 123.0])[:, np.newaxis, np.newaxis].astype("float32")


class S3FD:
    """S3FD face detector implementation class."""

    def __init__(self, device: str = "cuda"):
        """Initialize the S3FD face detector.

        Args:
            device: Device to run the model on (default: "cuda")
        """
        tstamp = time.time()
        self.device = device

        logging.info("[S3FD] loading with %s", self.device)
        self.net = S3FDNet(device=self.device).to(self.device)
        state_dict = torch.load(PATH_WEIGHT, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        logging.info("[S3FD] finished loading (%.4f sec)", time.time() - tstamp)

    def detect_faces(
        self,
        image: cv2.Mat | np.ndarray,
        confidence_threshold: float = 0.8,
        scales: Sequence[float | int] = (1,),
    ):
        """Detect faces in the input image.

        Args:
            image: Input image to detect faces in
            confidence_threshold: Confidence threshold for detection (default: 0.8)
            scales: List of scales to run detection at (default: [1])

        Returns:
            numpy.ndarray: Array of detected face bounding boxes
        """
        w, h = image.shape[1], image.shape[0]

        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(
                    image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR
                )

                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype("float32")
                scaled_img -= img_mean
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)
                y = self.net(x)

                detections = y.data
                scale = torch.Tensor([w, h, w, h])

                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] > confidence_threshold:
                        score = detections[0, i, j, 0]
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score)
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1

            keep = nms_(bboxes, 0.1)
            return bboxes[keep]
