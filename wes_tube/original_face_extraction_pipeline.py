#!/usr/bin/python
"""
Face Extraction Pipeline

This module implements a complete pipeline for extracting face tracks from videos.
It performs the following steps:
1. Video preprocessing (format conversion, frame extraction)
2. Face detection on individual frames
3. Scene boundary detection
4. Face tracking across frames within scenes
5. Face track cropping and video creation

The pipeline can be run from the command line with various parameters to control
the extraction process.
"""

import argparse
import glob
import logging
import os
import pickle
import subprocess
import sys
import time
from shutil import rmtree
from typing import TypedDict

import cv2
import numpy as np
from detectors import S3FD
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager
from scipy import signal
from scipy.interpolate import interp1d

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Type aliases for improved code readability
Num = int | float  # Numeric type that can be either int or float
BoundingBox = (
    tuple[Num, Num, Num, Num] | list[Num]
)  # Bounding box coordinates (x1, y1, x2, y2)


class FaceDict(TypedDict):
    """
    A dictionary representing a detected face in a frame.

    Attributes:
        frame: The frame number where the face was detected
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        conf: Confidence score of the detection
    """

    frame: int
    bbox: BoundingBox
    conf: float


class TrackDict(TypedDict):
    """
    A dictionary representing a tracked face across multiple frames.

    Attributes:
        frame: Array of frame numbers where the face appears
        bbox: Array of bounding box coordinates for each frame
    """

    frame: np.ndarray
    bbox: np.ndarray


class DetDict(TypedDict):
    """
    A dictionary containing processed face detection coordinates and sizes.

    Attributes:
        x: List of x-coordinates for the center of detected faces
        y: List of y-coordinates for the center of detected faces
        s: List of sizes (half the maximum of width or height) for detected faces
    """

    x: list[float]
    y: list[float]
    s: list[float]


class ProcTrackDict(TypedDict):
    """
    A dictionary containing both the original track and processed track data.

    Attributes:
        track: Original track data
        proc_track: Processed track data with smoothed coordinates
    """

    track: TrackDict
    proc_track: DetDict


# List of all the face detections in a given frame
FrameDetection = list[FaceDict]


# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========
def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the face extraction pipeline.

    Returns:
        argparse.Namespace: An object containing all the parsed arguments with additional
                           directory paths added for convenience.
    """
    parser = argparse.ArgumentParser(description="FaceTracker")
    parser.add_argument(
        "--data_dir", type=str, default="data/work", help="Output directory"
    )
    # noinspection SpellCheckingInspection
    parser.add_argument("--videofile", type=str, default="", help="Input video file")
    parser.add_argument("--reference", type=str, default="", help="Video reference")
    parser.add_argument(
        "--facedet_scale",
        type=float,
        default=0.25,
        help="Scale factor for face detection",
    )
    parser.add_argument(
        "--crop_scale", type=float, default=0.40, help="Scale bounding box"
    )
    parser.add_argument(
        "--min_track", type=int, default=100, help="Minimum face-track duration"
    )
    parser.add_argument("--frame_rate", type=int, default=25, help="Frame rate")
    parser.add_argument(
        "--num_failed_det",
        type=int,
        default=25,
        help="Number of missed detections allowed before tracking is stopped",
    )
    parser.add_argument(
        "--min_face_size", type=int, default=100, help="Minimum face size in pixels"
    )
    opt = parser.parse_args()

    # noinspection SpellCheckingInspection
    opt.avi_dir = os.path.join(opt.data_dir, "pyavi")
    opt.tmp_dir = os.path.join(opt.data_dir, "pytmp")
    opt.work_dir = os.path.join(opt.data_dir, "pywork")
    opt.crop_dir = os.path.join(opt.data_dir, "pycrop")
    opt.frames_dir = os.path.join(opt.data_dir, "pyframes")
    return opt


# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========


def bb_intersection_over_union(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box_a: First bounding box in format (x1, y1, x2, y2)
        box_b: Second bounding box in format (x1, y1, x2, y2)

    Returns:
        float: IoU value between 0 and 1, where 0 means no overlap and 1 means perfect overlap
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # IoU
    return inter_area / float(box_a_area + box_b_area - inter_area)


# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========


def track_shot(
    opt: argparse.Namespace, scene_faces: list[FrameDetection]
) -> list[TrackDict]:
    """
    Track faces across frames within a scene shot.

    This function identifies continuous face tracks by linking face detections
    across consecutive frames based on IoU (Intersection over Union) overlap.

    Args:
        opt: Command line arguments containing tracking parameters
        scene_faces: List of face detections for each frame in the scene

    Returns:
        list[TrackDict]: List of face tracks, each containing frame numbers and bounding boxes
    """
    iou_threshold = 0.5  # Minimum IOU between consecutive face detections
    tracks: list[TrackDict] = []

    while True:
        track: list[FaceDict] = []
        for frame_faces in scene_faces:
            for face in frame_faces:
                if track == []:
                    track.append(face)
                    frame_faces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= opt.num_failed_det:
                    iou = bb_intersection_over_union(face["bbox"], track[-1]["bbox"])
                    if iou > iou_threshold:
                        track.append(face)
                        frame_faces.remove(face)
                        continue
                else:
                    break

        if track == []:
            break
        if len(track) > opt.min_track:
            frame_num = np.array([f["frame"] for f in track])
            bboxes = np.array([np.array(f["bbox"]) for f in track])

            frame_i = np.arange(frame_num[0], frame_num[-1] + 1)

            bboxes_i: list[np.ndarray] = []
            for ij in range(4):
                interp_fn = interp1d(frame_num, bboxes[:, ij])
                bboxes_i.append(interp_fn(frame_i))
            bboxes_i_stacked = np.stack(bboxes_i, axis=1)

            if (
                max(
                    np.mean(bboxes_i_stacked[:, 2] - bboxes_i_stacked[:, 0]),
                    np.mean(bboxes_i_stacked[:, 3] - bboxes_i_stacked[:, 1]),
                )
                > opt.min_face_size
            ):
                tracks.append({"frame": frame_i, "bbox": bboxes_i_stacked})

    return tracks


# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========


def crop_video(
    opt: argparse.Namespace, track: TrackDict, crop_file: str
) -> ProcTrackDict:
    """
    Crop a face track from video frames and create a new video with the cropped face.

    This function:
    1. Extracts face regions from video frames based on tracking data
    2. Applies smoothing to the face track coordinates
    3. Creates a new video with the cropped face
    4. Extracts and adds the corresponding audio segment

    Args:
        opt: Command line arguments containing processing parameters
        track: Face track data with frame numbers and bounding boxes
        crop_file: Base path for the output video file

    Returns:
        ProcTrackDict: Dictionary containing the original track and processed track data
    """
    flist = glob.glob(os.path.join(opt.frames_dir, opt.reference, "*.jpg"))
    flist.sort()

    four_cc = cv2.VideoWriter_fourcc(*"XVID")
    v_out = cv2.VideoWriter(crop_file + "t.avi", four_cc, opt.frame_rate, (224, 224))

    dets: DetDict = {"x": [], "y": [], "s": []}

    for det in track["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)  # crop center x
        dets["x"].append((det[0] + det[2]) / 2)  # crop center y

    # Smooth detections
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

    for fidx, frame in enumerate(track["frame"]):
        cs = opt.crop_scale

        bs = dets["s"][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount

        image = cv2.imread(flist[int(frame)])

        padded_frame = np.pad(
            image,
            ((bsi, bsi), (bsi, bsi), (0, 0)),
            "constant",
            constant_values=(110, 110),
        )
        my = dets["y"][fidx] + bsi  # BBox center Y
        mx = dets["x"][fidx] + bsi  # BBox center X

        face = padded_frame[
            int(my - bs) : int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
        ]

        v_out.write(cv2.resize(face, (224, 224)))

    audio_tmp = os.path.join(str(opt.tmp_dir), str(opt.reference), "audio.wav")
    audio_start = (track["frame"][0]) / opt.frame_rate
    audio_end = (track["frame"][-1] + 1) / opt.frame_rate

    v_out.release()

    # ========== CROP AUDIO FILE ==========

    command = [
        "ffmpeg",
        "-y",
        "-i",
        os.path.join(opt.avi_dir, opt.reference, "audio.wav"),
        "-ss",
        f"{audio_start:.3f}",
        "-to",
        f"{audio_end:.3f}",
        f"{audio_tmp}",
    ]
    output = subprocess.call(command, stdout=None)

    if output != 0:
        logging.error(f"Failed to crop audio file {audio_tmp}")
        sys.exit(output)

    # ========== COMBINE AUDIO AND VIDEO FILES ==========

    command = (
        "ffmpeg",
        "-y",
        "-i",
        f"{crop_file}t.avi",
        "-i",
        f"{audio_tmp}",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        f"{crop_file}.avi",
    )
    output = subprocess.call(command, stdout=None)

    if output != 0:
        logging.error(f"Failed to combine audio and video files {crop_file}")
        sys.exit(output)

    logging.info(f"Written {crop_file}")

    os.remove(crop_file + "t.avi")

    logging.info(
        f"Mean pos: x {np.mean(dets['x']):.2f} y {np.mean(dets['y']):.2f} "
        f"s {np.mean(dets['s']):.2f}"
    )

    return {"track": track, "proc_track": dets}


# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========


def inference_video(opt: argparse.Namespace) -> list[FrameDetection]:
    """
    Perform face detection on all frames of a video.

    This function:
    1. Loads each frame from the extracted frames directory
    2. Detects faces in each frame using the S3FD face detector
    3. Saves the detection results to a pickle file

    Args:
        opt: Command line arguments containing detection parameters

    Returns:
        list[FrameDetection]: List of face detections for each frame
    """
    detector = S3FD(device="cuda")

    flist = glob.glob(os.path.join(opt.frames_dir, opt.reference, "*.jpg"))
    flist.sort()

    dets: list[FrameDetection] = []

    for f_idx, f_name in enumerate(flist):
        start_time = time.time()

        image = cv2.imread(f_name)

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = detector.detect_faces(
            image_np, confidence_threshold=0.9, scales=[opt.facedet_scale]
        )

        frame_det: FrameDetection = []
        for bbox in bboxes:
            frame_det.append(
                {"frame": f_idx, "bbox": (bbox[:-1]).tolist(), "conf": float(bbox[-1])}
            )
        dets.append(frame_det)

        elapsed_time = time.time() - start_time

        logging.info(
            f"{os.path.join(opt.avi_dir, opt.reference, 'video.avi')}-{f_idx:05d}; "
            f"{len(dets[-1])} dets; {(1 / elapsed_time):.2f} Hz"
        )

    save_path = os.path.join(opt.work_dir, opt.reference, "faces.pckl")

    with open(save_path, "wb") as fil:
        pickle.dump(dets, fil)

    return dets


# ========== ========== ========== ==========
# # SCENE DETECTION
# ========== ========== ========== ==========


def scene_detect(opt: argparse.Namespace) -> list[tuple]:
    """
    Detect scene changes in a video.

    This function:
    1. Uses the PySceneDetect library to identify scene boundaries
    2. Saves the scene detection results to a pickle file

    Args:
        opt: Command line arguments containing video paths

    Returns:
        list[tuple]: List of scene boundaries, each represented as a tuple of start and end timecodes
    """
    video_manager = VideoManager(
        [os.path.join(opt.avi_dir, opt.reference, "video.avi")]
    )
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    # Add ContentDetector algorithm
    # (constructor takes detector options like `threshold`).
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    video_manager.set_downscale_factor()

    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list(base_timecode)

    save_path = os.path.join(opt.work_dir, opt.reference, "scene.pckl")

    if scene_list == []:
        scene_list = [
            (video_manager.get_base_timecode(), video_manager.get_current_timecode())
        ]

    with open(save_path, "wb") as fil:
        pickle.dump(scene_list, fil)

    logging.info(
        f"{os.path.join(opt.avi_dir, opt.reference, 'video.avi')} "
        f"- scenes detected {len(scene_list)}"
    )

    return scene_list


# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========


def main():
    """
    Execute the complete face extraction pipeline.

    This function orchestrates the entire process:
    1. Parses command line arguments
    2. Creates necessary directories
    3. Converts the input video to a standard format
    4. Extracts video frames and audio
    5. Performs face detection on all frames
    6. Detects scene boundaries
    7. Tracks faces across frames within each scene
    8. Crops face tracks and creates individual face videos
    9. Saves the results
    """
    opt = parse_args()

    # ========== DELETE EXISTING DIRECTORIES ==========
    # ========== MAKE NEW DIRECTORIES ==========

    sys_dirs = (
        os.path.join(opt.work_dir, opt.reference),
        os.path.join(opt.crop_dir, opt.reference),
        os.path.join(opt.avi_dir, opt.reference),
        os.path.join(opt.frames_dir, opt.reference),
        os.path.join(opt.tmp_dir, opt.reference),
    )
    for dir_ in sys_dirs:
        if os.path.exists(dir_):
            rmtree(dir_)
        os.makedirs(dir_)

    # ========== CONVERT VIDEO AND EXTRACT FRAMES ==========

    # Convert to 25 fps (by dropping/duplicating frames) use a
    # high-quality encoding (level 2)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        opt.videofile,
        "-qscale:v",
        "2",
        "-async",
        "1",
        "-r",
        "25",
        os.path.join(opt.avi_dir, opt.reference, "video.avi"),
    ]
    subprocess.call(command, stdout=None)

    # Extract 25fps video frames to individual JPEG files.
    command = [
        "ffmpeg",
        "-y",
        "-i",
        os.path.join(opt.avi_dir, opt.reference, "video.avi"),
        "-qscale:v",
        "2",
        "-threads",
        "1",
        "-f",
        "image2",
        os.path.join(opt.frames_dir, opt.reference, "%06d.jpg"),
    ]
    subprocess.call(command, stdout=None)

    # Re-sample the audio as uncompressed 16 kHz 16-bit monaural samples.
    command = [
        "ffmpeg",
        "-y",
        "-i",
        os.path.join(opt.avi_dir, opt.reference, "video.avi"),
        "-ac",
        "1",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        os.path.join(opt.avi_dir, opt.reference, "audio.wav"),
    ]
    subprocess.call(command, stdout=None)

    # ========== FACE DETECTION ==========

    faces: list[FrameDetection] = inference_video(opt)

    # ========== SCENE DETECTION ==========

    scene: list[tuple] = scene_detect(opt)

    # ========== FACE TRACKING ==========

    all_tracks: list[TrackDict] = []
    vid_tracks: list[ProcTrackDict] = []

    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= opt.min_track:
            all_tracks.extend(
                track_shot(opt, faces[shot[0].frame_num : shot[1].frame_num])
            )

    # ========== FACE TRACK CROP ==========

    for ii, track in enumerate(all_tracks):
        vid_tracks.append(
            crop_video(
                opt,
                track,
                os.path.join(str(opt.crop_dir), str(opt.reference), f"{ii:05d}"),
            )
        )

    # ========== SAVE RESULTS ==========

    save_path = os.path.join(opt.work_dir, opt.reference, "tracks.pckl")

    with open(save_path, "wb") as fil:
        pickle.dump(vid_tracks, fil)

    rmtree(os.path.join(opt.tmp_dir, opt.reference))


if __name__ == "__main__":
    main()
