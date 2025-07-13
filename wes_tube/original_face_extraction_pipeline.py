#!/usr/bin/python

import argparse
import glob
import os
import pdb
import pickle
import subprocess
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
from scipy.io import wavfile

Num = int | float
BoundingBox = tuple[Num, Num, Num, Num] | list[Num]


class FaceDict(TypedDict):
    frame: int
    bbox: BoundingBox
    conf: float


class TrackDict(TypedDict):
    frame: np.ndarray
    bbox: np.ndarray


class DetDict(TypedDict):
    """A detection object"""

    x: list[float]
    y: list[float]
    s: list[float]


class ProcTrackDict(TypedDict):
    track: TrackDict
    proc_track: DetDict


# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========
def parse_args() -> argparse.Namespace:
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
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou


# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========


def track_shot(
    opt: argparse.Namespace, scene_faces: list[list[FaceDict]]
) -> list[TrackDict]:
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

    command = (
        f"ffmpeg -y "
        f"-i {os.path.join(opt.avi_dir, opt.reference, 'audio.wav')} "
        f"-ss {audio_start:.3f} "
        f"-to {audio_end:.3f} "
        f"{audio_tmp}"
    )
    output = subprocess.call(command, shell=True, stdout=None)

    if output != 0:
        pdb.set_trace()

    sample_rate, audio = wavfile.read(audio_tmp)

    # ========== COMBINE AUDIO AND VIDEO FILES ==========

    command = (
        f"ffmpeg -y "
        f"-i {crop_file}t.avi "
        f"-i {audio_tmp} "
        f"-c:v copy -c:a copy {crop_file}.avi"
    )
    output = subprocess.call(command, shell=True, stdout=None)

    if output != 0:
        pdb.set_trace()

    print(f"Written {crop_file}")

    os.remove(crop_file + "t.avi")

    print(
        f"Mean pos: x {np.mean(dets['x']):.2f} y {np.mean(dets['y']):.2f} "
        f"s {np.mean(dets['s']):.2f}"
    )

    return {"track": track, "proc_track": dets}


# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========


def inference_video(opt: argparse.Namespace) -> list[list[FaceDict]]:
    detector = S3FD(device="cuda")

    flist = glob.glob(os.path.join(opt.frames_dir, opt.reference, "*.jpg"))
    flist.sort()

    dets: list[list[FaceDict]] = []

    for f_idx, f_name in enumerate(flist):
        start_time = time.time()

        image = cv2.imread(f_name)

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = detector.detect_faces(
            image_np, confidence_threshold=0.9, scales=[opt.facedet_scale]
        )

        dets.append([])
        for bbox in bboxes:
            dets[-1].append(
                {"frame": f_idx, "bbox": (bbox[:-1]).tolist(), "conf": float(bbox[-1])}
            )

        elapsed_time = time.time() - start_time

        print(
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

    print(
        f"{os.path.join(opt.avi_dir, opt.reference, 'video.avi')} "
        f"- scenes detected {len(scene_list)}"
    )

    return scene_list


# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========


def main():
    """Execute demo."""
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

    command = (
        f"ffmpeg -y "
        f"-i {opt.videofile} "
        f"-qscale:v 2 -async 1 -r 25 "
        f"{os.path.join(opt.avi_dir, opt.reference, 'video.avi')}"
    )
    subprocess.call(command, shell=True, stdout=None)

    command = (
        f"ffmpeg -y "
        f"-i {os.path.join(opt.avi_dir, opt.reference, 'video.avi')} "
        f"-qscale:v 2 -threads 1 -f image2 "
        f"{os.path.join(opt.frames_dir, opt.reference, '%06d.jpg')}"
    )
    subprocess.call(command, shell=True, stdout=None)

    command = (
        f"ffmpeg -y "
        f"-i {os.path.join(opt.avi_dir, opt.reference, 'video.avi')} "
        f"-ac 1 -vn -acodec pcm_s16le -ar 16000 "
        f"{os.path.join(opt.avi_dir, opt.reference, 'audio.wav')}"
    )
    subprocess.call(command, shell=True, stdout=None)

    # ========== FACE DETECTION ==========

    faces: list[list[FaceDict]] = inference_video(opt)

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
