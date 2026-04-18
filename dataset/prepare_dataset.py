import glob
import logging
import multiprocessing
import os.path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

def process_single_video(video, overwrite):
    video = Path(video)
    pose = video.with_suffix(".pose")

    if pose.exists() and not overwrite:
        return True

    try:
        subprocess.run(
            ['video_to_pose', "--format", "mediapipe", "-i", str(video), "-o", str(pose)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        logging.error(f"[ERROR] Can't extract video: {video.name}")
        return False
    except FileNotFoundError:
        logging.error(f"[ERROR] Tools 'video_to_pose' not found!")
        return False

def extract_keypoint(args):
    logging.info(f"Extracting keypoints from videos in {args.DATA_PATH}")
    video_item = [os.path.join(args.DATA_PATH, directory) for directory in os.listdir(args.DATA_PATH)]
    video_directory = [directory for directory in video_item if os.path.isdir(directory)]
    videos = []
    for directory in video_directory:
        for extension in [".mp4", ".avi", ".mov", ".mkv"]:
            videos.extend(glob.glob(os.path.join(directory, "*" + extension)))

    num_videos = len(videos)
    logging.info(f"Found {num_videos} videos.")

    cpu_cores = multiprocessing.cpu_count()
    max_workers = min(args.WORKERS, cpu_cores - 4)
    logging.info(f"Using {max_workers} workers.")
    overwrite_flag = getattr(args, 'OVERWRITE', False)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video, v, overwrite_flag): v for v in videos}
        success_count = 0
        for future in tqdm(as_completed(futures), total=num_videos, desc="Processing"):
            if future.result():
                success_count += 1

    logging.info(f"Processed {success_count} videos and failed {num_videos - success_count} videos.")
    logging.info(f"Extracting keypoints done.")
def prepare(args):
    logging.info(f"Preparing dataset for {args.DATA_PATH}")
    extract_keypoint(args)
