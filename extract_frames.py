import cv2
import numpy as np
from download_dataset import get_video_ids
import os
from PIL import Image


def grab_frame(video_file, frame_ids: list):
    """Grab an image frame from the video file."""
    frames = []
    counter = 0
    cap = cv2.VideoCapture(video_file)

    success, image = cap.read()
    if counter in frame_ids:
        image = np.rot90(image, 2)
        image = np.array(image)
        frames.append(image)
    counter += 1

    while success:
        success, image = cap.read()
        if counter in frame_ids:
            image = np.rot90(image, 2)
            image = np.array(image)
            frames.append(image)
        counter += 1

        if counter > max(frame_ids):
            break

    return frames


if __name__ == "__main__":
    video_ids = get_video_ids()
    video_ids = [video for video in video_ids if len(video) != 0]


    parent_dir = os.getcwd()
    video_directory = "cup_annotations"
    frame_dir = "cup_annotations_frames"
    path = os.path.join(parent_dir, frame_dir)

    try:
        os.mkdir(path)
    except FileExistsError:
        print("Directory 'cup_annotations_frames' already exists")

    os.chdir(path)

    i = 0
    frame = 10

    while i < len(video_ids):
        video_filename = parent_dir + '/' + video_directory + '/' + "{}/video.MOV".format(video_ids[i]).replace('/', '_')
        frames = grab_frame(video_filename, [frame])
        im = Image.fromarray(frames[0])
        im.save("{}/frame.png".format(video_ids[i]).replace('/', '_'))
        del im
        i += 1

    os.chdir(parent_dir)
