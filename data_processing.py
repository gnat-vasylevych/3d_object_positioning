import os
import numpy as np
from PIL import Image
import tensorflow as tf
from objectron.schema import annotation_data_pb2 as annotation_protocol
from sklearn.model_selection import train_test_split
from download_dataset import get_video_ids


def get_frame_annotation(sequence, frame_id):
    """Grab an annotated frame from the sequence."""
    data = sequence.frame_annotations[frame_id]
    object_id = 0
    object_keypoints_2d = []
    object_keypoints_3d = []
    object_rotations = []
    object_translations = []
    object_scale = []
    num_keypoints_per_object = []
    object_categories = []
    annotation_types = []
    # Get the camera for the current frame. We will use the camera to bring
    # the object from the world coordinate to the current camera coordinate.
    camera = np.array(data.camera.transform).reshape(4, 4)

    for obj in sequence.objects:
        rotation = np.array(obj.rotation).reshape(3, 3)
        translation = np.array(obj.translation)
        object_scale.append(np.array(obj.scale))
        transformation = np.identity(4)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = translation
        obj_cam = np.matmul(camera, transformation)
        object_translations.append(obj_cam[:3, 3])
        object_rotations.append(obj_cam[:3, :3])
        object_categories.append(obj.category)
        annotation_types.append(obj.type)

    keypoint_size_list = []
    for annotations in data.annotations:
        num_keypoints = len(annotations.keypoints)
        keypoint_size_list.append(num_keypoints)
        for keypoint_id in range(num_keypoints):
            keypoint = annotations.keypoints[keypoint_id]
            object_keypoints_2d.append(
                (keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
            object_keypoints_3d.append(
                (keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))
        num_keypoints_per_object.append(num_keypoints)
        object_id += 1
    return (object_keypoints_2d, object_categories, keypoint_size_list,
            annotation_types)


def resize_image(image, new_size: tuple):
    return np.array(tf.image.resize(image, new_size)).astype(int)


def build_train_validation_test_dataset():
    video_ids = get_video_ids()
    video_ids = [video for video in video_ids if len(video) != 0]
    train_video_ids, test_video_ids = train_test_split(video_ids, test_size=0.2, random_state=0)
    train_video_ids, validation_video_ids = train_test_split(train_video_ids, test_size=0.1, random_state=0)
    TRAIN_LEN = 10#len(train_video_ids)
    VAL_LEN = 10#len(validation_video_ids)
    TEST_LEN = 10#len(test_video_ids)
    parent_dir = os.getcwd()
    annotation_dir = os.path.join(parent_dir, "cup_annotations\\")
    frame_dir = os.path.join(parent_dir, "cup_annotations_frames\\")
    frame_id = 10

    def yield_files(frame_filename, annotation_file):
        with open(annotation_file, 'rb') as pb:
            sequence = annotation_protocol.Sequence()
            sequence.ParseFromString(pb.read())
            image = Image.open(frame_filename)
            frame = np.asarray(image)
            annotation, cat, num_keypoints, types = get_frame_annotation(sequence, frame_id)


            frame = resize_image(frame, new_size=(224, 224))
            frame = frame.astype('float32') / 255

            annotation = np.array(annotation).flatten()

            return (frame, annotation, num_keypoints)

    def train_generator():
        for i in range(TRAIN_LEN):
            frame_filename = frame_dir + "{}/frame.png".format(train_video_ids[i]).replace('/', '_')
            annotation_file = annotation_dir + "{}/annotation.pbdata".format(train_video_ids[i]).replace('/', '_')

            frame, annotation, num_keypoints = yield_files(frame_filename, annotation_file)

            if len(num_keypoints) <= 1:
                yield frame, annotation

    def validation_generator():
        for i in range(VAL_LEN):
            frame_filename = frame_dir + "{}/frame.png".format(validation_video_ids[i]).replace('/', '_')
            annotation_file = annotation_dir + "{}/annotation.pbdata".format(validation_video_ids[i]).replace('/', '_')

            frame, annotation, num_keypoints = yield_files(frame_filename, annotation_file)

            if len(num_keypoints) <= 1:
                yield frame, annotation

    def test_generator():
        for i in range(TEST_LEN):
            frame_filename = frame_dir + "{}/frame.png".format(test_video_ids[i]).replace('/', '_')
            annotation_file = annotation_dir + "{}/annotation.pbdata".format(test_video_ids[i]).replace('/', '_')

            frame, annotation, num_keypoints = yield_files(frame_filename, annotation_file)

            if len(num_keypoints) <= 1:
                yield frame, annotation

    train_dataset = tf.data.Dataset.from_generator(train_generator, output_types=(tf.float32, tf.float32))
    validation_dataset = tf.data.Dataset.from_generator(validation_generator, output_types=(tf.float32, tf.float32))
    test_dataset = tf.data.Dataset.from_generator(test_generator, output_types=(tf.float32, tf.float32))

    return train_dataset, validation_dataset, test_dataset

