"""
Clean-up script for the Panoptic dataset we use in this project.
Can:
    - Remove frames missing annotations (humans out of view)
    - Remove bad ffmpeg frames that are only green.
    - Can keep only sequences of continuously annotated frames.
"""
from glob import glob
import os
import sys
from multiprocessing import Pool
import signal
import argparse
import json
import shutil
import re

from PIL import Image
import numpy as np

DELETE = False
FRAME_REGEX = re.compile(r'(\d+)\.\w+$')


def is_green_frame(frame_path):
    """
    Some frames might fail and contain only green.
    """
    img = Image.open(frame_path)

    # Sample down to 50x50 instead of HD for speed!
    img = img.crop((0, 0, 50, 50))

    # convert to HSV for easier color separation
    img = img.convert('HSV')

    # to numpy for faster computations
    img_data = np.asarray(img, dtype=np.uint8)

    img.close()

    mean_hsv = np.mean(np.mean(img_data, 1), 0)
    mean_hue = mean_hsv[0]
    mean_sat = mean_hsv[1]
    mean_value = mean_hsv[2]
    if 80 < mean_hue < 135 and mean_sat > 240 and mean_value > 120:
        return True

    return False


def frame_to_frame_id(frame):
    """
    Extracts the frame id / number from a frame filename.
    """
    return FRAME_REGEX.findall(frame)[0]


def get_ids_in_annotations(scene, frame, quality):
    """
    Returns a set of all ids found in annotations.
    """
    annotations_path = os.path.join(scene, '%sPose3d_stage1' % quality,
                                    'body3DScene_%s.json' % frame)

    if not os.path.exists(annotations_path):
        return set()

    with open(annotations_path, 'r') as f:
        annots = json.load(f)

    return set([b['id'] for b in annots['bodies']])


def is_bad_frame(scene, frame, camera, quality, check_green_frames,
                 check_empty_frame, min_nbr_people):
    """
    Check a specific camera frame scene if it is bad.
    """
    bad = False
    path = os.path.join(scene, '%sImgs' % quality, camera,
                        camera + '_' + frame + '.jpg')

    # Should be checked in order of increasing computational load.
    ids = get_ids_in_annotations(scene, frame, quality)
    if check_empty_frame and len(ids) < min_nbr_people:
        return True, ids

    if check_green_frames and is_green_frame(path):
        return True, ids

    return False, ids


def delete_framescene(scene, frame, cameras, quality):
    """
    Delete all frame scenes.
    """
    print('Deleting scene: %s frame: %s' % (os.path.basename(scene), frame))
    for camera in cameras:
        img_path = os.path.join(scene, '%sImgs' % quality, camera,
                                camera + '_' + frame + '.jpg')
        if DELETE:
            try:
                os.remove(img_path)
            except OSError as e:
                print('=> Error while deleting: %s' % img_path)

    annotations_path = os.path.join(scene, '%sPose3d_stage1' % quality,
                                    'body3DScene_%s.json' % frame)

    if DELETE:
        try:
            os.remove(annotations_path)
        except OSError as e:
            print('=> Error while deleting: %s' % annotations_path)


def check_framescene(scene, frame, cameras, quality, check_green_frames,
                     check_empty_frame, min_nbr_people):
    """
    Checks a frame for all cameras in one scene.
    Returns True if all frame is good.
    Returns False if any frame was bad.
    """
    for camera in cameras:
        bad, ids = is_bad_frame(scene, frame, camera, quality, check_green_frames,
                                check_empty_frame, min_nbr_people)
        if bad:
            return False, ids
    return True, ids


def list_frames(scene, cameras, quality, check_reduce_to_intersection):
    """
    Returns the intersection of all frames for all cameras.
    """
    union = set()
    intersection = None
    for camera in cameras:
        # List all frames (assumed to be same for all scenes)
        camera_path = os.path.join(scene, '%sImgs' % quality, camera)
        frames = set([frame_to_frame_id(f) for f in os.listdir(camera_path)])
        union = union | frames
        intersection = intersection & frames if intersection is not None else frames

    if check_reduce_to_intersection:
        for frame in union - intersection:
            delete_framescene(scene, frame, cameras, quality)

    return sorted(list(intersection))


def move_file(src, dest):
    """
    Moves or copies depending on the DELETE flag.
    """
    try:
        if DELETE:
            shutil.move(src, dest)
        else:
            shutil.copy(src, dest)
    except:
        print('Failed to move file %s' % src)


def split_scene(panoptic_path, scene, cameras, quality, good_sequences):
    """
    Splits a scenes into multiple scenes.
    """
    if len(good_sequences) == 0:
        return

    print('Splitting %s' % scene)
    for idx, sequence in enumerate(good_sequences):
        old_scene_name = os.path.basename(scene)
        new_scene_name = old_scene_name + '_%d' % (idx + 1)
        new_scene_path = os.path.join(panoptic_path, new_scene_name)
        os.mkdir(new_scene_path)

        for fp in os.listdir(scene):
            if fp.endswith('.tar'):
                # Copy calibration, tars etc.
                shutil.copy(os.path.join(scene, fp), os.path.join(new_scene_path, fp))

        # Copy and rename calibrations
        calib_src = os.path.join(scene, 'calibration_%s.json' % old_scene_name)
        calib_dst = os.path.join(new_scene_path, 'calibration_%s.json' % new_scene_name)
        shutil.copy(calib_src, calib_dst)

        # Transfer images and annotations
        os.mkdir(os.path.join(new_scene_path, '%sPose3d_stage1' % quality))
        for frame_id in sequence:
            print('Moving frame %s' % frame_id)
            # Transfer annotation
            annot = os.path.join('%sPose3d_stage1' % quality, 'body3DScene_%s.json' % frame_id)
            move_file(os.path.join(scene, annot), os.path.join(new_scene_path, annot))

            # Transfer images
            for camera in cameras:
                os.makedirs(os.path.join(new_scene_path, '%sImgs' % quality, camera), exist_ok=True)
                img = os.path.join('%sImgs' % quality, camera, camera + '_' + frame_id + '.jpg')
                move_file(os.path.join(scene, img), os.path.join(new_scene_path, img))

    # Remove old scene folder
    print("Remove leftover files from splitting scene")
    old_scene_name = os.path.basename(scene)
    old_scene_path = os.path.join(panoptic_path, old_scene_name)
    shutil.rmtree(old_scene_path)


def check_consecutive_frame_sequences(scene, frames, cameras, quality,
                                      check_green_frames, check_empty_frame,
                                      min_seq_length, same_people, min_nbr_people):
    """
    Checks a scene for the long consecutive sequence of good frames. Removes the
    rest. Returns list of good sequences.
    """
    pool = Pool()
    good_frames = [pool.apply_async(check_framescene,
                                    args=(scene, f, cameras, quality,
                                          check_green_frames, check_empty_frame,
                                          min_nbr_people))
                   for f in frames]
    sys.stdout.flush()
    pool.close()
    pool.join()
    good_frames = [future.get(timeout=1) for future in good_frames]

    bad_frames = [] # list of bad frames []
    good_sequences = [] # list of good sequences [[], []]
    curr_sequence = []
    curr_ids = set()
    for i in range(len(good_frames)):
        is_good, ids = good_frames[i]

        if same_people:
            # Checks whether the next frame contains the same people.
            # If not we end the sequence here
            if len(curr_ids) == 0:
                curr_ids = ids
            elif curr_ids != ids:
                # New set of people. This ends the sequence
                if len(curr_sequence) >= min_seq_length:
                    good_sequences.append(curr_sequence)
                else:
                    bad_frames.extend(curr_sequence)
                curr_sequence = []
                curr_ids = set()
                continue

        if is_good:
            curr_sequence.append(frames[i])

        else:
            bad_frames.append(frames[i])

            # End of sequence of good frames
            if len(curr_sequence) >= min_seq_length:
                good_sequences.append(curr_sequence)
            else:
                bad_frames.extend(curr_sequence)
            curr_sequence = []
            curr_ids = set()

    for frame in bad_frames:
        delete_framescene(scene, frame, cameras, quality)
    print('Found %d good sequences containing %d frames, removed %d frames.' %
          (len(good_sequences), len(frames) - len(bad_frames), len(bad_frames)))
    return good_sequences


def main(panoptic_path, quality, check_green_frames, check_empty_frame,
         min_consecutive, check_same_nbr_frames, do_split_scene, same_people,
         min_nbr_people):

    scenes = sorted(glob(os.path.join(panoptic_path, '*')))
    for scene in scenes:
        print('Checking %s' % scene)

        # List all cameras in that scene
        camera_path = os.path.join(scene, '%sImgs' % quality)
        if not os.path.exists(camera_path):
            print('No cameras for %s. Skipping!' % scene)
            continue

        cameras = os.listdir(camera_path)

        # List all frames (assumed to be same for all scenes)
        camera_path = os.path.join(scene, '%sImgs' % quality, cameras[0])
        frames = list_frames(scene, cameras, quality, check_same_nbr_frames)

        good_sequences = check_consecutive_frame_sequences(scene, frames, cameras, quality,
                                                           check_green_frames, check_empty_frame,
                                                           min_consecutive, same_people,
                                                           min_nbr_people)

        if do_split_scene:
            split_scene(panoptic_path, scene, cameras, quality, good_sequences)


def parse_args():
    parser = argparse.ArgumentParser(description='Panoptic Cleanup Script. Uses multiprocessing, stop using "killall python"')
    parser.add_argument('path', help='Path to dataset')
    parser.add_argument('--hd', action='store_const', help='Format is hd cameras', default='vga', const='hd')
    parser.add_argument('--check-green-frame', action='store_true', help='Check for green and broken frames', default=False)
    parser.add_argument('--check-empty-frame', action='store_true', help='Check for empty frames (missing annotions)', default=False)
    parser.add_argument('--min-consecutive', action='store', default=0, type=int,
                        help='Checks for empty frames and keep only long consecutive sequences of annotated frames for the each scene. Value is minimum sequence length. Default is 0.')
    parser.add_argument('--check-same-nbr-frames', action='store_true', help='Ensures that all cameras within a scenes has the same number of frames.', default=False)
    parser.add_argument('--delete', action='store_true', help='Actually perfom the delete operation, not just list files to be deleted.', default=False)
    parser.add_argument('--split-scene', action='store_true', help='Each sequence of good frames get it\'s own sub scene. With delete files are moved, without files are copied.', default=False)
    parser.add_argument('--same-people', action='store_true', help='A scene/subscene must have the same people in each frame.', default=False)
    parser.add_argument('--min-nbr-people', action='store', default=1, type=int,
                        help='If checking for empty annotation, specifies the minimum number of people to count as non empty.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    DELETE = args.delete
    main(args.path, args.hd, args.check_green_frame, args.check_empty_frame,
         args.min_consecutive, args.check_same_nbr_frames, args.split_scene,
         args.same_people, args.min_nbr_people)
