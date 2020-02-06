# !/bin/python3
"""
Subsamples the annotations in the same manner as the camera frames were sampled.
"""

import glob
import os
import re
import sys


def camera_frame_to_idx(frame_path):
    return int(re.findall(r'(\d+)\.jpg$', frame_path)[0])


def annot_to_idx(annot_path):
    return int(re.findall(r'(\d+)\.json$', annot_path)[0])


def list_frame_idx(camera_path):
    frames = glob.glob(os.path.join(camera_path, '*/*.jpg'))
    if len(frames) == 0:
        print('No frames found, possible error, exiting.')
        exit(-1)

    return [camera_frame_to_idx(f) for f in frames]


def list_annotations(annot_path):
    return sorted(glob.glob(os.path.join(annot_path, '*.json')))


def delete_annots(annots, frame_indices):
    frame_indices = set(frame_indices)

    for annot_path in annots:
        annot_idx = annot_to_idx(annot_path)
        if annot_idx not in frame_indices:
            print('Deleted %s' % annot_path)
            os.remove(annot_path)


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Usage: discard_annotations.py path-to-cameras path-to-annotations')
        exit(-1)

    print('Discarding annotations to match sampled frames...')

    camera_path = sys.argv[1]
    annotations_path = sys.argv[2]

    # List all camera frames indices
    indices = list_frame_idx(camera_path)

    # List all annotations
    annotations = list_annotations(annotations_path)

    # Delete all annots except those matching the camera frames
    delete_annots(annotations, indices)
