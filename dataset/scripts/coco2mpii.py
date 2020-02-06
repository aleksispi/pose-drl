# !/bin/python3
"""
Takes an panoptic coco19 annotation prints the corresponding mpii annotation.

Which, by comparing scenes that have both annotations, should just be the first 15 annotations.
"""

import json
import sys
import glob
import os

MPII_NECK = 0
MPII_HEADTOP = 1
MPII_BODYCENTER = 2
MPII_LSHOULDER = 3
MPII_LELBOW = 4
MPII_LWRIST = 5
MPII_LHIP = 6
MPII_LKNEE = 7
MPII_LANKLE = 8
MPII_RSHOULDER = 9
MPII_RELBOW = 10
MPII_RWRIST = 11
MPII_RHIP = 12
MPII_RKNEE = 13
MPII_RANKLE = 14


def convert(file_path):
    with open(file_path, 'r') as f:
        annot = json.load(f)

    new_bodies = []
    for body in annot['bodies']:
        new_bodies.append({
            'id': body['id'],
            'joints15': body['joints19'][:-4*4] # Remove 4 last keypoints (each having x,y,z,c)
        })
    annot['bodies'] = new_bodies
    annot['convertor'] = 'coco2mpii.py'
    return json.dumps(annot, indent=None)


if __name__ == '__main__':
    file_path = sys.argv[1]
    out_path = sys.argv[2]

    files = glob.glob(os.path.join(file_path, '*.json'))

    for path in files:
        annot_path = os.path.join(file_path, os.path.basename(path))
        out_file = os.path.join(out_path, os.path.basename(path))
        new_annot = convert(annot_path)

        with open(out_file, 'w') as f:
            f.write(new_annot)
        print('Converted Coco19 to MPII %s -> %s' % (annot_path, out_file), file=sys.stderr)
