#!/bin/bash
# Helper script to run other extraction tasks
# Input argument is output format for image files (png or jpg)
#
# Extracts all annotations and converts coco19 to MPII format.

# Format for extracted images.
# Use png for best quality.
scenePath=$1
saveFreq=${2-1}
fmt=${3-jpg}

echo "Extracting $scenePath, saveFreq = $saveFreq"

# Figure out the path of helper scripts
DIR=$(dirname $(readlink -f $0))
OLDDIR=$PWD

cd "$scenePath"

# Extract VGA images
$DIR/vgaImgsExtractor.sh ${fmt} ${saveFreq}

# Extract HD images
$DIR/hdImgsExtractor.sh ${fmt} ${saveFreq}

# Extract skeletons
if [ -f vgaPose3d_stage1.tar ]; then
	tar -xf vgaPose3d_stage1.tar
  python3 $DIR/discard_annotations.py $scenePath/vgaImgs $scenePath/vgaPose3d_stage1
fi

if [ -f hdPose3d_stage1.tar ]; then
	tar -xf hdPose3d_stage1.tar
  python3 $DIR/discard_annotations.py $scenePath/hdImgs $scenePath/hdPose3d_stage1
fi

# Extract 3D Keypoints
if [ -f vgaPose3d_stage1_coco19.tar ]; then
	tar -xf vgaPose3d_stage1_coco19.tar
  python3 $DIR/discard_annotations.py $scenePath/vgaImgs $scenePath/vgaPose3d_stage1_coco19
fi

if [ -f hdPose3d_stage1_coco19.tar ]; then
	tar -xf hdPose3d_stage1_coco19.tar
  python3 $DIR/discard_annotations.py $scenePath/hdImgs $scenePath/hdPose3d_stage1_coco19
fi

# If coco19 annotations exists, convert them to MPII. Overwrite any existing
# annotations from MPII.
if [ -d $scenePath/vgaPose3d_stage1_coco19 ]; then
	mv vgaPose3d_stage1 vgaPose3d_stage1_orig
	mkdir vgaPose3d_stage1

	python3 $DIR/coco2mpii.py $scenePath/vgaPose3d_stage1_coco19 $scenePath/vgaPose3d_stage1
fi

if [ -d $scenePath/hdPose3d_stage1_coco19 ]; then
	mv hdPose3d_stage1 hdPose3d_stage1_orig
	mkdir hdPose3d_stage1

	python3 $DIR/coco2mpii.py $scenePath/hdPose3d_stage1_coco19 $scenePath/hdPose3d_stage1
fi

cd "$OLDDIR"
