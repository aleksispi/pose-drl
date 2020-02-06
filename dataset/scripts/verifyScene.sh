#!/bin/bash
# Verifies that all data exists

scenePath=$1
sceneName=$(basename $scenePath)

echo ""
echo "======| Verifying $sceneName |======"

# Figure out the path of helper scripts
DIR=$(dirname $(readlink -f $0))
OLDDIR=$PWD

cd $scenePath

if [ ! -s "calibration_$sceneName.json" ]; then
  echo "--> Camera calibrations missing"
fi

if [ -d vgaImgs  ]; then
  vgaCount=$(find vgaImgs -maxdepth 2 -type f|wc -l)
  vgaCamCount=$(ls vgaImgs/ |wc -l)
else
  vgaCount=0
  vgaCamCount=0
fi

if [ -d vgaPose3d_stage1  ]; then
  vgaAnnotCount=$(find vgaPose3d_stage1/ -maxdepth 1 -type f|wc -l)
else
  vgaAnnotCount=0

  if [ ! -s vgaPose3d_stage1.tar ]; then
    echo "--> VGA annots tar missing"
  fi
fi

if [ -d hdImgs ]; then
  hdCount=$(find hdImgs -maxdepth 2 -type f|wc -l)
  hdCamCount=$(ls hdImgs/ |wc -l)
else
  hdCount=0
  hdCamCount=0
fi

if [ -d hdPose3d_stage1  ]; then
  hdAnnotCount=$(find hdPose3d_stage1 -maxdepth 1 -type f|wc -l)
else
  hdAnnotCount=0

  if [ ! -s hdPose3d_stage1.tar ]; then
    echo "--> HD annots tar missing"
  fi

  if [ ! -s hdPose3d_stage1_coco19.tar ]; then
    echo "--> HD coco19 annots tar missing"
  fi
fi

echo "VGA images: $vgaCount cameras: $vgaCamCount annots: $vgaAnnotCount"
echo "HD images: $hdCount cameras: $hdCamCount annots: $hdAnnotCount"

if [ $(($vgaCamCount * $vgaAnnotCount)) -ne $vgaCount ]; then
  echo "--> Nbr of VGA images and annots do not match, fix using clean_up_panoptic.py script."
fi

if [ $(($hdCamCount * $hdAnnotCount)) -ne $hdCount ]; then
  echo "--> Nbr of HD images and annots do not match, fix using clean_up_panoptic.py script."
fi

cd $OLDDIR
