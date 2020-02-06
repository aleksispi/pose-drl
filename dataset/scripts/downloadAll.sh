#!/bin/bash

# Downloads the scenes used for the results in the paper.
# Usage:
# bash donwloadAll.sh /data/panoptic

# WARNING! The extraction process of a scene requires alot of free space.
# To be safe allow for at least 500 GB free space.

dataPath=${1}

# Set data data path

DIR=$(dirname $(readlink -f $0))

trainScenes=(
"141216_pose2" "141217_pose3" "160224_mafia2" "160224_ultimatum1" "171026_pose3" "171204_pose6"
"141217_pose2" "141217_pose4" "160226_mafia1" "171204_pose1"
)
testScenes=(
"141216_pose1" "141216_pose5"  "141217_pose5" "160224_ultimatum2" "160226_mafia2" "171204_pose2"
)
valScenes=(
"141217_pose1" "160224_mafia1" "160422_ultimatum1" "171026_pose1"
)

allScenes=("160226_mafia2") #("${trainScenes[@]}" "${testScenes[@]}" "${valScenes[@]}")
for i in "${!allScenes[@]}"; do
  p=${allScenes[$i]}
  $DIR/downloadScene.sh $p ${dataPath} 0 31 3 # 0 VGA, 31 HD, 30/3=10 fps
  $DIR/extractScene.sh "$dataPath/$p" 3 # Sample from 30 fps to 10 fps
  $DIR/verifyScene.sh $dataPath/$p
done
