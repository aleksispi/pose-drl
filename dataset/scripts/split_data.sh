#!/bin/bash
# This script splits the dataset into train, val and test into the same
# splits as for the NeurIPS 2019 paper.
#
# Usage: bash split_data.sh <path to dataset>
#

dataPath=${1}
cd $dataPath

# Train
echo "Creating train split..."
mkdir train
mv 141216_pose2_2 train/
mv 141217_pose2_2 train/
mv 141217_pose3_2 train/
mv 141217_pose4_3 train/
mv 160224_mafia2_16 train/
mv 160224_mafia2_17 train/
mv 160224_mafia2_3 train/
mv 160224_mafia2_4 train/
mv 160224_mafia2_5 train/
mv 160224_ultimatum1_12 train/
mv 160224_ultimatum1_14 train/
mv 160224_ultimatum1_3 train/
mv 160224_ultimatum1_4 train/
mv 160224_ultimatum1_5 train/
mv 160226_mafia1_16 train/
mv 160226_mafia1_17 train/
mv 160226_mafia1_3 train/
mv 160226_mafia1_4 train/
mv 160226_mafia1_5 train/
mv 171026_pose3_2 train/
mv 171026_pose3_3 train/
mv 171204_pose1_12 train/
mv 171204_pose1_19 train/
mv 171204_pose1_7 train/
mv 171204_pose6_3 train/

# Val
echo "Creating val split..."
mkdir val
mv 141217_pose1_3 val/
mv 141217_pose1_4 val/
mv 141217_pose1_6 val/
mv 141217_pose1_8 val/
mv 160224_mafia1_12 val/
mv 160224_mafia1_13 val/
mv 160224_mafia1_3 val/
mv 160224_mafia1_5 val/
mv 160224_mafia1_9 val/
mv 160422_ultimatum1_1 val/
mv 171026_pose1_3 val/
mv 171026_pose1_5 val/
mv 171026_pose1_7 val/
mv 171026_pose1_8 val/

# Test
echo "Creating test split..."
mkdir test
mv 141216_pose1_2 test/
mv 141216_pose1_6 test/
mv 141216_pose5_3 test/
mv 141216_pose5_6 test/
mv 141217_pose5_2 test/
mv 141217_pose5_6 test/
mv 160224_ultimatum2_10 test/
mv 160224_ultimatum2_11 test/
mv 160224_ultimatum2_4 test/
mv 160224_ultimatum2_7 test/
mv 160224_ultimatum2_8 test/
mv 160224_ultimatum2_9 test/
mv 160226_mafia2_10 test/
mv 160226_mafia2_11 test/
mv 160226_mafia2_14 test/
mv 160226_mafia2_19 test/
mv 160226_mafia2_4 test/
mv 160226_mafia2_7 test/
mv 160226_mafia2_8 test/
mv 160226_mafia2_9 test/
mv 171204_pose2_11 test/
mv 171204_pose2_3 test/
mv 171204_pose2_8 test/

# Remove all leftover folders
echo "Removing leftover scenes..."
shopt -s extglob
rm -vrf !("train"|"test"|"val")
