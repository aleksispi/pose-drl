# Human Pose RL

A repository for human pose estimation using RL (HP-RL) in the Panoptic framework (see http://domedb.perception.cs.cmu.edu/index.html).

# Repo overview

* rl-pose (repo): Contains launcher files for various things, e.g. run_viewer launches the viewer.
* hprl: Main folder for training and testing HP-RL agents etc, including viewer.
* panoptic-toolbox: Contains scripts for downloading HP-RL dataset.

## Usage

### Panoptic

See the README.md file in the panoptic-toolbox folder.

How to remove incomplete folders in panoptic:
```bash
ls */hdImgs/
rm -rf 141114_pose2 141114_pose3 141121_pose1 141121_pose2 141121_pose3 141126_pose1 141126_pose2 141126_pose3 141126_pose4 
rm -rf 140109_person1 140109_person2 140109_person3 140109_person4 140109_person5 140110_person1 140110_person2 
rm -rf 141215_pose1 
```
OR simply run
```bash
./scripts/deleteCorrupt.sh
```
