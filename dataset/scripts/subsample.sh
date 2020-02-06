#!/bin/bash


dataPath=${1}
saveFreq=${2-1}

if [ -d "$dataPath" ]; then
  OLDPATH=$(pwd)
  cd "$dataPath"

  echo "Subsampling $dataPath with saveFreq = $saveFreq"
  # This nice line saves every nth file in a directory.
  ls | awk "(NR - 1) % $saveFreq != 0" | xargs -I {} rm -vf "{}"

  cd "$OLDPATH"

else
  echo "Folder $dataPath doesn't exist. Subsampling exiting."
fi
