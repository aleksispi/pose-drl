# !/bin/bash
# This extracts image files from HD videos
# It should be run from the same directory containing the folder "hdVideos"
# (E.g., sampleData/ )
# It takes an optional parameter specifying the output format (png or jpg)
# See http://domedb.perception.cs.cmu.edu

fmt=${1-jpg}
saveFreq=${2-1}

DIR=$(dirname $(readlink -f $0))

inputFolderName=hdVideos
outputFolderName=hdImgs
camIdx=-1;
for p in 0
	do
	for c in {0..30}
		do
		videoFileName=$(printf "$inputFolderName/hd_%02d_%02d.mp4" $p $c)
		outputvideoFileName=$(printf "$outputFolderName/%02d_%02d" $p $c)
		if [ -f $videoFileName ]; then

			mkdir -pv $outputvideoFileName
			echo "Generate Images from $videoFileName"
			fileName=$(printf "$outputvideoFileName/%02d_%02d_%%08d.${fmt}" $p $c)
			ffmpeg -i $videoFileName -q:v 1 -f image2 -start_number 0 "$fileName"

			$DIR/subsample.sh $outputvideoFileName ${saveFreq}

			rm -vf $videoFileName
		else
			echo "$videoFileName (Skipping.)"
		fi
	done
done
