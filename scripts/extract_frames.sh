#!/bin/bash

SOURCE_PATH=$1
OUTPUT_PATH=$2
extract="true"

echo "|- Processing " $SOURCE_PATH
mkdir -p $OUTPUT_PATH

for subdir in $SOURCE_PATH/*;
do
  break

  file=$subdir/*.mp4
  squash_output=$OUTPUT_PATH/$(basename $subdir).sqf
  tmp=$OUTPUT_PATH/tmp_$(basename $subdir)
  output=$tmp/frame/$(basename $subdir)

  if ls $file 1> /dev/null 2>&1; then
    if ls $subdir/*.skill_labels_by_frame.txt 1> /dev/null 2>&1; then
      if ls $subdir/THIS_DATA_SET_WAS_EXCLUDED 1> /dev/null 2>&1; then
        echo "|-- ERROR: THIS_DATA_SET_WAS_EXCLUDED $file"
      else
	      echo "|-- Extracting frames from " $file

        mkdir -p $output

        if [[ $extract == "true" ]]; then
          ffmpeg -i $file -qscale:v 2 "$output/frame_%010d.jpg" 
        else
          echo "|-- Creating SquashFS for $subdir"
          find $tmp/frame -type d -exec chmod 755 {} \;
          find $tmp/frame -type f -exec chmod 644 {} \;
 
          mksquashfs $tmp/frame $squash_output -keep-as-directory -noappend && rm -rv $tmp
        fi
      fi
    else
      echo "|-- ERROR: There is no *skill_labels_by_frame.txt file inside $subdir"
    fi
  else
    echo "|-- ERROR: There is no *mp4 file inside $subdir"
  fi
done
