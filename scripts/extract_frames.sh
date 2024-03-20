#!/bin/bash

SOURCE_PATH=$1
OUTPUT_PATH=$2
SKILL=$3
VIDEO_DEFAULT_FRAME_RATE=30
SLOW_FAST_AUDIO_DEFAULT_SAMPLE_RATE=24000
extract="false"
type="frame" ## frame sound


echo "|- Processing " $SOURCE_PATH
mkdir -p $OUTPUT_PATH

for subdir in $SOURCE_PATH/*;
do
#break

  file=$subdir/*.mp4
  name=$(basename $subdir)
  squash_output=$OUTPUT_PATH/$name.sqf
  squash_output_sound=$OUTPUT_PATH/$SKILL.sqf
  tmp=$OUTPUT_PATH/tmp_$name
  output=$tmp/frame/$name

  if ls $file 1> /dev/null 2>&1; then
    if ls $subdir/*.skill_labels_by_frame.txt 1> /dev/null 2>&1; then
      if ls $subdir/THIS_DATA_SET_WAS_EXCLUDED 1> /dev/null 2>&1; then
        echo "|-- ERROR: THIS_DATA_SET_WAS_EXCLUDED $subdir"
      else
        file_rows=`cat $subdir/$name.skill_labels_by_frame.txt | wc -l`

        if [[ $file_rows == 0 ]]; then
          echo "|-- ERROR: *skill_labels_by_frame.txt has no lines $subdir"
        else  
          echo "|-- Extracting frames from " $file

          if [[ $type == "frame" ]]; then
            mkdir -p $output
          fi

          if [[ $extract == "true" ]]; then
            if [[ $type == "frame" ]]; then
#            ffmpeg -i $file -qscale:v 2 "$output/frame_%010d.jpg"
              ffmpeg -i $file -filter:v fps=$VIDEO_DEFAULT_FRAME_RATE -qscale:v 2 "$output/frame_%010d.jpg" 
            else
              ffmpeg -i $file -vn -acodec pcm_s16le -ac 1 -ar $SLOW_FAST_AUDIO_DEFAULT_SAMPLE_RATE  $OUTPUT_PATH/$name.wav
            fi
          else
            echo "|-- Creating SquashFS for $subdir"

            if [[ $type == "frame" ]]; then
              find $tmp/frame -type d -exec chmod 755 {} \;
              find $tmp/frame -type f -exec chmod 644 {} \;

              mksquashfs $tmp/frame $squash_output -keep-as-directory -noappend && rm -rv $tmp
            fi
          fi
        fi  
      fi
    else
      echo "|-- ERROR: There is no *skill_labels_by_frame.txt file inside $subdir"
    fi
  else
    echo "|-- ERROR: There is no *mp4 file inside $subdir"
  fi
done

if [[ $extract != "true" && $type == "sound" ]]; then
  find $OUTPUT_PATH -type d -exec chmod 755 {} \;
  find $OUTPUT_PATH -type f -exec chmod 644 {} \; 

  mksquashfs $OUTPUT_PATH $squash_output_sound -keep-as-directory -noappend  && rm $OUTPUT_PATH/*wav
fi  