#!/bin/bash

SOURCE_PATH=$1
OUTPUT_PATH=$2
SKILL=$3
TYPE=$4
EXTRACT=$5
SLOW_FAST_AUDIO_DEFAULT_SAMPLE_RATE=24000
POSSIBLE_SKILLS=("M1" "M2" "M3" "M5" "R18")
POSSIBLE_TYPES=("frame" "sound")

if [[ ! -d $SOURCE_PATH ]]; then
  echo "Source path [$SOURCE_PATH] doesn't exists"
  exit 
fi
if [[ ! -d $OUTPUT_PATH ]]; then
  echo "Output path [$OUTPUT_PATH] doesn't exists"
  exit 
fi
if [[ ! ${POSSIBLE_SKILLS[*]} =~ $SKILL ]]; then
  echo "Skill [$SKILL] not defined."
  echo "Try one of these options [${POSSIBLE_SKILLS[*]}]"
  exit 
fi
if [[ ! ${POSSIBLE_TYPES[*]} =~ $TYPE ]]; then
  echo "Type [$TYPE] not defined."
  echo "Try one of these options [${POSSIBLE_TYPES[*]}]"
  exit 
fi

echo "|- Processing " $SOURCE_PATH
mkdir -p $OUTPUT_PATH

for subdir in $SOURCE_PATH/*;
do
  file=$subdir/*.mp4
  name=$(basename $subdir)
  output=$OUTPUT_PATH/$name

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

          if [[ $TYPE == "frame" ]]; then
            mkdir -p $output
          fi

          if [[ $TYPE == "frame" ]]; then
            ffmpeg -i $file -qscale:v 2 "$output/frame_%010d.jpg"
          else
            ffmpeg -i $file -vn -acodec pcm_s16le -ac 1 -ar $SLOW_FAST_AUDIO_DEFAULT_SAMPLE_RATE  $OUTPUT_PATH/$name.wav
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
