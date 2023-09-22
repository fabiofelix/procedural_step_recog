#!/bin/bash

path=$1
output_path=$2
echo $output_path
mkdir -p $output_path

for subdir in $path/*;
do
  file=$subdir/*.mp4
  output=$output_path/$(basename $subdir)

  if ls $file 1> /dev/null 2>&1;  then
	  echo "Processing " $file

    mkdir $output
    ffmpeg -i $file -qscale:v 2 "$output/frame_%010d.jpg" 

    echo "Create SquashFS for $subdir"
    find $output -type d -exec chmod 755 {} \;
    find $output -type f -exec chmod 644 {} \;

    mksquashfs $output $output.sqf -keep-as-directory
    rm -rv $output
  else
    echo "There is no mp4 file inside $subdir"
  fi

done
