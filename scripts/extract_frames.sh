#!/bin/bash

path=$1
output_path=$2
echo $output_path
mkdir -p $output_path

for subdir in $path/*;
do
	echo $subdir/*.mp4 
	mkdir $output_path/$(basename $subdir)
	ffmpeg -i $subdir/*mp4 -qscale:v 2 "$output_path/$(basename $subdir)/frame_%010d.jpg" 
done
