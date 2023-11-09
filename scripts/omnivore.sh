#!/bin/bash

SOURCE_PATH=$1
OUTPUT_PATH=$2               #the same inside $CONFIG_PATH
CONFIG_PATH=$3
squash_files="true"
frame_path="/rgb"            #emtpy if you DO NOT want to process RAW frames in $SOURCE_PATH  
frame_aug_path="/rgb_aug"    #emtpy if you DO NOT want to process AUGMENTED frames in $SOURCE_PATH
squash_root="/omnivore"      #root folder of the squash files

function exec()
{
echo "Sending omnivore batch"

sbatch <<EOSBATCH
#!/bin/bash
#SBATCH -c 1
#SBATCH --mem 64GB
#SBATCH --time 24:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name omnivore-action
#SBATCH --output logs/%J_omnivore-action.out
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=$USER@nyu.edu

if [[ ! -x sing  ]]; then
chmod u+x sing
fi

./sing << EOF

python tools/run_act_recog.py --cfg $CONFIG_PATH

EOF
EOSBATCH

}

function exec_with_squash()
{
echo "|- Action recognition"

for video in "${SEARCH_FOR[@]}"; do

NAME="${video%.*}"
ADD_OVER=""

if [[ ! -z $frame_path ]]; then
  ADD_OVER="$ADD_OVER --overlay $SOURCE_PATH/$frame_path/$video:ro"
fi
if [[ ! -z $frame_aug_path ]]; then
  ADD_OVER="$ADD_OVER --overlay $SOURCE_PATH/$frame_aug_path/$video:ro"
fi

echo "|-- Create SquashFS for $NAME"

TMP=$OUTPUT_PATH/tmp_$NAME

sbatch <<EOSBATCH
#!/bin/bash
#SBATCH -c 12
#SBATCH --mem 64GB
#SBATCH --time 24:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name action-$NAME
#SBATCH --output logs/%J_$NAME.out
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=$USER@nyu.edu

if [[ ! -x sing  ]]; then
chmod u+x sing
fi

./sing $ADD_OVER << EOF

python tools/run_act_recog.py --cfg $CONFIG_PATH

EOF

echo "Create SquashFS for $NAME"

TMP=$OUTPUT_PATH/tmp_$NAME

mkdir -p $TMP/$squash_root
mv $OUTPUT_PATH/$NAME* $TMP/$squash_root

find $TMP/$squash_root -type d -exec chmod 755 {} \;
find $TMP/$squash_root -type f -exec chmod 644 {} \;

mksquashfs $TMP$squash_root $OUTPUT_PATH/$NAME.sqf -keep-as-directory -noappend && rm -rv $TMP

EOSBATCH

done
}

if $squash_files == "true"; then

##We have to pass only a couple of k files at once because of HPC quota or time restrictions
SEARCH_FOR=(
#  "M1-13.sqf"     #OK
#  "M1-14.sqf"     #OK 
)

  exec_with_squash
else
  exec  
fi
