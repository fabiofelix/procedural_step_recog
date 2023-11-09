#!/bin/bash

SOURCE_PATH=$1
OUT_ROOT=$2
SKILL=$3
squash_files="true"
frame_path="/rgb"            #emtpy if you DO NOT want to process RAW frames in $SOURCE_PATH  
frame_aug_path="/rgb_aug" #emtpy if you DO NOT want to process AUGMENTED frames in $SOURCE_PATH
frame_squash_root="/frame"   #searches squash files inside it
squash_root="/obj_frame"     #root folder of the squash files

function exec()
{
for f in "${SEARCH_FOR[@]}"; do

NAME=$(basename $f)
NAME="${NAME%.*}"
echo $NAME

sbatch <<EOSBATCH
#!/bin/bash
#SBATCH -c 1
#SBATCH --mem 8GB
#SBATCH --time 24:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name detic-$NAME
#SBATCH --output logs/%J_$NAME.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@nyu.edu

./sing << EOF

python tools/detic_bbn.py run_one "$f" "$OUT_ROOT" --skill $SKILL

EOF
EOSBATCH
done
}

function exec_with_squash()
{
for f in "${SEARCH_FOR[@]}"; do

if [[ ! -f "$f" ]]; then
  echo "There is no RAW file [$f]"
  continue
fi

NAME=$(basename $f)
NAME="${NAME%.*}"
echo $NAME

ADD_OVER="--overlay $f:ro"
JOBLIST=""

if [[ ! -z $frame_path && ! -z $frame_aug_path ]]; then
  aug_file="$SOURCE_PATH/$frame_aug_path/$NAME.sqf"

  if [[ ! -f "$aug_file" ]]; then
    echo "There is no AUGMENTATION file [$aug_file]"
    continue
  fi

  ADD_OVER="$ADD_OVER --overlay $aug_file:ro"
fi

##Possibile folders to iterate
DIRLIST=() 

if [[ ! -z $frame_path ]]; then
  DIRLIST=($frame_squash_root/$NAME)
fi
if [[ ! -z $frame_aug_path ]]; then
  for i in {0..19}; do
    DIRLIST+=("$frame_squash_root/$NAME""_aug$i")
  done
fi

TMP_FOLDER="$OUT_ROOT/tmp_$NAME/$squash_root"

##Iterating every folder (raw and/or augmented) and sending a job for each one
for sub_dir in "${DIRLIST[@]}"; do

JOBNAME=$(basename $sub_dir)
JOBID=$(sbatch <<EOSBATCH
#!/bin/bash
#SBATCH -c 1
#SBATCH --mem 8GB
#SBATCH --time 24:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name detic-$JOBNAME
#SBATCH --output logs/%J_detic-$JOBNAME.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@nyu.edu

if [[ ! -x sing  ]]; then
chmod u+x sing
fi

./sing $ADD_OVER << EOF

python tools/detic_bbn.py run_one $sub_dir $TMP_FOLDER --skill $SKILL

EOF
EOSBATCH
)

##A list of jobs to use as dependency for creating squashFS
JOBID=(${JOBID// / })  #split JOBID='Submitted batch job 38616519' in a list. ${variable//<CHAR ORIGIN>/<CHAR TARGET>} outputs an array splitted by <CHAR TARGET>  
JOBID=${JOBID[-1]}

if [[ $JOBLIST == "" ]] ; then
 JOBLIST="$JOBID"
else
 JOBLIST="$JOBLIST,$JOBID"
fi

done

echo "Create SquashFS for $NAME $JOBLIST"

sbatch --dependency=afterok:$JOBLIST <<EOSBATCH
#!/bin/bash
#SBATCH -c 1
#SBATCH --mem 8GB
#SBATCH --time 24:00:00
#SBATCH --job-name squash-$NAME
#SBATCH --output logs/%J_squash-$NAME.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@nyu.edu

find $TMP_FOLDER -type d -exec chmod 755 {} \;
find $TMP_FOLDER -type f -exec chmod 644 {} \;

mksquashfs $TMP_FOLDER $OUT_ROOT/$NAME.sqf -keep-as-directory -noappend && rm -rfv $OUT_ROOT/tmp_$NAME/
EOSBATCH

done
}

if $squash_files == "true"; then
  SEARCH_FOR=$SOURCE_PATH/$frame_path/*sqf

  if [[ -z $frame_path ]]; then
    SEARCH_FOR=$SOURCE_PATH/$frame_aug_path/*sqf
  fi  


#---------------------------------------------#
  aux_path=$SOURCE_PATH/$frame_path

  if [[ -z $frame_path ]]; then
    aux_path=$SOURCE_PATH/$frame_aug_path
  fi 

##We have to pass only a couple of k files at once because of HPC quota or time restrictions
SEARCH_FOR=(
#  "$aux_path/M1-13.sqf"   #OK      
#  "$aux_path/M1-14.sqf"   #OK      
)

  exec_with_squash

else
  SEARCH_FOR=$SOURCE_PATH/$frame_path/*

  if [[ -z $frame_path ]]; then
    SEARCH_FOR=$SOURCE_PATH/$frame_aug_path/*
  fi  

  exec
fi
