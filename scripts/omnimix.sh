#!/bin/bash

ACTION_PATH=$1/*sqf
ACTION_TEST_PATH=$1/test_subset/*sqf
IMG_PATH=$2/*sqf
IMG_TEST_PATH=$2/test_subset/*sqf
AUDIO_PATH=$3/*sqf
CONFIG_PATH=$4
ADD_OVER=""

if [[ ! -z $1 ]]; then
  for action in $ACTION_PATH; do
    ADD_OVER="$ADD_OVER --overlay $action:ro"
  done
  for action in $ACTION_TEST_PATH; do
    ADD_OVER="$ADD_OVER --overlay $action:ro"
  done
fi
if [[ ! -z $2 ]]; then
  for img in $IMG_PATH; do
    ADD_OVER="$ADD_OVER --overlay $img:ro"
  done
  for img in $IMG_TEST_PATH; do
    ADD_OVER="$ADD_OVER --overlay $img:ro"
  done
#echo $ADD_OVER
fi
if [[ ! -z $3 ]]; then
  for sound in $AUDIO_PATH; do
    ADD_OVER="$ADD_OVER --overlay $sound:ro"
  done
fi

sbatch <<EOSBATCH
#!/bin/bash
#SBATCH -c 12
#SBATCH --mem 64GB
#SBATCH --time 4:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name step-recog
#SBATCH --output logs/%J_step-recog.out
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=$USER@nyu.edu

if [[ ! -x sing  ]]; then
chmod u+x sing
fi

./sing $ADD_OVER << EOF

python tools/run_step_recog.py --cfg $CONFIG_PATH

EOF

EOSBATCH
