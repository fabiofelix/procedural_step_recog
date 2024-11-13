#!/bin/bash

POSSIBLE_SKILLS=("A8" "M1" "M2" "M3" "M4" "M5" "R16" "R18" "R19")
MAIN_PATH="/scratch/user/data"

SKILL=$1
CONFIG_PATH=$2
DESC=${3:+-"$3"}
IMG_PATH="$MAIN_PATH$SKILL/frame/*sqf"
AUDIO_PATH="$MAIN_PATH$SKILL/sound/*sqf"
ADD_OVER=""
ENV_PATH="/scratch/user/environment/cuda11.8"
CROSS_VALIDATION="true"
PROJECT_ACCOUNT="pr_ID"

if [[ -z $SKILL ||  ! ${POSSIBLE_SKILLS[*]} =~ $SKILL ]]; then
  echo "Skill [$SKILL] not defined."
  echo "Try one of these options [${POSSIBLE_SKILLS[*]}]"
  exit 
fi

if [[ ! -z $1 ]]; then
  for img in $IMG_PATH; do
    ADD_OVER="$ADD_OVER --overlay $img:ro"
  done
fi
if [[ ! -z $2 ]]; then
  for sound in $AUDIO_PATH; do
    ADD_OVER="$ADD_OVER --overlay $sound:ro"
  done
fi

if [[ $CROSS_VALIDATION == "true" ]]; then
  echo "Cross-validation"

for KFOLD_ITER in {1..10}; do

sbatch <<EOSBATCH
#!/bin/bash
#SBATCH -c 12
#SBATCH --mem 64GB
#SBATCH --time 2-00:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name step-recog-k$KFOLD_ITER$DESC
#SBATCH --output logs/%J_step-k$KFOLD_ITER$DESC.out
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=$USER@nyu.edu
#SBATCH --account=$PROJECT_ACCOUNT

if [[ ! -x /$ENV_PATH/sing  ]]; then
chmod u+x /$ENV_PATH/sing
fi

/$ENV_PATH/sing $ADD_OVER << EOF

python tools/run_step_recog.py --cfg $CONFIG_PATH -i $KFOLD_ITER

EOF
EOSBATCH

done

else

echo "Simple training"

sbatch <<EOSBATCH
#!/bin/bash
#SBATCH -c 12
#SBATCH --mem 64GB
#SBATCH --time 2-00:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name step$DESC
#SBATCH --output logs/%J_step-recog$DESC.out
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=$USER@nyu.edu
#SBATCH --account=$PROJECT_ACCOUNT

if [[ ! -x /$ENV_PATH/sing  ]]; then
chmod u+x /$ENV_PATH/sing
fi

/$ENV_PATH/sing $ADD_OVER << EOF

python tools/run_step_recog.py --cfg $CONFIG_PATH

EOF
EOSBATCH

fi
