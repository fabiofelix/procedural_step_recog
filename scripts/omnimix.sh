#!/bin/bash

IMG_PATH=$1/*sqf
AUDIO_PATH=$2/*sqf
CONFIG_PATH=$3
DESC=${4:+-"$4"}
ADD_OVER=""
ENV_PATH="/scratch/user/environment/cuda11.8"
CROSS_VALIDATION="true"

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
#SBATCH --output logs/%J_step-recog-k$KFOLD_ITER$DESC.out
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=$USER@nyu.edu

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
#SBATCH --job-name step-recog-$DESC
#SBATCH --output logs/%J_step-recog-$DESC.out
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=$USER@nyu.edu

if [[ ! -x /$ENV_PATH/sing  ]]; then
chmod u+x /$ENV_PATH/sing
fi

/$ENV_PATH/sing $ADD_OVER << EOF

python tools/run_step_recog.py --cfg $CONFIG_PATH

EOF
EOSBATCH

fi
