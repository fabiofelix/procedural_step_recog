#!/bin/bash

IMG_PATH=$1/*sqf
AUDIO_PATH=$2/*sqf
CONFIG_PATH=$3
DESC=${4:+-"$4"}
ADD_OVER=""

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

#echo $ADD_OVER
#exit

for KFOLD_ITER in {1..10}; do
#break

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

if [[ ! -x sing  ]]; then
chmod u+x sing
fi

./sing $ADD_OVER << EOF

python tools/run_step_recog.py --cfg $CONFIG_PATH -i $KFOLD_ITER

EOF
EOSBATCH

done
