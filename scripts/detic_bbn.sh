#!/bin/bash

SOURCE_PATH=$1
OUT_ROOT=$2
SKILL=$3

for f in $SOURCE_PATH/*; do
#for f in $SOURCE_PATH/*aug*; do
#for f in $SOURCE_PATH/M5-2*; do
NAME=$(basename $f)
echo $NAME


sbatch <<EOSBATCH
#!/bin/bash
#SBATCH -c 1
#SBATCH --mem 8GB
#SBATCH --time 4:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name detic-$NAME
#SBATCH --output logs/%J_$NAME.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@nyu.edu

./sing << EOF

python tools/detic_bbn.py run_one "$f" "$OUT_ROOT" --skill $SKILL

EOF

EOSBATCH
#break
done
