#!/bin/bash

IN_DIR=${1:-/vast/irr2020/BBN/rgb_frames}
OUT_DIR=${2:-/vast/$USER/BBN/aug_rgb_frames}

for f in $IN_DIR/*/; do

NAME=$(basename $f)
echo $NAME

sbatch <<EOSBATCH
#!/bin/bash
#SBATCH -c 1
#SBATCH --mem 4GB
#SBATCH --time 4:00:00
#SBATCH --job-name augmentation-$NAME
#SBATCH --output logs/%J_$NAME.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$USER@nyu.edu

./sing << EOF

python tools/augment.py "$f" "$OUT_DIR/${NAME}_aug{}" -n 20

EOF

EOSBATCH
break
done
