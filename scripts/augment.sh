#!/bin/bash

IN_DIR=$1
OUT_DIR=$2
squash_files="true"
squash_root="/frame"

if $squash_files == "true"; then
 SEARCH_FOR=$IN_DIR/*sqf
else
 SEARCH_FOR=$IN_DIR/*/
fi

##We have to pass only a couple of k files at once because of HPC quota or time restrictions
SEARCH_FOR=(
  "/home/user/data/BBN/new/M1/frame/rgb/M1-13.sqf"    
  "/home/user/data/BBN/new/M1/frame/rgb/M1-14.sqf"   
)

for f in "${SEARCH_FOR[@]}"; do

NAME=$(basename $f)
NAME="${NAME%.*}"
echo $NAME

SOURCE=$f
TARGET="$OUT_DIR/${NAME}_aug{}"
ADD_OVER=""
TMP=""

if $squash_files == "true"; then
 SOURCE="$squash_root/$NAME"
 TMP="$OUT_DIR/tmp_$NAME"
 TARGET="$TMP/$squash_root/${NAME}_aug{}"
 ADD_OVER="--overlay $f:ro"

 mkdir -p "$TMP/$squash_root"
fi

sbatch <<EOSBATCH
#!/bin/bash
#SBATCH -c 1
#SBATCH --mem 4GB
#SBATCH --time 12:00:00
#SBATCH --job-name aug-$NAME
#SBATCH --output logs/%J_$NAME.out
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=$USER@nyu.edu

if [[ ! -x sing  ]]; then
chmod u+x sing
fi

./sing $ADD_OVER << EOF

python tools/augment.py $SOURCE $TARGET -n 20

EOF

if $squash_files == "true"; then
  echo "Create SquashFS for $NAME"
  find $TMP/$squash_root -type d -exec chmod 755 {} \;
  find $TMP/$squash_root -type f -exec chmod 644 {} \;

  mksquashfs $TMP/$squash_root $OUT_DIR/$NAME.sqf -keep-as-directory -noappend && rm -rv $TMP
fi

EOSBATCH
done
