#!/bin/bash

IN_DIR=${1:-/vast/irr2020/BBN/rgb_frames}
OUT_DIR=${2:-/vast/$USER/BBN/aug_rgb_frames}
squash_files="true"

if $squash_files == "true"; then
 SEARCH_FOR=$IN_DIR/*sqf
 sed -e 's/$NV $@ --overlay "$OVERLAY:ro" "$SIF"/$NV --overlay "$OVERLAY:ro" $@ "$SIF"/' -i sing
else
 SEARCH_FOR=$IN_DIR/*/
fi

for f in $SEARCH_FOR; do

NAME=$(basename $f)
NAME="${NAME%.*}"
echo $NAME

SOURCE=$f
TARGET="$OUT_DIR/${NAME}_aug{}"
ADD_OVER=""

if $squash_files == "true"; then
 SOURCE="/$NAME"
 TARGET="$OUT_DIR/$NAME/${NAME}_aug{}"
 ADD_OVER="--overlay $f:ro"

 mkdir "$OUT_DIR/${NAME}"
fi

sbatch <<EOSBATCH
#!/bin/bash
#SBATCH -c 1
#SBATCH --mem 4GB
#SBATCH --time 4:00:00
#SBATCH --job-name aug-$NAME
#SBATCH --output logs/%J_$NAME.out
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=$USER@nyu.edu

./sing $ADD_OVER << EOF

python tools/augment.py $SOURCE $TARGET -n 20

EOF

if $squash_files == "true"; then
  echo "Create SquashFS for $NAME"
  find $OUT_DIR/${NAME} -type d -exec chmod 755 {} \;
  find $OUT_DIR/${NAME} -type f -exec chmod 644 {} \;

  mksquashfs $OUT_DIR/${NAME} $OUT_DIR/${NAME}.sqf -keep-as-directory
  rm -rv $OUT_DIR/${NAME}
fi

EOSBATCH
#break
done

sed -e 's/$NV --overlay "$OVERLAY:ro" $@ "$SIF"/$NV $@ --overlay "$OVERLAY:ro" "$SIF"/' -i sing
