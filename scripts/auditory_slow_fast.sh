#!/bin/bash

CONFIG_PATH=$1

sbatch <<EOSBATCH
#!/bin/bash
#SBATCH -c 12
#SBATCH --mem 128GB
#SBATCH --time 5-0:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name auditory-slow-fast
#SBATCH --output logs/%J_auditory-slow-fast.out
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=$USER@nyu.edu

if [[ ! -x sing  ]]; then
chmod u+x sing
fi

./sing <<EOF

python auditory-slow-fast/tools/run_net.py --cfg $CONFIG_PATH

EOF

EOSBATCH
