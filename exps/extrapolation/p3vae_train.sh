#!/bin/bash

# architecture
phy="--range-xi 0.0 1.0"
dec="--hidlayers-aux1-dec 64 64 --hidlayers-aux2-dec 128 128 --x-lnvar -9.0"
feat="--arch-feat mlp --hidlayers-feat 128 128 --num-units-feat 256"
enc="--hidlayers-aux1-enc 64 32 --hidlayers-aux2-enc 64 32 --hidlayers-xi 64 32"

# optimization
optim="--learning-rate 1e-3 --batch-size 200 --epochs 1000 --grad-clip 5.0 --intg-lev 1 --weight-decay 1e-6 --adam-eps 1e-3 --balance-kld 0.01"

# other options
others="--save-interval 500" # --cuda

# ------------------------------------------------

outdir="./out_pendulum/free_damped_pendulum/p3vae"

commands="--dim-z-aux1 1 --dim-z-aux2 -1"


for i in $(seq 1 5);
do
    i_outdir=${outdir}_$i
    mkdir ${i_outdir}
    options="--datadir ./../../data/pendulum/free_damped_pendulum --outdir ${i_outdir} ${phy} ${dec} ${feat} ${unmix} ${enc} ${optim} ${others}"
    python -m p3vae_train ${options} ${commands}
done
