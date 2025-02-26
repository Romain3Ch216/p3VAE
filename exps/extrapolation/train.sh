# Source: https://github.com/n-takeishi/phys-vae
#!/bin/bash

# architecture
phy="--range-xi 0.0 1.0"
dec="--hidlayers-aux1-dec 64 64 --hidlayers-aux2-dec 128 128 --x-lnvar -9.0"
feat="--arch-feat mlp --hidlayers-feat 128 128 --num-units-feat 256"
unmix="--hidlayers-unmixer 128 128"
enc="--hidlayers-aux1-enc 64 32 --hidlayers-aux2-enc 64 32 --hidlayers-xi 64 32"

# optimization
optim="--learning-rate 1e-3 --train-size 1000 --batch-size 200 --epochs 1000 --grad-clip 5.0 --intg-lev 1 --weight-decay 1e-6 --adam-eps 1e-3 --balance-kld 1.0"

# other options
others="--save-interval 999999 --num-workers 0 --activation elu" # --cuda

# ------------------------------------------------

outdir="./out_pendulum/free_damped_pendulum/phy_vae"

if [ "$1" = "physonly" ]; then
    # Physcs-only
    beta=1e-1; gamma=1e-3
    commands="--dim-z-aux1 -1 --dim-z-aux2 -1 --balance-unmix ${gamma} --balance-dataug ${beta}"
elif [ "$1" = "nnonly0" ]; then
    # NN-only, w/o ODE
    commands=" --dim-z-aux1 -1 --dim-z-aux2 4 --no-phy"
elif [ "$1" = "nnonly1" ]; then
    # NN-only, w/ ODE
    outdir="./out_pendulum/free_damped_pendulum/nn_only"
    commands="--dim-z-aux1 2 --dim-z-aux2 -1 --no-phy --dim-z-add 3"
elif [ "$1" = "physnn" ]; then
    # Phys+NN
    alpha=1e-2; beta=1e-1; gamma=1e-3  
    commands="--dim-z-aux1 1 --dim-z-aux2 -1 --dim-z-add 1 --balance-unmix ${gamma} --balance-dataug ${beta} --balance-lact-dec ${alpha} --balance-lact-enc ${alpha}"
elif [ "$1" = "physnn_noreg" ]; then
    # Phys+NN
    outdir="./out_pendulum/free_damped_pendulum/phy_vae_no_reg"
    alpha=0; beta=0; gamma=0  
    commands="--dim-z-aux1 1 --dim-z-aux2 -1 --dim-z-add 1 --balance-unmix ${gamma} --balance-dataug ${beta} --balance-lact-dec ${alpha} --balance-lact-enc ${alpha}"
else
    echo "unknown option"
    commands=""
fi

for i in $(seq 1 5);
do
    i_outdir=${outdir}_$i
    mkdir ${i_outdir}
    options="--datadir ./../../data/pendulum/free_damped_pendulum --outdir ${i_outdir} ${phy} ${dec} ${feat} ${unmix} ${enc} ${optim} ${others}"
    python -m train ${options} ${commands}
done


