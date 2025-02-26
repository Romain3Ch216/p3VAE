#!/bin/bash

opt="--len-episode 50 --dt 0.05 --noise-std 5e-2 --outdir ./free_damped_pendulum"
# range="--range-init -1.57 1.57 --range-omega 0.785 3.14 --range-gamma 0.0 0.8 --range-f 3.14 6.28 --range-A 0.0 0.0"
range="--range-init -1.57 1.57 --range-omega 0.1 6.28 --range-gamma 0.0 0.5 --range-f 3.14 6.28 --range-A 0.0 0.0"

python generate.py ${opt} --name train --n-samples 1000 --seed 1234 ${range}
python generate.py ${opt} --name valid --n-samples 500 --seed 1235 ${range}

# generate longer episodes in order to test model at extrapolation
opt="--len-episode 400 --dt 0.05 --noise-std 5e-2 --outdir ./free_damped_pendulum"
range="--range-init -1.57 1.57 --range-omega 0.1 6.28 --range-gamma 0.0 2 --range-f 3.14 6.28 --range-A 0.0 0.0"

python generate.py ${opt} --name test --n-samples 1000 --seed 1236 ${range}
