#!/bin/bash

PROJECT_DIR=$(pwd)
cd ${PROJECT_DIR}/experiment/

LIB=${PROJECT_DIR}
export PYTHONPATH=$LIB:$PYTHONPATH

# root folder containing data
export USER_DATA=/ssd/bacnguyen/project/disDM/data

# root folder for output
export USER_OUTPUT=/home/bacnguyen/output/temp

# export HYDRA_FULL_ERROR=1


DATASET=${1}

OUTPUT=${USER_OUTPUT}/${DATASET}
PORT=16998

echo "Evaluating on '${DATASET}' with ${NSLOTS} slots"

# Evaluate the model
accelerate launch \
    --multi_gpu \
    --main_process_port ${PORT} \
    eval.py \
    hydra.run.dir=${OUTPUT}/eval/generation  \
    dataset=${DATASET} \
    model_path=${OUTPUT}/model


if [[ $DATASET == "movi-e" || $DATASET == "movi-c" ]]; then
    python linear_prob.py \
        hydra.run.dir=${OUTPUT}/eval/linear_prob  \
        dataset=prob_${DATASET} \
        model_path=${OUTPUT}/model
fi