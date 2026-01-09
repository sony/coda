#!/bin/bash

PROJECT_DIR=$(pwd)
cd ${PROJECT_DIR}/experiment/

LIB=${PROJECT_DIR}
export PYTHONPATH=$LIB:$PYTHONPATH

# root folder containing data
export USER_DATA=/ssd/bacnguyen/project/disDM/data

# root folder for output
export USER_OUTPUT=/home/bacnguyen/output/temp


DATASET=${1}

OUTPUT=${USER_OUTPUT}/${DATASET}

PORT=16998

if [ $DATASET == "voc" ]; then
    NSLOTS=6
    TOTAL_STEPS=250001
    CONTRASTIVE_WEIGHT=0.05
elif [ $DATASET == "coco" ]; then
    NSLOTS=7
    TOTAL_STEPS=500001
    CONTRASTIVE_WEIGHT=0.03
elif [ $DATASET == "movi-e" ]; then
    NSLOTS=24
    TOTAL_STEPS=250001
    CONTRASTIVE_WEIGHT=0.05
elif [ $DATASET == "movi-c" ]; then
    NSLOTS=11
    TOTAL_STEPS=250001
    CONTRASTIVE_WEIGHT=0.05
fi

echo "Training on '${DATASET}' with ${NSLOTS} slots"

# Training the model
accelerate launch \
    --multi_gpu \
    --main_process_port ${PORT} \
    train.py \
    hydra.run.dir=${OUTPUT}  \
    dataset=${DATASET} \
    trainer.mixed_precision="fp16" \
    trainer.total_steps=${TOTAL_STEPS} \
    pipeline.encoder.slot_n_slots=${NSLOTS} \
    pipeline.contrastive_weight=${CONTRASTIVE_WEIGHT}