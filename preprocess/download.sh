#!/bin/bash

PROJECT_DIR=$(pwd)

mkdir -p $USER_DATA


for DATASET in "${@:2}"; do

    cd $USER_DATA

    echo "Downloading ${DATASET}"
    echo "Current directory is $(pwd)"

    # movi datasets
    if [[ ${DATASET:0:4} == "movi" ]]; then

        LEVEL=${DATASET:5:6}

        mkdir -p ${DATASET}
        cd ${DATASET}
        
        python3 ${PROJECT_DIR}/preprocess/download_movi.py \
            --out_path="./" \
            --level=${LEVEL} \
            --image_size=256

    fi

    if [[ ${DATASET} == "coco" ]]; then

        mkdir -p ${DATASET}
        cd ${DATASET}
        aria2c -x 16 -s 16 http://images.cocodataset.org/zips/train2017.zip
        aria2c -x 16 -s 16 http://images.cocodataset.org/zips/val2017.zip
        aria2c -x 16 -s 16 http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        unzip train2017.zip -d images
        unzip val2017.zip -d images
        unzip annotations_trainval2017.zip
    fi

    if [[ ${DATASET} == "voc" ]]; then

        mkdir -p ${DATASET}
        cd ${DATASET}

        pip install gdown

        FILE_ID=1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH
        python -m gdown https://drive.google.com/uc?id=${FILE_ID}

        tar -xvzf PASCAL_VOC.tgz

        # instance segmentation masks for evaluation
        aria2c -x 16 -s 16 http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
        tar -xvf VOCtrainval_11-May-2012.tar

        mv VOCdevkit/VOC2012/SegmentationObject/ .
        mv VOCSegmentation/images/ .
        mv VOCSegmentation/SegmentationClass/ .
        mv VOCSegmentation/SegmentationClassAug/ .
        mv VOCSegmentation/sets/ .

        rm -rf VOCdevkit  VOCSegmentation

    fi

done