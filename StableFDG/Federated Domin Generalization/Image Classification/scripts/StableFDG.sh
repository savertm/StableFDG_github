#!/bin/bash

cd ..

DATA=/drive1/data
DASSL=/home/savertm/Project/FedDG/Dassl.pytorch

################### leave one domain out setting
DATASET=pacs # office_home_dg, pacs, vlcs, digits_dg
TRAINER=Vanilla2

NET=resnet18_OMA_ms_l123 # resnet50_OMA_ms_l123
MIX=random

if [ ${DATASET} == pacs ]; then
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch
elif [ ${DATASET} == office_home_dg ]; then
    D1=art
    D2=clipart
    D3=product
    D4=real_world
elif [ ${DATASET} == vlcs ]; then
    D1=caltech
    D2=labelme
    D3=pascal
    D4=sun
elif [ ${DATASET} == digits_dg ]; then
    D1=mnist
    D2=mnist_m
    D3=svhn
    D4=syn
fi


GPU_number=0
for SEED in $(seq 2025 2025)
do
    for SETUP in $(seq 1 1)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi


        CUDA_VISIBLE_DEVICES=${GPU_number} python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --source-domains ${S1} ${S2} ${S3} \
        --target-domains ${T} \
        --dataset-config-file ${DASSL}/configs/datasets/dg/${DATASET}.yaml \
        --config-file configs/trainers/StableFDG/${DATASET}_${MIX}.yaml \
        --output-dir FedDG/${DATASET}/${TRAINER}/${NET}_StableFDG/${MIX}/${T}/seed${SEED} \
        --data_distribution Single \
        --exploration_level 3.0 \
        --oversampling_size 32 \
        MODEL.BACKBONE.NAME ${NET}

    done
done

