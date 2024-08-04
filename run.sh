#!/bin/bash

DATASETS  = ("mutag" "benz" "car1" "car2")
BACKBONES = ("pge" "gnne" "kfact")
DEVICE    = $1

echo "device: $DEVICE"

echo ""
echo "==========================================================================="
echo "                 Starting running experiment"
echo "==========================================================================="
echo ""

for DATA in "${DATASETS[@]}"
do
    for BACKBONE in "${BACKBONES[@]}"
    do
        echo "=================================================="
        echo "         Explainer: $BACKBONE Dataset: $DATA      "
        echo "=================================================="
        python main.py --mode "ed" --data "$DATA" --backbone "$BACKBONE" --device "$DEVICE" --read_configs --save_params
    done
done