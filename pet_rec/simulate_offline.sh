#!/bin/bash

python3 /root/SGS_IPU_SDK_25070213/Scripts/calibrator/simulator.py   \
        -i ./test_imgs/		\
        -m ./model/pet_mobilenetv2_attr.img	\
        -c Unknown				\
        -t Offline				\
        -n ./preprocess_attr.py \
        --num_process 8 \
        --soc_version 388g