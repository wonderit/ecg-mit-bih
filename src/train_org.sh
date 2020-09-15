#!/bin/bash
nohup  python train_org.py --filter_length 32 --kernel_size 16 --is_train True --epochs 80 > train_org.log &