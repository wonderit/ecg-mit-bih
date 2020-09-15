#!/bin/bash
nohup  python train_org.py --filter_length 8 --kernel_size 16 --is_train True --epochs 1 > train_org.log &