#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL=3
root_dir=/home/mmolinare/chiron
/home/mmolinare/miniconda3/envs/chiron/bin/python $root_dir/src/train.py \
  $root_dir/configs/training.yaml \
  $root_dir/tfrecord/combined/train.tfrecord \
  $root_dir/results \
  --val_file $root_dir/tfrecord/cheng-et-al/fold-1/val.tfrecord \
  $root_dir/tfrecord/combined/val.tfrecord
