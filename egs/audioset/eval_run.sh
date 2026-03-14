#!/bin/bash
set -e

dataset=audioset
imagenetpretrain=True
bal=bal
batch_size=64
n_workers=8

# Use the SAME python env you trained with
python_path=/WAVE/projects2/oignat_lab/Parth-Personal/ENV/dev/bin/python

# DTFAT repo paths
main_dir=/WAVE/projects/oignat_lab/Parth-Personal/AudioProj/DTFAT
working_dir=${main_dir}/egs/audioset
py_file=${main_dir}/src/eval.py
label_csv=${working_dir}/data/class_labels_indices.csv

# Your eval manifest (clean)
te_data=/WAVE/projects/oignat_lab/Parth-Personal/Dataset/eval_converted_clean.json

# Your trained checkpoint
as_ckpt=/WAVE/projects/oignat_lab/Parth-Personal/AudioProj/DTFAT/experiments/balanced_50ep/models/best_audio_model.pth

# Not used by eval.py but required by argparser
tr_data=not_needed

CUDA_CACHE_DISABLE=1 ${python_path} -u ${py_file} \
  --dataset ${dataset} \
  --data-train ${tr_data} \
  --data-val ${te_data} \
  --working_dir ${working_dir} \
  --label-csv ${label_csv} \
  --batch-size ${batch_size} \
  --num-workers ${n_workers} \
  --bal ${bal} \
  --imagenet_pretrain ${imagenetpretrain} \
  --as2m_ckpt ${as_ckpt}
