#!/bin/bash
python ckd_modeling.py \
  --embedding-root ckd_embeddings_100 \
  --window-size 10 \
  --embed-dim 768 \
  --epochs 50 \
  --batch-size 64 \
  --lr 5e-3 \
  --patience 5 \
  --scheduler-patience 2 \
  --metadata-file patient_embedding_metadata.csv \
  --hidden-dim 128 \
  --num-layers 2 \
  --rnn-dropout 0.2 \
  --rnn-bidir \
  --transformer-nhead 4 \
  --transformer-dim-feedforward 256 \
  --transformer-dropout 0.2
