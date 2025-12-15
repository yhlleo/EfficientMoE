CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  train.py --config ./config/000_DiffMoE_S_E16_Flow.yaml