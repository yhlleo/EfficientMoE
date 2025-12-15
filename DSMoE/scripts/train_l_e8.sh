CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  train.py --config ./config/000_DiffMoE_L_E8_Flow.yaml