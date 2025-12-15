
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_port 29511 \
  train.py --config ./config/000_DSMoE_B_E48_Flow_half_misx0.3_s1a5_rope2d_l12.yaml \
  --wandb \
  --exp_name train_b_e48_dsmoe_half_misx0.3_s1a5_rope2d_l12
