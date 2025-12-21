
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_port 29511 \
  train.py --config ./config/000_DSMoE_S_E16_Flow_half_misx2_s1a2_rope2d_l10_kda.yaml \
  --wandb \
  --exp_name train_s_e16_dsmoe_half_misx2_s1a2_rope2d_l10_kda
