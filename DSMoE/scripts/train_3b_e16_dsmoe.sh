
NODE_RANK=$1
MASTER_ADDR=set_master_addr

torchrun --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=$NODE_RANK \
  --master_port 29511 \
  --master_addr= $MASTER_ADDR \
  train_fsdp.py --config ./config/000_DSMoE_3B_E16_Flow_half_misx2.5_s1a2_rope2d_l30.yaml \
  --wandb \
  --exp_name train_3b_e16_dsmoe_half_misx2.5_s1a2_rope2d_l30 
