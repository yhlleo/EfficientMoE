
NODE_RANK=$1
MASTER_ADDR=set_master_addr

torchrun --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=$NODE_RANK \
  --master_port 29511 \
  --master_addr=$MASTER_ADDR \
  train_fsdp.py --config ./config/000_DSMoE_L_E48_Flow_half_misx0.4_s1a5_rope2d_l24.yaml \
  --wandb \
  --exp_name train_l_e48_dsmoe_half_misx0.4_s1a5_rope2d_l24-run2-fsdp \
  --resume_path exps/0215-000_DSMoE_L_E48_Flow/checkpoints/0200000.pt 
