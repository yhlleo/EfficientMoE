NODE_RANK=$1
OUTPUT_DIR="./exps/JiT-L-16-E16"
IMAGENET_PATH=/path/to/imagenet
MASTER_ADDR=set_your_master_addr

torchrun --nproc_per_node=8 --nnodes=2 --node_rank=${NODE_RANK} \
--master_port 29511 \
--master_addr=$MASTER_ADDR \
main_jit.py \
--model JiT-L/16-E16 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 64 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 64 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval
