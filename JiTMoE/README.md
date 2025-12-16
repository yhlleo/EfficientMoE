### Training of JiTMoE

After configuring the environment, we can start training according to the privided scripts:

| Script                | Training Model |
|----------------------------|-------------------------|
|[`train_jit_b16_e16.sh`](./train_jit_b16_e16.sh)|JiTMoE-B/16-E16|
|[`train_jit_l16_e16.sh`](./train_jit_l16_e16.sh)|JiTMoE-L/16-E16|

- Training `JiT-B-16-E16` examples:

```
sh ./train_jit_b16_e16.sh
```

- Training `JiT-L-16-E16` with FSDP for larger models:

Please set the `master_addr` first, then start the script on master and slaver servers respectively:

```
# on Master server
sh ./train_jit_l16_e16.sh 0

# on Slave server
sh ./train_jit_l16_e16.sh 1
```
