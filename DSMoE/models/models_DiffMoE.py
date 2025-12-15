import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed
import torch.nn.functional as F
import torch.distributed as dist
from .modules import get_2d_sincos_pos_embed, Attention, modulate, TimestepEmbedder, LabelEmbedder, FinalLayer, MoeMLP, Mlp


#################################################################################
#                                DiffMoE Layer                                  #
#################################################################################


class SparseMoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, experts, hidden_dim, num_experts, n_shared_experts=0, capacity=2, mlp_ratio=4.0):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.empty((num_experts, hidden_dim)))
        nn.init.normal_(self.gate_weight, std=0.006)
        self.experts = nn.ModuleList(experts)
        self.capacity = capacity
        self.num_experts = num_experts

        self.n_shared_experts = n_shared_experts

        self.capacity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_experts, bias=True),
        )

        if self.n_shared_experts > 0:
            print("self.n_shared_experts", self.n_shared_experts)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio * 2)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.shared_experts = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        ema_decay = 0.95
        expert_threshold = torch.tensor([0.0]*num_experts)
        self.register_buffer('expert_threshold', expert_threshold)
        ema_decay = torch.tensor([ema_decay])
        self.register_buffer('ema_decay', ema_decay)



    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_eval(x)


    def update_threshold(self, capacity_pred):
        if self.training:
            capacity_pred = F.sigmoid(capacity_pred)
            S = capacity_pred.size(0)
            topk = int((S / self.num_experts) * self.capacity)
            threshold = self.expert_threshold
            ema_decay = self.ema_decay

            for i in range(self.num_experts):
                capacity_pred_scores, _ = torch.topk(capacity_pred[:, i], k=topk, dim=-1, sorted=True)
                quantile = capacity_pred_scores[-1].detach()
                threshold[i] = threshold[i] * ema_decay + (1-ema_decay) * quantile
        
            dist.all_reduce(threshold, op=dist.ReduceOp.SUM)
            threshold /= dist.get_world_size()
            self.expert_threshold = threshold


    def forward_train(self, x):
        B, s, D = x.shape
        identity = x

        # Flatten the input for processing
        x = x.view(-1, D)  # (S, D), where S = B * s
        S = x.shape[0]

        # Predict capacity
        capacity_pred = self.capacity_predictor(x.detach())  # (S, num_experts)
        k = int((S/self.num_experts) * self.capacity)

        # Compute gating logits and scores
        logits = F.linear(x, self.gate_weight, None)  # (S, num_experts)
        scores = logits.softmax(dim=-1).permute(1, 0)  # (num_experts, S)

        # Get top-k gating values and indices for each expert
        gating, index = torch.topk(scores, k=k, dim=-1, sorted=False)  # gating: (num_experts, k), index: (num_experts, k)

        # Prepare a mask for selected indices
        mask = torch.zeros((self.num_experts, S), dtype=x.dtype, device=x.device)
        mask.scatter_(1, index, 1.0)  # Set selected indices to 1

        # Expand gating weights for broadcasting
        gating_expanded = gating.unsqueeze(-1)  # (num_experts, k, 1)

        # # Gather inputs for each expert
        expert_inputs = x[index]  # (num_experts, k, D)
        expert_outputs = torch.stack([expert(expert_inputs[i]) for i, expert in enumerate(self.experts)])  # (num_experts, k, D)
        # Apply gating to expert outputs
        gated_outputs = gating_expanded * expert_outputs  # (num_experts, k, D)
        # Scatter the gated outputs back to the full output tensor
        y = torch.zeros((S * self.num_experts, D), dtype=x.dtype, device=x.device)
        offset = torch.arange(0, self.num_experts).unsqueeze(1).to(device=x.device) * S  # (num_experts, 1)
        index = (index + offset.long()).view(-1)  # Flatten the index to shape [num_experts * k]

        # Ensure gated_outputs is correctly reshaped
        gated_outputs_flat = gated_outputs.view(-1, D)  # Reshape to [num_experts * k, D]

        # Use torch.scatter with correct shapes
        y = torch.scatter(
            y,  # Target tensor of shape [S * num_experts, D]
            0,  # Dimension to scatter along
            index.unsqueeze(1).expand(-1, D),  # Indices of shape [num_experts * k, D]
            gated_outputs_flat  # Source values of shape [num_experts * k, D]
        )

        # Sum the outputs from all experts
        y = y.view(self.num_experts, S, D).sum(dim=0, keepdim=False)  # (S, 1, D)

        # Update capacity prediction
        self.update_threshold(capacity_pred)

        # Reshape the output to match the input shape
        x_out = y.view(B, s, D)

        # Prepare the ones tensor
        ones = mask.permute(1, 0).view(B, s, self.num_experts)

        # Reshape capacity prediction
        capacity_pred = capacity_pred.view(B, s, self.num_experts)

        # Add shared expert outputs if applicable
        if self.n_shared_experts > 0:
            x_out = x_out + self.shared_experts(identity)

        return x_out, ones, capacity_pred


    def forward_eval(self, x):
        B, s, D = x.shape
        identity = x

        x = x.view(-1, D)
        S = x.shape[0]

        capacity_pred = self.capacity_predictor(x.detach())
        capacity_pred = F.sigmoid(capacity_pred)
        threshold = self.expert_threshold


        logits = F.linear(x, self.gate_weight, None)    # bs*seq_len, num_experts
        scores = logits.softmax(dim=-1).permute(-1, -2)     #  num_experts, bs*seq_len

        y = torch.zeros_like(x, dtype=x.dtype)
        processed_tokens = 0

        for i, expert in enumerate(self.experts): 
            k_fixed = torch.where(capacity_pred[:,i]>threshold[i], 1, 0).sum()
            processed_tokens += k_fixed
            gating, index = torch.topk(scores[i], k=k_fixed, dim=-1, sorted=False) 
            y[index, :] += gating.unsqueeze(-1) * expert(x[index, :])

        real_capacity = processed_tokens / S / self.num_experts
        x_out = y.view(B, s, D)

        if self.n_shared_experts > 0:
            x_out = x_out + self.shared_experts(identity)
        return x_out, None, None


#################################################################################
#                                 Core DiffMoE Model                            #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, head_dim=None, mlp_ratio=4.0, 
                 use_swiglu=False, MoE_config=None,
                 use_moe=False,
                 **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, head_dim=head_dim, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.use_moe = use_moe
        if use_moe:
            if use_swiglu==False:
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = SparseMoEBlock(
                    experts=[Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0) for _ in range(MoE_config.num_experts)],
                    hidden_dim=hidden_size,
                    num_experts=MoE_config.num_experts,
                    capacity=MoE_config.capacity,
                    n_shared_experts=MoE_config.n_shared_experts,
                    mlp_ratio=4.0
                    )
            else:
                self.mlp = SparseMoEBlock(
                    experts=[MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim, ) for _ in range(MoE_config.num_experts)],
                    hidden_dim=hidden_size,
                    num_experts=MoE_config.num_experts,
                    capacity=MoE_config.capacity,
                    n_shared_experts=MoE_config.n_shared_experts,
                    )
        else:
            if use_swiglu==False:
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
            else:
                self.mlp = MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim, )

        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_moe:
            x_mlp, ones, pred_c = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            x = x + gate_mlp.unsqueeze(1) * x_mlp
            return x, ones, pred_c
        else:
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x, None, None


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,

        use_swiglu=False,
        MoE_config=None,
        init_MoeMLP=False,
        head_dim=None,
        use_capacity_schedule=None,
        CapacityPred_loss_weight = 0.01,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads


        self.MoE_config = MoE_config
        use_moe_flag = [True] * depth
        if self.MoE_config.interleave:
            use_moe_flag = [i%2==1 for i in range(depth)]
        print(use_moe_flag)
        self.CapacityPred_loss_weight = CapacityPred_loss_weight

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads // (1 if use_moe_flag[_] else 1), head_dim=head_dim, mlp_ratio=mlp_ratio, 
                     use_swiglu=use_swiglu, MoE_config=MoE_config, use_moe=use_moe_flag[_]) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.init_MoeMLP= MoE_config.init_MoeMLP
        self.initialize_weights()


        self.capacity_schedule = MoE_config.get("capacity_schedule", None)
        if self.capacity_schedule:
            self.training_iters = -1
        print(self.capacity_schedule)


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # new init 
        def init_MoeMLP(module, std=0.006):
            nn.init.normal_(module.gate_proj.weight, std=std)
            nn.init.normal_(module.up_proj.weight, std=std)
            nn.init.normal_(module.down_proj.weight, std=std)
        if self.init_MoeMLP:
            for block in self.blocks:
                for expert in block.mlp.experts:
                    init_MoeMLP(expert)
            print("init MoE related module with std 0.006 like DeepSeek-MoE")

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        if self.training and self.capacity_schedule:
            num_experts = self.MoE_config.num_experts
            capacity = self.MoE_config.capacity
            capacity_schedule_stage_I_iters = self.MoE_config.capacity_schedule.capacity_schedule_stage_I_iters
            capacity_schedule_stage_II_iters = self.MoE_config.capacity_schedule.capacity_schedule_stage_II_iters

            if self.training_iters <= capacity_schedule_stage_I_iters:
                capacity = num_experts
            elif self.training_iters <= capacity_schedule_stage_II_iters:
                capacity = capacity + (num_experts - capacity) * \
                            (capacity_schedule_stage_II_iters - self.training_iters) / (capacity_schedule_stage_II_iters - capacity_schedule_stage_I_iters)
            else:
                pass
            for i, block in enumerate(self.blocks):
                block.mlp.capacity = capacity

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        ones_list = []
        pred_c_list = []
        layer_idx_list = []
        for layer_idx, block in enumerate(self.blocks):
            x, ones, pred_c = block(x, c)                      # (N, T, D)
            if ones is not None:
                ones_list.append(ones)
                pred_c_list.append(pred_c)
                layer_idx_list.append(layer_idx)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x, "Capacity_Pred", layer_idx_list, ones_list, pred_c_list, self.CapacityPred_loss_weight

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        model_out = model_out[0]
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

