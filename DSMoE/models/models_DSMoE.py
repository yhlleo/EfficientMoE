
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed
from .modules import get_2d_sincos_pos_embed, Attention, modulate, TimestepEmbedder, LabelEmbedder, FinalLayer, Mlp

# https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        #hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class MLP(nn.Module):
    def __init__(
        self, 
        hidden_size,
        intermediate_size,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class DSNaiveMoE(nn.Module):
    def __init__(self, num_experts, hidden_size, moe_intermediate_size):
        super().__init__()
        self.num_experts = num_experts
        self.blocks = nn.ModuleList()
        for _ in range(self.num_experts):
            self.blocks.append(MLP(hidden_size, moe_intermediate_size))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """
        Args:
            hidden_states: (batch_size * sequence_length, hidden_dim)
            selected_experts: (batch_size * sequence_length, top_k)
            routing_weights: (batch_size * sequence_length, top_k)
        Returns:
            (batch_size * sequence_length, hidden_dim)
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = self.blocks[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states

class TopkRouter(nn.Module):
    def __init__(self, hidden_size, n_routed_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.hidden_size)))
        nn.init.kaiming_normal_(self.weight)
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return router_logits


class DSMoE(nn.Module):
    def __init__(
        self,
        num_experts,
        hidden_size, 
        moe_intermediate_size, 
        n_group,
        topk_group,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        num_experts_per_tok=4,
        use_shared_expert=True,
    ):
        super().__init__()

        self.experts =  DSNaiveMoE(num_experts, hidden_size, moe_intermediate_size)
        self.gate = TopkRouter(hidden_size, num_experts)
        self.use_shared_expert = use_shared_expert 
        if self.use_shared_expert:
            self.shared_experts = MLP(hidden_size, moe_intermediate_size)

        self.n_routed_experts = num_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.top_k = num_experts_per_tok


    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias

        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        groups_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, groups_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )

        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)

        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        if self.use_shared_expert:
            hidden_states = hidden_states + self.shared_experts(residuals)
        else:
            hidden_states = hidden_states + residuals
        return hidden_states

#--------------------#
#  DiT
#--------------------#

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, head_dim=None, mlp_ratio=4.0, 
                 use_swiglu=False, MoE_config=None,
                 use_moe=False, rope_type="none", rope_max_pos=256, 
                 use_sinks=False, sliding_window=0, enable_gqa=False,
                 norm_type="layernorm",
                 **block_kwargs):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "rmsnorm":
            self.norm1 = DeepseekV3RMSNorm(hidden_size)
            self.norm2 = DeepseekV3RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.attn = Attention(hidden_size, num_heads=num_heads, head_dim=head_dim, qkv_bias=True, 
            rope_type=rope_type, rope_max_pos=rope_max_pos, use_sinks=use_sinks, 
            sliding_window=sliding_window, enable_gqa=enable_gqa, **block_kwargs)
       
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.use_moe = use_moe
        
        if self.use_moe:
            self.mlp = DSMoE(
                MoE_config.num_experts,
                hidden_size, 
                MoE_config.moe_intermediate_size, 
                MoE_config.n_group,
                MoE_config.topk_group,
                norm_topk_prob=True,
                routed_scaling_factor=MoE_config.routed_scaling_factor,
                num_experts_per_tok=MoE_config.num_experts_per_tok,
                use_shared_expert=MoE_config.use_shared_expert,
            )
        else:  
            self.mlp = Mlp(
                in_features=hidden_size, 
                hidden_features=mlp_hidden_dim, 
                act_layer=nn.SiLU, 
                drop=0
            )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x_mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * x_mlp
        return x


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
        rope_type="none", 
        use_sinks=False,
        sliding_window=0,
        enable_gqa=False,
        norm_type="layernorm",
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
        self.rope_type = rope_type
        self.rope_max_pos = int((input_size // patch_size) ** 2)
        if self.rope_type == "none":
            print("DiT PE: sin-cos embedding")
        else:
            print(f"DiT PE: RoPE ({self.rope_type})")
        self.use_sinks = use_sinks
        print(f"Using Attention Sinks: {self.use_sinks}")
        
        self.sliding_window = sliding_window
        if self.use_sinks:
            print(f"Sliding window: {self.sliding_window}")
        self.enable_gqa = enable_gqa
        if self.enable_gqa:
            print(f"Using GQA: {self.enable_gqa}")
        self.norm_type = norm_type
        print(f"Norm type: {self.norm_type}")

        self.MoE_config = MoE_config
        use_moe_flag = [True] * depth
        if self.MoE_config.interleave:
            use_moe_flag = [i%2==1 for i in range(depth)]
        if isinstance(self.MoE_config.skip_first2, bool):
            if self.MoE_config.skip_first2:
                use_moe_flag[0] = False
                use_moe_flag[1] = False
        elif isinstance(self.MoE_config.skip_first2, int):
            for i in range(self.MoE_config.skip_first2):
                use_moe_flag[i] = False
        print(use_moe_flag)
    
        self.CapacityPred_loss_weight = CapacityPred_loss_weight

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        if self.rope_type == "none":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads // (1 if use_moe_flag[_] else 1), head_dim=head_dim, mlp_ratio=mlp_ratio, 
                     use_swiglu=use_swiglu, MoE_config=MoE_config, use_moe=use_moe_flag[_], 
                     rope_type=self.rope_type, rope_max_pos=self.rope_max_pos, 
                     use_sinks=self.use_sinks, sliding_window=self.sliding_window,
                     enable_gqa=self.enable_gqa, norm_type=self.norm_type) for _ in range(depth)
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

        if self.rope_type == "none":
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

    def forward(self, x, t, y, train_step=0):
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

        if self.rope_type == "none":
            x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        else:
            x = self.x_embedder(x)
        t_embed = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y                                # (N, D)

        for layer_idx, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x, "Nothing", None, None, None, self.CapacityPred_loss_weight

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

