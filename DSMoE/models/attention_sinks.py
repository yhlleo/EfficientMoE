import torch
import torch.nn as nn

def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    #assert K.shape == (n_tokens, n_heads, d_head)
    #assert V.shape == (n_tokens, n_heads, d_head)
    #K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    #V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    #print(Q.shape, K.shape, V.shape, S.shape)
    #S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1).to(Q.device)
    S = S.reshape(n_heads, 1, 1, 1).expand(-1, q_mult, n_tokens, -1).to(Q.device)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn #attn.reshape(n_tokens, -1)
