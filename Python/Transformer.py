import numpy as np
import torch
import torch.nn as nn

def get_len_mask(b: int, max_len: int, feat_lens: torch.Tensor, device: torch.device) -> torch.Tensor:
    """生成 Encoder 自注意力的长度掩码。
    Returns: shape (b, max_len, max_len)，True 表示该位置需要被屏蔽
    """
    attn_mask = torch.ones((b, max_len, max_len), device=device)
    for i in range(b):
        attn_mask[i, :, :feat_lens[i]] = 0
    return attn_mask.to(torch.bool)


def get_subsequent_mask(b: int, max_len: int, device: torch.device) -> torch.Tensor:
    """生成 Decoder 自注意力的因果掩码（上三角矩阵）。
    Returns: shape (b, max_len, max_len)，True 表示未来位置（需屏蔽）
    """
    return torch.triu(torch.ones((b, max_len, max_len), device=device), diagonal=1).to(torch.bool)


def get_enc_dec_mask(
    b: int,
    max_feat_len: int,
    feat_lens: torch.Tensor,
    max_label_len: int,
    device: torch.device,
) -> torch.Tensor:
    """生成 Encoder-Decoder Cross-Attention 掩码。
    Returns: shape (b, max_label_len, max_feat_len)，True 表示 Encoder padding 位置
    """
    attn_mask = torch.zeros((b, max_label_len, max_feat_len), device=device)
    for i in range(b):
        attn_mask[i, :, feat_lens[i]:] = 1
    return attn_mask.to(torch.bool)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, p=0.):
        """
        Args:
            d_k: 每个注意力头的 Key/Query 维度
            d_v: 每个注意力头的 Value 维度
            d_model: 输入/输出的总维度
            num_heads: 注意力头数
        """
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p)

        # 四个线性投影：Q、K、V 投影和输出投影
        self.W_Q = nn.Linear(d_model, d_k * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)
        self.W_out = nn.Linear(d_v * num_heads, d_model)

        # 权重初始化（参考 He initialization 思路）
        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.W_out.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

    def forward(self, Q, K, V, attn_mask):
        """
        Args:
            Q: (batch, q_len, d_model)
            K: (batch, k_len, d_model)
            V: (batch, v_len, d_model)  注意 k_len == v_len
            attn_mask: (batch, q_len, k_len) 或 None
        Returns:
            output: (batch, q_len, d_model)
        """
        N = Q.size(0)
        q_len, k_len = Q.size(1), K.size(1)
        d_k, d_v, num_heads = self.d_k, self.d_v, self.num_heads

        # Step 1: 线性投影 + 拆分多头
        # (N, q_len, d_model) -> (N, num_heads, q_len, d_k)
        Q = self.W_Q(Q).view(N, -1, num_heads, d_k).transpose(1, 2)
        K = self.W_K(K).view(N, -1, num_heads, d_k).transpose(1, 2)
        V = self.W_V(V).view(N, -1, num_heads, d_v).transpose(1, 2)

        # Step 2: 广播 Mask 到 head 维度
        if attn_mask is not None:
            assert attn_mask.size() == (N, q_len, k_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1).bool()

        # Step 3: Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e4)
        attns = torch.softmax(scores, dim=-1)
        attns = self.dropout(attns)

        # Step 4: 加权求和 + 合并多头 + 输出投影
        output = torch.matmul(attns, V)
        output = output.transpose(1, 2).contiguous().reshape(N, -1, d_v * num_heads)
        output = self.W_out(output)
        return output

def pos_sinusoid_embedding(seq_len: int, d_model: int) -> torch.Tensor:
    """生成正弦/余弦位置编码矩阵（不可学习，冻结权重）。
    Returns: (seq_len, d_model)
    """
    embeddings = torch.zeros((seq_len, d_model))
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos
        embeddings[:, i] = f(
            torch.arange(0, seq_len) / np.power(1e4, 2 * (i // 2) / d_model)
        )
    return embeddings.float()


class PoswiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, p: float = 0.):
        """d_ff 通常为 4 * d_model（如 d_model=512 时 d_ff=2048）"""
        super(PoswiseFFN, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)

    def forward(self, X):
        # X: (N, seq_len, d_model)
        out = self.fc1(X)          # -> (N, seq_len, d_ff)
        out = self.relu(out)
        out = self.fc2(out)        # -> (N, seq_len, d_model)
        return self.dropout(out)


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, n: int, dff: int, dropout_posffn: float, dropout_attn: float):
        assert dim % n == 0
        super(EncoderLayer, self).__init__()
        hdim = dim // n  # 每个注意力头的维度

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.multi_head_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)
        self.poswise_ffn = PoswiseFFN(dim, dff, p=dropout_posffn)

    def forward(self, enc_in, attn_mask):
        # Self-Attention：Q=K=V=enc_in（对应第 3.3 节双向注意力）
        residual = enc_in
        context = self.multi_head_attn(enc_in, enc_in, enc_in, attn_mask)
        out = self.norm1(residual + context)   # 残差 + LayerNorm

        # FFN
        residual = out
        out = self.poswise_ffn(out)
        out = self.norm2(residual + out)       # 残差 + LayerNorm
        return out


class Encoder(nn.Module):
    def __init__(self, dropout_emb, dropout_posffn, dropout_attn,
                 num_layers, enc_dim, num_heads, dff, tgt_len):
        super(Encoder, self).__init__()
        self.tgt_len = tgt_len
        # 固定的正弦位置编码（freeze=True 不参与梯度更新）
        self.pos_emb = nn.Embedding.from_pretrained(
            pos_sinusoid_embedding(tgt_len, enc_dim), freeze=True
        )
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.layers = nn.ModuleList(
            [EncoderLayer(enc_dim, num_heads, dff, dropout_posffn, dropout_attn)
             for _ in range(num_layers)]
        )

    def forward(self, X, X_lens, mask=None):
        # X: (batch, seq_len, d_model)
        seq_len = X.size(1)
        out = X + self.pos_emb(torch.arange(seq_len, device=X.device))  # 叠加位置编码
        out = self.emb_dropout(out)
        for layer in self.layers:
            out = layer(out, mask)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, n: int, dff: int, dropout_posffn: float, dropout_attn: float):
        assert dim % n == 0
        super(DecoderLayer, self).__init__()
        hdim = dim // n

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.poswise_ffn = PoswiseFFN(dim, dff, p=dropout_posffn)
        self.dec_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)     # Masked Self-Attn
        self.enc_dec_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn) # Cross-Attn

    def forward(self, dec_in, enc_out, dec_mask, dec_enc_mask):
        """
        dec_in:       (batch, dec_len, d_model)  Decoder 当前层输入
        enc_out:      (batch, enc_len, d_model)  Encoder 最终输出
        dec_mask:     Causal Mask (batch, dec_len, dec_len)
        dec_enc_mask: Cross-Attn Mask (batch, dec_len, enc_len)
        """
        # 1. Masked Self-Attention（对应第 4.2 节）
        residual = dec_in
        context = self.dec_attn(dec_in, dec_in, dec_in, dec_mask)
        dec_out = self.norm1(residual + context)

        # 2. Cross-Attention（对应第 4.3 节）：Q 来自 Decoder，K/V 来自 Encoder
        residual = dec_out
        context = self.enc_dec_attn(dec_out, enc_out, enc_out, dec_enc_mask)
        dec_out = self.norm2(residual + context)

        # 3. FFN
        residual = dec_out
        out = self.poswise_ffn(dec_out)
        dec_out = self.norm3(residual + out)
        return dec_out


class Decoder(nn.Module):
    def __init__(self, dropout_emb, dropout_posffn, dropout_attn,
                 num_layers, dec_dim, num_heads, dff, tgt_len, tgt_vocab_size):
        super(Decoder, self).__init__()
        # Word Embedding：将 token ID 映射为 d_model 维向量
        self.tgt_emb = nn.Embedding(tgt_vocab_size, dec_dim)
        self.dropout_emb = nn.Dropout(p=dropout_emb)
        # 固定正弦位置编码
        self.pos_emb = nn.Embedding.from_pretrained(
            pos_sinusoid_embedding(tgt_len, dec_dim), freeze=True
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(dec_dim, num_heads, dff, dropout_posffn, dropout_attn)
             for _ in range(num_layers)]
        )

    def forward(self, labels, enc_out, dec_mask, dec_enc_mask):
        # labels: (batch, dec_len) token ID 序列
        tgt_emb = self.tgt_emb(labels)
        pos_emb = self.pos_emb(torch.arange(labels.size(1), device=labels.device))
        dec_out = self.dropout_emb(tgt_emb + pos_emb)
        for layer in self.layers:
            dec_out = layer(dec_out, enc_out, dec_mask, dec_enc_mask)
        return dec_out

class Transformer(nn.Module):
    def __init__(self, frontend: nn.Module, encoder: Encoder,
                 decoder: Decoder, dec_out_dim: int, vocab: int):
        """
        frontend:    输入特征变换（如从 fbank_dim → d_model 的线性层）
        dec_out_dim: Decoder 输出维度（= d_model）
        vocab:       目标词表大小
        """
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(dec_out_dim, vocab)  # LM Head

    def forward(self, X: torch.Tensor, X_lens: torch.Tensor, labels: torch.Tensor):
        """
        X:      (batch, enc_len, fbank_dim)  输入特征序列
        X_lens: (batch,)                      每个样本的实际输入长度
        labels: (batch, dec_len)              目标 token ID 序列
        Returns: logits (batch, dec_len, vocab_size)
        """
        X_lens, labels = X_lens.long(), labels.long()
        b, device = X.size(0), X.device

        # Frontend + Encoder（对应第 3 节 Encoder 结构）
        out = self.frontend(X)
        max_feat_len = out.size(1)
        enc_mask = get_len_mask(b, max_feat_len, X_lens, device)
        enc_out = self.encoder(out, X_lens, enc_mask)

        # Decoder（对应第 4 节 Decoder 结构）
        max_label_len = labels.size(1)
        dec_mask = get_subsequent_mask(b, max_label_len, device)
        dec_enc_mask = get_enc_dec_mask(b, max_feat_len, X_lens, max_label_len, device)
        dec_out = self.decoder(labels, enc_out, dec_mask, dec_enc_mask)

        # LM Head
        return self.linear(dec_out)


if __name__ == "__main__":
    batch_size    = 16
    max_feat_len  = 100    # Encoder 输入序列最大长度
    max_label_len = 50     # Decoder 目标序列最大长度
    fbank_dim     = 80     # 输入特征维度（如 80 维 Fbank）
    hidden_dim    = 512    # d_model
    vocab_size    = 26     # 词表大小

    fbank_feature = torch.randn(batch_size, max_feat_len, fbank_dim)
    feat_lens     = torch.randint(1, max_feat_len, (batch_size,))
    labels        = torch.randint(0, vocab_size, (batch_size, max_label_len))

    feature_extractor = nn.Linear(fbank_dim, hidden_dim)
    encoder = Encoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=6, enc_dim=hidden_dim, num_heads=8, dff=2048, tgt_len=2048,
    )
    decoder = Decoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=6, dec_dim=hidden_dim, num_heads=8, dff=2048,
        tgt_len=2048, tgt_vocab_size=vocab_size,
    )
    model = Transformer(feature_extractor, encoder, decoder, hidden_dim, vocab_size)

    logits = model(fbank_feature, feat_lens, labels)
    print(f"logits shape: {logits.shape}")
    # 输出：logits shape: torch.Size([16, 50, 26])
