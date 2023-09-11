class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # 多头自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 残差连接和层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 编码器deformable交叉注意力层
        self.cross_attn = MSDeformAttn(d_model, nhead, dropout=dropout)

        # 残差连接和层归一化
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 前馈全连接层
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 残差连接和层归一化
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 第一个多头自注意力层
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 编码器deformable交叉注意力层
        tgt2 = self.MSDeformAttn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 前馈全连接层
        tgt2 = self.linear2(self.dropout3(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    class EncoderLayer(nn.Module):
        def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
            super(EncoderLayer, self).__init__()

            # 多头自注意力层
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
            # 前馈神经网络的第一层
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            # Dropout层，用于防止过拟合
            self.dropout = nn.Dropout(dropout)
            # 前馈神经网络的第二层
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            # Layer normalization 层，用于归一化输入
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, src, src_mask=None, src_key_padding_mask=None):

            # 多头自注意力计算，src2是注意力加权的输出
            src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                                key_padding_mask=src_key_padding_mask)
            # 残差连接和Dropout
            src = src + self.dropout(src2)
            # Layer normalization
            src = self.norm1(src)

            # 前馈神经网络的第一层
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
            # 残差连接和Dropout
            src = src + self.dropout(src2)
            # Layer normalization
            src = self.norm2(src)

            return src, attn_weights