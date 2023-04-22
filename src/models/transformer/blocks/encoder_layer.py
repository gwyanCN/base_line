"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from src.models.transformer.layers.layer_norm import LayerNorm
from src.models.transformer.layers.multi_head_attention import MultiHeadAttention
from src.models.transformer.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, s_mask=None):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=s_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
 
if __name__ == '__main__':
    import torch
    model = EncoderLayer(256,1024,4,drop_prob=0.5)
    Input_ = torch.randn((60,431,256))
    out = model(Input_)
    print(out.shape)