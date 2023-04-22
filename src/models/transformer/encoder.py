"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from src.models.transformer.blocks.encoder_layer import EncoderLayer
from src.models.transformer.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask):
        x = self.emb(x) # vec-> embedding+pos

        for layer in self.layers:
            x = layer(x, s_mask)

        return x

class EEND_encoder(nn.Module):
    def __init__(self,d_model, ffn_hidden, n_head, drop_prob,n_layers):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
    
    def forward(self,x,s_mask=None):
        for layer in self.layers:
            x = layer(x, s_mask)
        return x

if __name__ == '__main__':
    import torch
    model = EEND_encoder(2048,2048,8,0.5,1).cuda()
    data = torch.randn((60,431,2048)).cuda()
    optim = torch.optim.Adam(model.parameters(),lr=0.01)
    for i in range(100):
        out = model(data)
        loss = 1-out.max()
        optim.zero_grad()
        loss.backward()
    
    