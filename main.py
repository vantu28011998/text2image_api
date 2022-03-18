import uvicorn
from fastapi import FastAPI
# import string
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from torchvision.utils import make_grid
# import numpy as np
# import io
# import os
# from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# '''DEFINITION MODEL'''
# class SelfAttention(nn.Module):
#     def __init__(self, embed_size, heads):
#         super(SelfAttention, self).__init__()
#         self.embed_size = embed_size
#         self.heads = heads
#         self.head_dim = embed_size // heads

#         assert (
#             self.head_dim * heads == embed_size
#         ), "Embedding size needs to be divisible by heads"

#         self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

#     def forward(self, values, keys, query, mask):
#         # Get number of training examples
#         N = query.shape[0]

#         value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

#         # Split the embedding into self.heads different pieces
#         values = values.reshape(N, value_len, self.heads, self.head_dim)
#         keys = keys.reshape(N, key_len, self.heads, self.head_dim)
#         query = query.reshape(N, query_len, self.heads, self.head_dim)

#         values = self.values(values)  # (N, value_len, heads, head_dim)
#         keys = self.keys(keys)  # (N, key_len, heads, head_dim)
#         queries = self.queries(query)  # (N, query_len, heads, heads_dim)

#         # Einsum does matrix mult. for query*keys for each training example
#         # with every other training example, don't be confused by einsum
#         # it's just how I like doing matrix multiplication & bmm

#         energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
#         # queries shape: (N, query_len, heads, heads_dim),
#         # keys shape: (N, key_len, heads, heads_dim)
#         # energy: (N, heads, query_len, key_len)

#         # Mask padded indices so their weights become 0
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, float("-1e20"))

#         # Normalize energy values similarly to seq2seq + attention
#         # so that they sum to 1. Also divide by scaling factor for
#         # better stability
#         attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
#         # attention shape: (N, heads, query_len, key_len)

#         out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
#             N, query_len, self.heads * self.head_dim
#         )
#         # attention shape: (N, heads, query_len, key_len)
#         # values shape: (N, value_len, heads, heads_dim)
#         # out after matrix multiply: (N, query_len, heads, head_dim), then
#         # we reshape and flatten the last two dimensions.

#         out = self.fc_out(out)
#         # Linear layer doesn't modify the shape, final shape will be
#         # (N, query_len, embed_size)

#         return out
# class TransformerBlock(nn.Module):
#     def __init__(self, embed_size, heads, dropout, forward_expansion):
#         super(TransformerBlock, self).__init__()
#         self.attention = SelfAttention(embed_size, heads)
#         self.norm1 = nn.LayerNorm(embed_size)
#         self.norm2 = nn.LayerNorm(embed_size)

#         self.feed_forward = nn.Sequential(
#             nn.Linear(embed_size, forward_expansion * embed_size),
#             nn.ReLU(),
#             nn.Linear(forward_expansion * embed_size, embed_size),
#         )

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, value, key, query, mask):
#         attention = self.attention(value, key, query, mask)

#         # Add skip connection, run through normalization and finally dropout
#         x = self.dropout(self.norm1(attention + query))
#         forward = self.feed_forward(x)
#         out = self.dropout(self.norm2(forward + x))
#         return out
# class Encoder(nn.Module):
#     def __init__(
#         self,
#         src_vocab_size,
#         embed_size,
#         num_layers,
#         heads,
#         device,
#         forward_expansion,
#         dropout,
#         max_length,
#     ):

#         super(Encoder, self).__init__()
#         self.embed_size = embed_size
#         self.device = device
#         self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
#         self.position_embedding = nn.Embedding(max_length, embed_size)

#         self.layers = nn.ModuleList(
#             [
#                 TransformerBlock(
#                     embed_size,
#                     heads,
#                     dropout=dropout,
#                     forward_expansion=forward_expansion,
#                 )
#                 for _ in range(num_layers)
#             ]
#         )

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, mask):
#         N, seq_length = x.shape
#         positions = torch.arange(0, seq_length).expand(N, seq_length)
#         out = self.dropout(
#             (self.word_embedding(x) + self.position_embedding(positions))
#         )

#         # In the Encoder the query, key, value are all the same, it's in the
#         # decoder this will change. This might look a bit odd in this case.
#         for layer in self.layers:
#             out = layer(out, out, out, mask)

#         return out
# class DecoderBlock(nn.Module):
#     def __init__(self, embed_size, heads, forward_expansion, dropout, device):
#         super(DecoderBlock, self).__init__()
#         self.norm = nn.LayerNorm(embed_size)
#         self.attention = SelfAttention(embed_size, heads=heads)
#         self.transformer_block = TransformerBlock(
#             embed_size, heads, dropout, forward_expansion
#         )
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, value, key, src_mask, trg_mask):
#         attention = self.attention(x, x, x, trg_mask)
#         query = self.dropout(self.norm(attention + x))
#         out = self.transformer_block(value, key, query, src_mask)
#         return out


# class Decoder(nn.Module):
#     def __init__(
#         self,
#         trg_vocab_size,
#         embed_size,
#         num_layers,
#         heads,
#         forward_expansion,
#         dropout,
#         device,
#         max_length,
#     ):
#         super(Decoder, self).__init__()
#         self.device = device
#         self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
#         self.position_embedding = nn.Embedding(max_length, embed_size)

#         self.layers = nn.ModuleList(
#             [
#                 DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
#                 for _ in range(num_layers)
#             ]
#         )
#         self.fc_out = nn.Linear(embed_size, trg_vocab_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, enc_out, src_mask, trg_mask):
#         N, seq_length = x.shape
#         positions = torch.arange(0, seq_length).expand(N, seq_length)
#         x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
#         for layer in self.layers:
#             x = layer(x, enc_out, enc_out, src_mask, trg_mask)
#         out = self.fc_out(x)
#         return out
# class Transformer(nn.Module):
#     def __init__(
#         self,
#         src_vocab_size,
#         trg_vocab_size,
#         src_pad_idx,
#         trg_pad_idx,
#         device = device,
#         embed_size = 256,
#         num_layers = 6,
#         forward_expansion = 4,
#         heads = 8,
#         dropout = 0.1,
#         enc_max_length = 20,
#         dec_max_length = 66,
#     ):

#         super(Transformer, self).__init__()

#         self.encoder = Encoder(
#             src_vocab_size,
#             embed_size,
#             num_layers,
#             heads,
#             device,
#             forward_expansion,
#             dropout,
#             enc_max_length,
#         )

#         self.decoder = Decoder(
#             trg_vocab_size,
#             embed_size,
#             num_layers,
#             heads,
#             forward_expansion,
#             dropout,
#             device,
#             dec_max_length,
#         )

#         self.src_pad_idx = src_pad_idx
#         self.trg_pad_idx = trg_pad_idx
#         self.device = device

#     def make_src_mask(self, src):
#         src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
#         # (N, 1, 1, src_len)
#         return src_mask

#     def make_trg_mask(self, trg):
#         N, trg_len = trg.shape
#         trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
#             N, 1, trg_len, trg_len
#         )

#         return trg_mask

#     def forward(self, src, trg):
#         src_mask = self.make_src_mask(src)
#         trg_mask = self.make_trg_mask(trg)
#         enc_src = self.encoder(src, src_mask)
#         out = self.decoder(trg, enc_src, src_mask, trg_mask)
#         return out
# def loadTransformer(device) -> Transformer:
#     return torch.load("dlmodel/cpu_transformer_model.pth",map_location=device)
# def loadVocab():
#     vocab = {}
#     repo = []
#     f = io.open("dlmodel/vocab.txt", mode="r", encoding="utf-8")
#     count = 0
#     for line in f:
#         count += 1
#         repo.append(line.strip())
#     f.close()
#     for index,r in enumerate(repo):
#         vocab.update({r:index})
#     return vocab
# class VQDecoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, kernel_sizes=(1, 3, 2, 2), stride=2):
#         super(VQDecoder, self).__init__()
        
#         kernel_1, kernel_2, kernel_3, kernel_4 = kernel_sizes
        
#         self.residual_conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_1, padding=0)
#         self.residual_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, padding=1)
        
#         self.strided_t_conv_1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_3, stride, padding=0)
#         self.strided_t_conv_2 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_4, stride, padding=0)
        
#     def forward(self, x):
        
#         y = self.residual_conv_1(x)
#         y = y+x
#         x = F.relu(y)
        
#         y = self.residual_conv_2(x)
#         y = y+x
#         y = F.relu(y)
        
#         y = self.strided_t_conv_1(y)
#         y = self.strided_t_conv_2(y)

#         return y
# def loadDecoder(device):
#     return torch.load("dlmodel/gpu_decoder_model.pth",map_location=device)
# def loadCodebook(device):
#     return torch.load("dlmodel/codebook_storage.pth",map_location=device)
# class Text2Image:
#     def __init__(self):
#         self.vocabMap = loadVocab()
#         self.softmax = nn.Softmax(dim=2)
#         self.transformer_model = loadTransformer(device=device)
#         self.decoder = loadDecoder(device)
#         self.codebooks = loadCodebook(device).view(179200,4096)
#     def predict(self,text: string):
#         vocabs = text.split(" ")
#         token = []
#         name = 'img'
#         token.append(self.vocabMap['<start>'])
#         for vocab in vocabs:
#             if vocab in self.vocabMap:
#                 token.append(self.vocabMap[vocab])
#             elif vocab.capitalize() in self.vocabMap:
#                 token.append(self.vocabMap[vocab.capitalize()])
#             else:
#                 token.append(self.vocabMap['<unk>'])
#         strToken = ''
#         for t in token:
#             strToken += str(t)
#         name += strToken
#         if len(token)<20:
#             for i in range(0,20-len(token)):
#                 token.append(self.vocabMap['<pad>'])
#         token_tensor = torch.tensor(token).view(1,-1)
#         dec = torch.tensor([1]).view(1,-1)
#         for i in range(1,66):
#             dec = torch.cat((dec,torch.tensor([[0]])),dim=1)
#             out = self.transformer_model(token_tensor,dec)
#             out = self.softmax(out)
#             out = torch.argmax(out, dim=2)
#             dec[-1] = out[-1]
#         dec = dec[0,1:65]
#         dec = dec - 3
#         img_codebook = torch.tensor([])
#         for c in dec:
#             img_codebook = torch.cat((img_codebook,self.codebooks[c].view(1,64,64)),dim=0)
#         img = self.decoder(img_codebook.view(1,64,64,64))
#         plt.figure(figsize=(8,8))
#         plt.axis("off")
#         plt.imshow(np.transpose(make_grid(img.detach().cpu(), padding=0, normalize=True), (1, 2, 0)))
#         plt.savefig(f'image/{name}.png',transparent=True, bbox_inches="tight", pad_inches=0)
#         return name

'''API'''
origins = ["*"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# text2Image = Text2Image()
@app.get("/")
def root_api():
    return ""
@app.get("/ping")
def ping():
    return "pong"
# @app.get("/text/{text}")
# async def text2image_api(text):
#     name = text2Image.predict(text)
#     return {
#     "text":text,
#     "image":name
#     }
# @app.get("/image/{image}")
# async def get_image_api(image):
#     filePath = f'./image/{image}.png'
#     return FileResponse(filePath)
# @app.delete("/image/{image}")
# async def remove_image_api(image):
#     filePath = f'./image/{image}.png'
#     os.remove(filePath)
#     return {"deleted":1}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)