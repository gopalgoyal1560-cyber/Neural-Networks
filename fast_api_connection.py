from fastapi import FastAPI
from pydantic import BaseModel,create_model
api = FastAPI()
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from json import loads
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s = {"Value":(str,...)}

User = create_model("User",**s)

seq_len = 64
d = 320
tokenizer = Tokenizer.from_file("token4.pt")
vocab_size = tokenizer.get_vocab_size()
class transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(d,d)
        self.wk = nn.Linear(d,d)
        self.wv = nn.Linear(d,d)

        self.final = nn.Linear(d,d)
        self.drop = nn.Dropout(p = 0.2)
        self.layer_norm = nn.LayerNorm(d)
        self.soft = nn.Softmax(dim=-1)
        self.layer_norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d,d*2),
            self.drop,
            nn.GELU(),
            nn.Linear(d*2,d)
        )
    def forward(self,x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(batch_size,seq_len,10,d//10)
        k = k.view(batch_size,seq_len,10,d//10)
        v = v.view(batch_size,seq_len,10,d//10)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        scores = q@k.transpose(-2,-1)
        scores = scores/(torch.sqrt(torch.tensor(d//10,device=device)))

        mask = torch.ones(seq_len,seq_len,device=device)
        triangle = torch.triu(mask,diagonal=1).bool()
        scores = scores.masked_fill(triangle,float("-inf"))
        prob = self.soft(scores)
        attention = prob@v
        attention = attention.transpose(1,2).reshape(batch_size,seq_len,d)
        output = self.final(attention)
        output = output + x

        output = self.layer_norm(output)
        feed = self.ffn(output)
        feed = feed + output
        feed = self.layer_norm2(feed)

        return feed


block = nn.ModuleList(
    [
        transformer(),
        transformer(),
        nn.Embedding(vocab_size,d),
        nn.Embedding(seq_len,d),
        nn.LayerNorm(d),
        nn.Linear(d,vocab_size)
    ]
)
block.to(device)
block.load_state_dict(torch.load("chatbot.pt", map_location=device))
import torch.nn.functional as F
@torch.no_grad()
def generate(text, temperature=0.5, top_k=20, max_new=60):
    block.eval()
    eos = tokenizer.token_to_id("<eos>")   # change to your EOS token, or None
    ids = tokenizer.encode(text, add_special_tokens=False).ids[-seq_len:]
    n_prompt = len(ids)
    for _ in range(max_new):
        x = torch.tensor(ids[-seq_len:], device=device).unsqueeze(0)
        pos = torch.arange(x.size(1), device=device).unsqueeze(0)
        h = block[2](x) + block[3](pos)
        h = block[1](block[0](h))
        logits = block[5](block[4](h))[0, -1] / temperature
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, -1), 1).item()
        if eos is not None and nxt == eos:
            break
        ids.append(nxt)
    return tokenizer.decode(ids[n_prompt:])


@api.post("/Post_values")
def post(user:User):
    data = user.model_dump()
    predictions = generate(data["Value"])
    return {
        "output":predictions
    }
