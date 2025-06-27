import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


# hyper-parameters
batch_size = 64  # how many independent sequence will be run in paraller
block_size = 256 # what is the max context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_emb = 384 # dim of embedding 
n_head = 6
n_layer = 6
dropout = 0.2
# --------------------------------------------------------------


torch.manual_seed(42)


# downloading the dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# reading the file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# here is all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from chars to ints
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encode takes a str and returns list of int
decode = lambda l: ''.join([itos[i] for i in l]) # decode takes list of int and converts it to str


# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # 90% train data and 10% val data
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """1 head of self attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention 
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (BxTxC) . (BxCxT) -> (BxTxT)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """mutltiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """simple linear layer followed by a non linearity"""
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)                  # batch mean
        xvar = x.var(1, keepdim=True)                    # batch var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize unit var
        self.out = self.gamma * xhat + self.beta
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]


class Block(nn.Module):
    """Transformer Block: communication followed by computation"""
    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits from the next token from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb) # final layer norm
        self.lm_head = nn.Linear(n_emb, vocab_size) 

    
    def forward(self, idx, targets=None):
        # idx and target are both (B,T) tensors of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # B,T,C -> (Batch,Training,Vocab_Size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, target=targets)
        return logits, loss
    

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size token
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)


# training_loop
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
for iter in tqdm(range(max_iters)):
    # once every 300 epochs printing the lossses
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {iter}: train_loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}")

    # get a sample batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))