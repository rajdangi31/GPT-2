import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparamters
batch_size = 32 # Number of independent sequences that we will process in parallel.
block_size = 8 # Maximum context length for predictions.
max_iters = 6000
eval_interval = 600
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 400
n_embd = 32
 
#_______________________________________________________________________

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/rajdangi31/GPT-2/refs/heads/main/Plato.txt

with open('Plato.txt' , 'r' , encoding='utf-8') as f:
    text = f.read()

# Unique characters that occur in the text.

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Creating a mapping from characters to integers. {TOKENIZATION}

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # Encoder - takes a string & outputs a list of integeres.
decode = lambda l: ''.join([itos[i] for i in l]) # Decoder - takes a list of integers, outputs a string.

# Train and splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # First 90% of the data is going to be for training, and the remaining will be for valuation.
train_data = data[:n]
val_data = data[n:]

# Data Loading
def get_batch(split):

    # Generate a small batch of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train' , 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # Computing attention scores
        wei = q @ k.transpose(-2, -1) * C** -0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, C) @ (B, T, C) --> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range (num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
    

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """ Transformer Block : Communication followed by computation"""
    
    def __init__(self, n_embd, n_head):
        # n_embd : Embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x

# Simplistic Bigram Model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Token reading off the logits for the next token from a lookup table of vocab_size * vocab_size 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )
        self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 Heads of 8-Dimenstional self-attention
        self.ffwd = FeedForward(n_embd)
        self.ln_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        
        B, T = idx.shape
        
        # idx and targets are both (B , T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_heads(x) # Applying one head of self-attention. (B, T, C)
        x = self.ffwd(x) # (B, T, C)
        logits = self.ln_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B , T , C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # Getting the predictions
            logits , loss = self(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :] # Becomes (B, C)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1) #(B, C)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # Appending sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Every once in a while evaluate the loss on train and evaluation sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sampling a batch of data
    xb , yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))
