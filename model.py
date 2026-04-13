import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    """Configuration for the GPTLanguageModel"""
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2


class Head(nn.Module):
    """
    One head of self-attention.
    
    In self-attention, each token emits a Query, Key, and Value vector.
    - Query: What information am I looking for?
    - Key: What information do I contain?
    - Value: If selected, what information do I propagate?
    
    The 'affinities' between tokens are simply the dot products of their Queries and Keys.
    We apply a causal mask (tril) to ensure tokens only attend to past/current tokens.
    """

    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Compute attention scores ("affinities")
        # Scaled by 1/sqrt(head_size) to keep the variance of the dot products near 1,
        # preventing the softmax from becoming too peaky initialization.
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        
        # Apply causal mask: tokens cannot "look ahead" into the future.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention operating in parallel.
    This allows the model to jointly attend to information from different representation subspaces.
    """

    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate the outputs of all attention heads along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, C)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    A multi-layer perceptron strictly applied independently to each token position.
    After tokens communicate via attention, they "think" individually via this feedforward network.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """
    A single Transformer decoder block.
    Consists of communication (Self-Attention) followed by computation (FeedForward),
    with residual connections and LayerNorm applied for stable deep gradients.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note the usage of 'pre-norm' architecture (LayerNorm applied before the residual payload)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    The full Transformer Decoder Language Model.
    Employs token embeddings, positional embeddings, and stacked transformer blocks
    to autoregressively predict the next token in a sequence.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token embedding: Maps a token vocabulary index to an initial vector representation.
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Positional embedding: Injecting sequence-order information.
        # Unlike standard neural networks, Attention itself is permutation-equivariant.
        # This table informs the model where tokens are situated relative to each other.
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        # The core computational trunk: stack of Transformer blocks.
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        # We leverage the device from the input indices for proper placement
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x)    # (B, T, C)
        x = self.ln_f(x)      # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Autoregressively sample tokens from the model distribution.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop the context to the maximum block length enforced by positional embeddings
            idx_cond = idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Focus on the probabilities of the final timestep
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # Sample from the next-token distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Concatenate to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        self.train()
        return idx
