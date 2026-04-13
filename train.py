import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
from model import GPTConfig, GPTLanguageModel

def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 from Scratch")
    parser.add_argument('--max_iters', type=int, default=5000, help='Maximum training iterations')
    parser.add_argument('--eval_interval', type=int, default=500, help='Interval between evaluations')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of iterations for evaluation')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--block_size', type=int, default=256, help='Context length')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--generate_only', action='store_true', help='Only generate sample, do not train')
    args = parser.parse_args()

    # Model and Hyperparameter config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Using device: {device}")
    sys.stdout.reconfigure(encoding='utf-8')
    
    torch.manual_seed(1337)

    # ----------------- Data Loading & Tokenization -----------------
    if not os.path.exists('Plato.txt'):
        raise FileNotFoundError("Dataset 'Plato.txt' not found.")
        
    with open('Plato.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    print(f"[*] Dataset loaded. Vocabulary size: {vocab_size} characters.")

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split: str):
        data_src = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_src) - args.block_size, (args.batch_size,))
        x = torch.stack([data_src[i: i + args.block_size] for i in ix])
        y = torch.stack([data_src[i + 1: i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model: torch.nn.Module, eval_iters: int = 200):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # ----------------- Model Initialization -----------------
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.2
    )
    
    model = GPTLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ----------------- Training Loop -----------------
    train_loss_history = []
    val_loss_history = []
    step_history = []

    if not args.generate_only:
        print("[*] Commencing training loop...")
        for iter in range(args.max_iters):
            if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
                losses = estimate_loss(model, args.eval_iters)
                print(f"Step {iter:04d}: Train Loss {losses['train']:.4f} | Val Loss {losses['val']:.4f}")
                train_loss_history.append(losses['train'])
                val_loss_history.append(losses['val'])
                step_history.append(iter)

            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
        print("[*] Training completed.")
        
        # Plot and save
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        plt.plot(step_history, train_loss_history, label='Train Loss', color='cyan')
        plt.plot(step_history, val_loss_history, label='Val Loss', color='magenta')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('GPT-2 from Scratch: Training Convergence')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[*] Saved loss_plot.png")

        # Save a basic checkpoint if needed
        torch.save(model.state_dict(), 'model_weights.pth')
        print("[*] Model weights saved to model_weights.pth")
    else:
        if os.path.exists('model_weights.pth'):
            model.load_state_dict(torch.load('model_weights.pth', map_location=device))
            print("[*] Loaded existing model weights.")
        else:
            print("[!] Warning: Generating text with an untrained model.")

    # ----------------- Text Generation -----------------
    print("\n--- SAMPLE GENERATION ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_ids = model.generate(context, max_new_tokens=400)[0].tolist()
    generated_text = decode(generated_ids)
    print(generated_text)
    print("-------------------------\n")

if __name__ == '__main__':
    main()
