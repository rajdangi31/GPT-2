# GPT-2 from Scratch: Architecting Transformers

An from-scratch, deep-dive implementation of the GPT architecture using standard PyTorch primitives. This repository is not a thin wrapper around a pre-trained model or high-level API; it is a ground-up derivation of the exact mechanisms that power large language models.

By building the causal self-attention mechanism, the feedforward multi-layer perceptrons, and the character-level custom tokenizer manually, this repository serves as a technical showcase for **Deep ML Fundamentals**.

## Objective & Narrative

The transition from a high-level API consumer to an ML fundamentalist requires peeling back the abstraction layers. In this project, I implement:
- **Custom Character-Level Tokenizer**: Bypassing external libraries like `tiktoken` to demonstrate the raw ingestion of text into integer space.
- **Causal Scaled Dot-Product Attention**: Implementing the query, key, and value (QKV) matrices mathematically alongside the lower-triangular causal masking (`tril`) to enforce autoregressivity.
- **Positional Embeddings**: Structuring the tensor manipulations so the attention heads recognize sequential permutations.
- **Multi-Head Self-Attention**: Splitting the embedding dimension across multiple parallel communication channels to capture diverse sequence semantics.

## Mathematical Architecture

The core of the network relies on the Attention mechanism formulated as:
```text
Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
```
In `model.py`, this is achieved precisely. We scale the dot-product affinities by `(C ** -0.5)` to ensure the variance remains stable at initialization, preventing the `softmax` from bottlenecking gradients early in the training phase.

## Empirical Convergence

Below is a loss plot detailing the trajectory of training over a standard dataset (Plato's Republic & Euthyphro dialogs). The divergence between `train` and `val` loss showcases the classic learning curve and points of optimal convergence prior to extreme over-fitting.

<p align="center">
  <img src="loss_plot.png?v=2" alt="Training Loss Convergence" width="600">
</p>

## Sample Output

Once trained on the Platonic dialogs using the standard cross-entropy objective, the network begins to dream mathematically. Despite operating entirely on a character level (having no hard-coded concept of the English word), it learns spaces, vocabulary, and even semantic structures such as dialogue turn-taking.

*Generated Sample (Epoch 5000)*:
```text
SOCRATES: Are we to say anything else, Euthyphro, or if we cannot?
EUTHYPHRO: I do not know what you mean, Socrates.
SOCRATES: Well now, the things that they say are pious, whether it is being loved by the gods or being different, I should say it is not being loved because it is something changed.
EUTHYPHRO: Necessarily.
SOCRATES: Then what is the pious and the impious? Do they disagree as to the things they do?
```

## System Execution

The codebase is organized modularly for strict logical separation of responsibilities:
```bash
# Verify environment dependencies
pip install -r requirements.txt

# Execute the training sequence and parameter update loop
python train.py --max_iters 5000 --batch_size 64
```

> **Note**: I am utilizing `AdamW` optimization traversing a dataset composed of approximately 1 million characters. Hardware acceleration is configured strictly, with silent fallbacks gracefully dropping to CPU tensors when `cuda` is unavailable.
