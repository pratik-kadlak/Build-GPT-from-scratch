# 🧠 Bigram Transformer Language Model (TinyShakespeare)

This notebook implements a **character-level Transformer-based Bigram Language Model** trained on the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). It's built using **only PyTorch**, without external modeling libraries, to teach the core principles behind modern language models.

---

## 🚀 Features

- Multi-head self-attention from scratch
- Feedforward networks with residual connections
- Positional encoding using learnable embeddings
- Causal masking for autoregressive modeling
- Character-level tokenization and decoding
- Text generation after training

---

## 📦 Requirements

Install dependencies if not already available:

```bash
pip install torch tqdm
```

## 🗂 Dataset
Dataset: Tiny Shakespeare corpus

To download:
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
- Place input.txt in the same directory as this notebook/script.
  
## ⚙️ Hyperparameters
- batch_size = 64: Number of sequences per batch
- block_size = 256: Max context length (sequence length)
- n_emb = 384: Embedding dimension
- n_head = 6: Number of attention heads
- n_layer = 6: Number of transformer blocks
- dropout = 0.2: Dropout rate
- learning_rate = 3e-3
- max_iters = 5000
- eval_interval = 500: Evaluate loss every N steps
- eval_iters = 200: Use 200 batches to average loss

## 📝 Text Generation
After training:
- Model is seeded with a single token
- Generates one character at a time using predicted probability distribution
- Continues for max_new_tokens steps (default: 500)

## 🙏 Acknowledgments
Andrej Karpathy for minGPT and char-rnn
