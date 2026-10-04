# Corrections to My Learning Journey File

This file goes with my main file ("Initial idea and personal goals / MLP in NumPy / How I created the transformer"). I did not change the original, because it records what I understood at the time. Here I list what I got wrong or only half right, and what the correct understanding is.

Labels used:
- **Wrong**: the statement is incorrect.
- **Partly right**: the idea is close, but the explanation or reason is off.
- **Missing**: something important I left out.

---

## Part 0: My original goal

### 0.1 Training an 8B–16B parameter LLM (Partly right / unrealistic)
I thought I could build an 8–16 billion parameter model myself. The architecture is not the hard part. The hard parts are compute and data.
- Training needs about 16 bytes per parameter (weights, gradients, and the two AdamW states), so an 8B model needs well over 100 GB of GPU memory just for training state. That means many GPUs.
- It also needs a very large amount of text. A common rule of thumb is about 20 tokens per parameter, so billions of parameters means hundreds of billions of tokens.
- Published 8B-class models took on the order of a million GPU-hours.
- Parameter count alone does not decide quality. Data quantity, data quality and training compute matter just as much. A well-trained small model beats a badly trained large one.

### 0.2 "Convert the LLM into a chatbot" (Missing)
A model trained only on next-token prediction is a **base model**. It continues text and does not follow instructions or hold a conversation. To get a chatbot you need extra stages:
1. **Supervised fine-tuning (SFT)** on instruction/conversation data, written in a chat template.
2. **Preference tuning** (RLHF, DPO and similar) to improve helpfulness and safety.
3. For an **agent**, tool-use training or prompting, plus a loop that runs tools and feeds results back to the model.

---

## Part 1: MLP and basic neural network concepts

### 1.1 Weights "form the neuron" (Wrong)
Weights are the strengths of connections. A **neuron** is one unit that computes `sum(inputs * weights) + bias` and then applies an activation. In a layer, each column of `W` holds the weights of one neuron.

### 1.2 Bias is a "threshold added to each element of the weight matrix" (Partly right)
- Bias is added to the output of the dot product, not to the weight matrix.
- There is one bias per neuron (a vector). It is broadcast across all rows (samples) in the batch.
- It works like a threshold in the sense that it shifts when a neuron "turns on", but it does not make neurons "biased toward some values". It gives the layer an offset so the output does not have to pass through zero.

### 1.3 Shapes for the dot product (Wrong)
I said the weight matrix is random numbers "of the same shape" as the input. That only works by accident for square matrices. For `z = X @ W + b` with `X` of shape `(N, in_features)`, `W` must be `(in_features, out_features)` and `b` must be `(out_features,)`. The output is `(N, out_features)`.

### 1.4 "Each neuron is connected to some neurons of the next layer" (Wrong)
In a dense (fully connected) layer, every neuron is connected to every neuron of the next layer. That is exactly what the matrix multiplication does.

### 1.5 Why ReLU is used (Partly right)
- Leaky ReLU is `x` for `x > 0` and `0.01 * x` for `x < 0`. It is not "0 or 0.01".
- The main reason for activation functions is **non-linearity**. Without them, any number of stacked linear layers collapses into a single linear layer and the network cannot learn complex patterns.
- ReLU does not "remove neurons that give no useful information". It is a simple non-linearity that is cheap and trains well.

### 1.6 GELU "uses current and past values" (Wrong)
GELU is a smooth, stateless function of the **current value only**: roughly `x * Φ(x)`, where `Φ` is the normal cumulative distribution. It has no memory. It is used in transformers because it is smooth and has worked well in practice.

### 1.7 Sigmoid vs softmax (Partly right)
- **Sigmoid** squashes each value independently into 0–1. It is used for binary classification, and also for multi-label problems where several classes can be true at once.
- **Softmax** turns a whole row into probabilities that **sum to 1**. It is used when exactly one class is correct (multi-class).

### 1.8 Loss can be "positive or negative" (Wrong)
Common losses (MSE, cross-entropy) are always **≥ 0**. They measure *how wrong* the model is, not in which direction.
- The signed difference `(prediction - target)` is the **error**. It shows up inside the gradient and tells us which way to move.
- Loss is the single score we want to minimize.

### 1.9 Backpropagation vs gradient descent (Wrong / mixed up)
I mixed these two up.
- **Backpropagation** is the algorithm that *computes* all the gradients using the chain rule, going from the loss backward through the layers.
- **Gradient descent** (SGD, Adam and so on) is the step that *uses* those gradients to update the parameters.

It is also wrong to say we "may have a direct formula for dL/dW but gradient descent is better". There is no direct formula that skips the chain. The chain rule is how we get dL/dW.

### 1.10 The backward chain for earlier layers (Wrong)
I wrote that to move from the last layer to the previous one we compute `dW/da` (gradient of the weights w.r.t. the previous activation). That is wrong. **Weights do not depend on activations**, so `dW/da` is meaningless here.

The correct chain, for layer `L` with `z_L = a_{L-1} @ W_L + b_L` and `a_L = f(z_L)`, is:

```
dL/dz_L        = (starting gradient from loss, or from activation if there is one)
dL/dW_L        = a_{L-1}.T @ dL/dz_L
dL/db_L        = sum of dL/dz_L over the batch
dL/da_{L-1}    = dL/dz_L @ W_L.T          <- this is how the gradient moves back one layer
dL/dz_{L-1}    = dL/da_{L-1} * f'(z_{L-1})  <- through the activation (elementwise)
```

Then the same steps repeat for layer `L-1`, and so on to the first layer. So the quantity passed backward is `dL/da`, and it is obtained using `W` (specifically `dz/da_prev = W`), not `dw/da`.

Another detail: with softmax + cross-entropy, the combined gradient simplifies to `dL/dz = probabilities - one_hot_targets`. That is why the softmax derivative is often not computed separately.

### 1.11 SGD and batches (Partly right)
- "SHG" was a typo for **SGD**. Strictly, SGD uses one sample at a time. What I did with batches is **mini-batch gradient descent**, which most people also call SGD.
- Learning rate: too high does not just make values "large". It can **overshoot the minimum or make the loss diverge**. Too low makes learning very slow and it can get stuck. Values like 3e-4 are typical for Adam/AdamW, while plain SGD often uses larger values.
- Batching is not required for the network to learn "at all". Full-batch training works, but it is slow, needs a lot of memory and gives fewer updates per pass. Mini-batches are a good trade-off between cost and gradient quality.

### 1.12 Epoch vs iteration (Wrong)
- **Epoch** = one full pass through the whole training set.
- **Iteration / step** = one update on one batch.

So 1000 rows with batch size 10 is 100 iterations per epoch. I wrote "epoch (iteration range)", which mixed the two.

### 1.13 Overfitting, underfitting and data leakage (Partly right / mixed)
- **Overfitting**: the model memorizes details and noise of the training set. Training loss is low but validation/test loss is high. It usually happens when the model is too large for the data, the data is small, or training runs too long. It is not caused by data patterns being "too complex".
- **Data leakage** is a **separate problem**. It is when test information accidentally gets into training, so test results look falsely good. It is not a type of overfitting.
- **Underfitting**: the model is too simple or undertrained to capture the patterns, so it does badly on **both** training and test data. It is not simply "not giving enough patterns".

### 1.14 Shuffling (Partly right)
Shuffling does not make the model "see more patterns". It stops the model from depending on the **order** of the data and makes each batch a better sample of the whole dataset.

### 1.15 Dropout (Wrong)
Dropout randomly sets a fraction `p` of activations to **0 during training only**, and scales the rest by `1/(1-p)`. It does not make neurons "go to higher changes". Its purpose is to stop neurons from **co-adapting** (relying on specific other neurons), which acts as regularization against overfitting. At inference it is turned off (`model.eval()` in PyTorch).

---

## Part 2: Tokenization (BPE)

### 2.1 How my BPE worked (mostly right)
The core idea was right: start from characters, count the most frequent adjacent pair, merge it, and repeat. Details I got wrong or left out:
- The final **vocab** should be the **base characters (or bytes) + one new token per merge**. It is not just "the unique values in the merged text".
- The **merge list, in the order learned**, is what you need to tokenize *new* text. You apply the merges in that same order.
- Real tokenizers do a **pre-tokenization** step (splitting on spaces/punctuation) and usually work on **bytes**, which means unseen characters never need an `<unk>`.
- In my example I wrote `1` where I meant the letter `l`.
- The library tokenizer was faster mainly because it is implemented in compiled code (Rust/C), not because it is a different idea.

---

## Part 3: Transformer

My model is a **decoder-only (GPT-style) transformer** with causal masking. Using that name helps me describe it correctly.

### 3.1 Data layout and targets (Missing)
The input `x` has shape `(batch, seq_len)`. The target `y` is the **same sequence shifted by one token** (`y = x` moved one step left). This is **next-token prediction**. Every position predicts the next token, so one sequence gives `seq_len` training examples at once.

### 3.2 "Multiple transformers form a multilayer perceptron" (Wrong)
Stacked transformer blocks do not form an MLP. Each transformer block **contains** a small MLP (the feed-forward network) plus attention. The stack is a sequence of blocks, with the output of one as the input of the next.

### 3.3 Embeddings of "variable length" (Wrong)
Every token has an embedding of the **same fixed size** `d_model`. The embedding table has shape `(vocab_size, d_model)` and is a lookup (the same as a one-hot vector times a matrix). The values are trainable.

### 3.4 Position embeddings (mostly right)
Correct that without position information attention cannot tell "dog bites man" from "man bites dog", because attention on its own ignores order. My model used **learned** position embeddings, so the table has only `seq_len` rows. That is also why input longer than `seq_len` fails at inference.

### 3.5 Q, K, V (mostly right)
Three separate linear layers on the embeddings give Q, K and V. My analogy (question / index / content) is a good intuition. They are not just "numbers": each one is a learned projection of the input.

### 3.6 Splitting into heads (Partly right)
- I wrote that the last dimension is divided "into 2 parts". It is divided into `n_heads` parts: `d_model = n_heads * head_dim`.
- Shape: `(B, T, C) -> (B, T, H, D) -> (B, H, T, D)`. It is a reshape and transpose, so you are right that it adds **no new parameters**.
- But heads are not "just a shape" in effect. Each head runs its **own attention** on its own slice, so different heads can learn different relationships.

### 3.7 Attention scores (Partly right)
- The product is `Q @ K^T` (the key is **transposed** on its last two dimensions), giving shape `(B, H, T, T)`.
- It is scaled by `1 / sqrt(head_dim)`. Without scaling, large dot products push softmax into extreme values and gradients become tiny.
- The result of `Q K^T / sqrt(d)` is the "score". After softmax it becomes the **attention weights** (each row sums to 1). Multiplying those weights by `V` gives the **attention output**, a weighted average of the values. I used "attention score" for the final output, which is not the standard term.

### 3.8 Causal masking (Partly right)
- It is called **causal masking**, not "attention head masking".
- It applies to **positions/tokens**, not "neurons".
- The part of the matrix above the diagonal is set to `-inf` before softmax, so those weights become 0. Each token can only attend to itself and earlier tokens.
- Real reason: without it the model could copy the answer (the next token) from the input during training, so it would learn nothing useful.

### 3.9 Output projection after attention (Wrong reason)
I said we pass the attention output through another linear layer "to train the attention score too". The real reasons are:
- After merging the heads back (`(B,H,T,D) -> (B,T,C)`), the output projection **mixes information across heads**.
- It lets the model learn how to write the attention result back into the main stream.

### 3.10 Residual connections (Wrong reason)
Adding the input back to the block output is not mainly "so the original values aren't useless". The main reasons are:
- It creates a direct path for **gradients** to flow backward, which makes deep networks trainable.
- Each block only has to learn a *change* to the representation, not rebuild it.

### 3.11 Layer normalization (Partly right)
LayerNorm does not just "scale down" values. For each token's vector it **subtracts the mean and divides by the standard deviation**, then applies learned `gamma` (scale) and `beta` (shift). This keeps activations stable. It works per token across features, not across the batch.

I also described the order *attention -> add -> norm -> FFN -> add -> norm*. This is **post-LN** (original Transformer). GPT-2 and most modern models use **pre-LN** (norm applied *before* attention and FFN, with the residual added after), which trains more stably. Both work, but I should know which one my code uses.

### 3.12 Feed-forward network (Missing explanation)
I said we add an FFN "to capture more patterns" and that "we use this excuse for everything". The actual roles are different:
- **Attention** moves information **between** positions.
- **FFN** processes each position **independently**, adds non-linearity and stores much of the model's learned knowledge.
- A standard FFN expands to about `4 * d_model`, applies GELU, then projects back.

### 3.13 Final loss step (Partly right)
- The final linear layer (the "LM head") outputs **logits** of shape `(B, T, vocab_size)`.
- These are reshaped to `(B*T, vocab_size)` and the targets flattened to `(B*T)`.
- `nn.CrossEntropyLoss` expects **raw logits**. It applies log-softmax internally, so I should not apply softmax myself before it.
- PyTorch's autograd does the backpropagation. The principle is the same as in my NumPy MLP.

### 3.14 AdamW (Partly right)
- Adam keeps running averages of the **gradients** (momentum) and of the **squared gradients** (to give each parameter its own adaptive step size). It does not use "previous values of weights".
- AdamW's difference: **weight decay is decoupled** from the gradient update. It directly shrinks the weights each step instead of being mixed into the gradient as L2 regularization. That is why it is preferred for transformers.
- Training usually also uses a **learning-rate warmup and decay schedule** and gradient clipping.

### 3.15 Special tokens (Partly right)
- `<pad>` is used to make sequences the same length in a batch. It does not teach the model "space". Pad positions should be ignored in the loss (`ignore_index`) and ideally masked in attention.
- `<eos>` teaches the model when to stop. This is correct.
- `<unk>` is for unknown tokens. Byte-level BPE rarely needs it.

---

## Part 4: Inference

### 4.1 `no_grad` and `eval` (Missing)
I turned gradients off with `torch.no_grad()`. I also need **`model.eval()`** so dropout is switched off during generation. Otherwise the output is noisier than it should be.

### 4.2 Context length (Partly right)
Input longer than `seq_len` fails because the **position embedding table only has `seq_len` rows**, not because of a generic "shape error". Taking `tokens[-seq_len:]` keeps the most recent context, and padding short prompts is fine as long as the model was trained with the same `<pad>` handling.

### 4.3 Temperature (Wrong)
I explained temperature as scaling "belief in the model's probabilities". That is not what it does.
- Temperature divides the **logits before softmax**: `softmax(logits / T)`.
- `T < 1` makes the distribution **sharper** (the top tokens dominate, more deterministic). `T > 1` makes it **flatter** (more random, more variety).
- It is a knob I choose to trade off safe vs creative output. It has nothing to do with "how strong my belief is" in the model.
- `T = 0` would divide by zero, so greedy decoding (always take the argmax) is handled as a special case. The "cap of 2.0" is just a common practical limit, not a rule.

### 4.4 Top-k and sampling (mostly right)
Keeping only the `k` highest logits, setting the rest to `-inf`, applying softmax and sampling with `torch.multinomial` is correct. Note that sampling picks tokens **in proportion to their probability**. It does not always pick the highest one.

### 4.5 Generation loop (Missing efficiency note)
My loop re-runs the full model on the entire window for each new token. Real systems use a **KV cache**, which stores the keys and values of earlier tokens so they are not recomputed. Dropping the first token to keep the window at `seq_len` is also a simple form of context limit that loses older text.

---

## Summary of my biggest misunderstandings

| Topic | What I believed | What is correct |
|---|---|---|
| Backprop vs gradient descent | Gradient descent computes the chain of gradients | Backprop computes gradients; gradient descent uses them to update |
| Backward chain | Uses `dW/da` | Uses `dL/da_prev = dL/dz @ W.T` |
| Loss | Can be positive or negative | Always non-negative; the error carries the sign |
| GELU | Uses past values | Function of the current value only |
| Dropout | Makes neurons change more | Randomly zeroes activations to prevent co-adaptation |
| Temperature | Relates to belief in the model | Divides logits to sharpen or flatten the distribution |
| Residual connections | Keep original values useful | Let gradients flow and ease training of deep stacks |
| AdamW | Uses previous weight values | Uses running averages of gradients; decoupled weight decay |
| Goal | Parameter count is the main challenge | Compute, data and post-training (SFT/RLHF) are the main challenges |

## What I should study next
- Matrix shapes and the matrix form of backprop (a small amount of calculus goes a long way here, and I do not need heavy math).
- Train/validation split and tracking validation loss to detect overfitting.
- Pre-LN vs post-LN, and positional encodings like RoPE.
- KV cache, learning-rate schedules and mixed-precision training.
- Fine-tuning a small base model (SFT) to turn it into a chatbot, before attempting to pretrain anything large.
