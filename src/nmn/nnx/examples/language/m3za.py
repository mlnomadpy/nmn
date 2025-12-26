import os
import time
import json
from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as orbax
import wandb
from datasets import load_dataset
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from mteb import MTEB, get_tasks
from tokenizers import Tokenizer
from nmn.nnx.nmn import YatNMN
from nmn.nnx.attention import RotaryYatAttention

# --- JAX Device and Mesh Setup ---
if jax.default_backend() == 'tpu':
    mesh = Mesh(mesh_utils.create_device_mesh((8, 1)), ('batch', 'model'))
else:
    num_devices = len(jax.devices())
    mesh_shape = (num_devices, 1)
    mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape), ('batch', 'model'))

# --- Modern Architecture Components ---

class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs = None):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(dim))

    def __call__(self, x):
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + self.eps) * self.weight

class ModernTransformerBlock(nnx.Module):
    """Transformer block with RMSNorm and Rotary YAT Attention."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, maxlen: int, *, rngs: nnx.Rngs, rate: float = 0.1):
        if mesh is not None:
            kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model')))
        else:
            kernel_init = nnx.initializers.xavier_uniform()

        # Store rngs for dropout during training
        self.rngs = rngs

        # Use RotaryYatAttention - combines RoPE with YAT attention
        # constant_alpha=True uses sqrt(2) as alpha scaling for attention scores
        self.attn = RotaryYatAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=maxlen,
            kernel_init=kernel_init,
            use_bias=False,
            dropout_rate=rate,
            # Alpha scaling for YAT attention
            constant_alpha=True,  # Use sqrt(2) as constant alpha
            rngs=rngs,
        )
        self.norm1 = RMSNorm(embed_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        
        self.ffn = nnx.Sequential(
            YatNMN(embed_dim, ff_dim, kernel_init=kernel_init, rngs=rngs, use_bias=False),
            nnx.Linear(ff_dim, embed_dim, kernel_init=kernel_init, rngs=rngs, use_bias=False)
        )
        self.norm2 = RMSNorm(embed_dim, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)

    def __call__(self, x, training: bool = False):
        # Pre-Norm Architecture
        # x = x + Drop(Attn(Norm(x)))
        h = self.norm1(x)
        attn_out = self.attn(h, deterministic=not training, rngs=self.rngs if training else None)
        x = x + self.dropout1(attn_out, deterministic=not training)
        
        h = self.norm2(x)
        ffn_out = self.ffn(h)
        x = x + self.dropout2(ffn_out, deterministic=not training)
        return x

class TokenEmbedding(nnx.Module):
    """Just Token Embeddings (No Absolute Positional Embeddings)."""
    def __init__(self, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)

    def __call__(self, x):
        return self.token_emb(x)

class MiniBERT(nnx.Module):
    """Modernized MiniBERT with Rotary YAT Attention, RMSNorm, and Tied Weights.
    
    This model uses RotaryYatAttention which combines:
    - Rotary Position Embeddings (RoPE) for position encoding
    - YAT attention formula: softmax((Q·K)² / (||Q-K||² + ε)) for geometric attention
    - YatNMN in the feed-forward network
    """
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, num_heads: int, feed_forward_dim: int, num_transformer_blocks: int, rngs: nnx.Rngs):
        self.embedding_layer = TokenEmbedding(vocab_size, embed_dim, rngs=rngs)
        # Pass maxlen to each block for RoPE precomputation
        self.transformer_blocks = [
            ModernTransformerBlock(embed_dim, num_heads, feed_forward_dim, maxlen=maxlen, rngs=rngs) 
            for _ in range(num_transformer_blocks)
        ]
        self.norm_final = RMSNorm(embed_dim, rngs=rngs)
        self.head_dim = embed_dim // num_heads

    def __call__(self, inputs, training: bool = False):
        x = self.embedding_layer(inputs)
        
        # RoPE is now handled internally by RotaryYatAttention
        for block in self.transformer_blocks:
            x = block(x, training=training)
            
        x = self.norm_final(x)
        
        # Weight Tying: Reuse embedding weights for output projection
        # x: [Batch, Seq, Dim]
        # emb: [Vocab, Dim]
        # logits = x @ emb.T -> [Batch, Seq, Vocab]
        embedding_weights = self.embedding_layer.token_emb.embedding.value
        logits = x @ embedding_weights.T
        return logits

    def embed(self, inputs, training: bool = False):
        """Gets embeddings before the final output layer."""
        x = self.embedding_layer(inputs)
        
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        return self.norm_final(x)

def create_model(rngs, config):
    return MiniBERT(
        maxlen=config['maxlen'], vocab_size=config['vocab_size'], embed_dim=config['embed_dim'],
        num_heads=config['num_heads'], feed_forward_dim=config['feed_forward_dim'],
        num_transformer_blocks=config['num_transformer_blocks'], rngs=rngs
    )

# --- Utilities ---
def mean_pooling(hidden_states, attention_mask):
    """Performs mean pooling on the token embeddings."""
    # attention_mask: [batch, seq_len] (1 for valid, 0 for pad)
    input_mask_expanded = jnp.expand_dims(attention_mask, -1) # [batch, seq_len, 1]
    sum_embeddings = jnp.sum(hidden_states * input_mask_expanded, axis=1)
    sum_mask = jnp.clip(input_mask_expanded.sum(axis=1), a_min=1e-9)
    return sum_embeddings / sum_mask

# --- Data Preprocessing (MLM) ---
def create_masked_lm_predictions(tokens, mask_prob, maxlen, vocab_size, mask_token_id, pad_token_id):
    labels = np.full(maxlen, -100)
    # Filter out PAD tokens. RoBERTa pad_token_id is typically 1.
    non_padding_indices = np.where(tokens != pad_token_id)[0]
    
    if len(non_padding_indices) == 0: return tokens, labels
    
    num_to_predict = max(1, int(round(len(non_padding_indices) * mask_prob)))
    masked_indices = np.random.choice(non_padding_indices, size=min(num_to_predict, len(non_padding_indices)), replace=False)
    
    labels[masked_indices] = tokens[masked_indices]
    for i in masked_indices:
        rand = np.random.rand()
        if rand < 0.8: tokens[i] = mask_token_id
        elif rand < 0.9: tokens[i] = np.random.randint(0, vocab_size)
    return tokens, labels

def process_dataset_for_mlm(dataset, tokenizer, maxlen, mask_prob, vocab_size):
    # RoBERTa special tokens
    mask_token_id = tokenizer.token_to_id("<mask>")
    pad_token_id = tokenizer.token_to_id("<pad>")
    
    # Safety check in case the tokenizer doesn't have these exact strings
    if mask_token_id is None: 
        print("Warning: '<mask>' not found, falling back to '[MASK]'")
        mask_token_id = tokenizer.token_to_id("[MASK]")
    if pad_token_id is None:
        print("Warning: '<pad>' not found, falling back to '[PAD]'")
        pad_token_id = tokenizer.token_to_id("[PAD]")

    def tokenize_pad_and_mask(examples):
        input_ids, labels = [], []
        for text in examples['text']:
            # Tokenizer.encode returns an Encoding object with .ids
            # RoBERTa tokenizer handles <s> and </s> automatically in encode()
            encoded = tokenizer.encode(text)
            tokens = encoded.ids[:maxlen]
            
            # Pad manually to ensure numpy consistency
            if len(tokens) < maxlen:
                tokens = tokens + [pad_token_id] * (maxlen - len(tokens))
            else:
                tokens = tokens[:maxlen]
                
            token_array = np.array(tokens)
            masked, label = create_masked_lm_predictions(token_array.copy(), mask_prob, maxlen, vocab_size, mask_token_id, pad_token_id)
            input_ids.append(masked.tolist())
            labels.append(label.tolist())
        return {'input_ids': input_ids, 'labels': labels}
    
    columns_to_remove = [col for col in dataset.column_names if col not in ['input_ids', 'labels']]
    dataset = dataset.map(tokenize_pad_and_mask, batched=True, batch_size=1000, remove_columns=columns_to_remove)
    return dataset.shuffle(buffer_size=10_000, seed=42)

# --- Data Preprocessing (Contrastive/SNLI) ---
def process_dataset_for_contrastive(dataset, tokenizer, maxlen):
    """
    Processes SNLI dataset for contrastive learning.
    Filters for 'entailment' (label 0) and creates (premise, hypothesis) pairs.
    """
    dataset = dataset.filter(lambda x: x['label'] == 0)
    
    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is None: pad_token_id = tokenizer.token_to_id("[PAD]")

    def tokenize_pairs(examples):
        pairs_input_ids = []
        for p, h in zip(examples['premise'], examples['hypothesis']):
            # Process Premise
            p_ids = tokenizer.encode(p).ids[:maxlen]
            if len(p_ids) < maxlen:
                p_ids = p_ids + [pad_token_id] * (maxlen - len(p_ids))
            else:
                p_ids = p_ids[:maxlen]
            
            # Process Hypothesis
            h_ids = tokenizer.encode(h).ids[:maxlen]
            if len(h_ids) < maxlen:
                h_ids = h_ids + [pad_token_id] * (maxlen - len(h_ids))
            else:
                h_ids = h_ids[:maxlen]
            
            pairs_input_ids.append([p_ids, h_ids])
            
        return {'input_ids': pairs_input_ids}

    columns_to_remove = dataset.column_names
    dataset = dataset.map(tokenize_pairs, batched=True, batch_size=1000, remove_columns=columns_to_remove)
    return dataset.shuffle(buffer_size=10_000, seed=42)

# --- JAX Loss and Step Functions (MLM) ---
def loss_fn_mlm(model, batch, training: bool):
    logits = model(batch['input_ids'], training=training)
    labels = batch['labels']
    logits_flat, labels_flat = logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
    loss_per_pos = optax.softmax_cross_entropy_with_integer_labels(logits=logits_flat, labels=labels_flat)
    num_masked = jnp.sum(labels_flat != -100)
    return jnp.where(num_masked > 0, jnp.sum(loss_per_pos) / num_masked, 0.0), logits

@nnx.jit
def train_step_mlm(model: MiniBERT, optimizer: nnx.Optimizer, batch):
    grad_fn = nnx.value_and_grad(lambda m, b: loss_fn_mlm(m, b, training=True), has_aux=True)
    (loss, _), grads = grad_fn(model, batch)
    optimizer.update(grads)
    return loss, model, optimizer

@nnx.jit
def eval_step_mlm(model: MiniBERT, batch):
    loss, _ = loss_fn_mlm(model, batch, training=False)
    return loss

# --- JAX Loss and Step Functions (Contrastive) ---
def contrastive_loss_fn(model, batch, training: bool, temperature: float = 0.05):
    input_ids = batch['input_ids'] # [B, 2, L]
    batch_size = input_ids.shape[0]
    flat_input_ids = input_ids.reshape(batch_size * 2, -1)
    
    hidden_states = model.embed(flat_input_ids, training=training)
    # Masking: Assume pad_token is typically non-zero in BERT/RoBERTa indices, 
    # but strictly speaking we should pass the pad_token_id to create the mask properly.
    # However, since we don't have pad_token_id inside JIT easily without partials,
    # we can rely on the fact that tokenizer.encode usually produces non-zero IDs for content.
    # Standard RoBERTa <pad> is 1. Content is > 3.
    # A safe bet is checking strict equality if we pass pad_token_id, or just > pad_token_id if pad is 0.
    # For RoBERTa where pad=1, we can't just do != 0.
    # Simplification: We will treat '1' as padding.
    attention_mask = (flat_input_ids != 1).astype(jnp.int32)
    
    embeddings = mean_pooling(hidden_states, attention_mask)
    embeddings = embeddings.reshape(batch_size, 2, -1)
    
    z1 = embeddings[:, 0, :]
    z2 = embeddings[:, 1, :]
    z1 = z1 / jnp.linalg.norm(z1, axis=1, keepdims=True)
    z2 = z2 / jnp.linalg.norm(z2, axis=1, keepdims=True)
    
    sim_matrix = jnp.matmul(z1, z2.T) / temperature
    labels = jnp.arange(batch_size)
    
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=sim_matrix, labels=labels)
    return jnp.mean(loss)

@nnx.jit
def train_step_contrastive(model: MiniBERT, optimizer: nnx.Optimizer, batch):
    grad_fn = nnx.value_and_grad(lambda m, b: contrastive_loss_fn(m, b, training=True), has_aux=False)
    loss, grads = grad_fn(model, batch)
    optimizer.update(grads)
    return loss, model, optimizer

# --- MTEB Wrapper ---
class MiniBERTForMTEB:
    """Wrapper class for MTEB compatibility."""
    def __init__(self, model: MiniBERT, tokenizer, maxlen: int, batch_size: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.batch_size = batch_size
        self._embed_fn = jax.jit(self.model.embed, static_argnames='training')

    def encode(self, sentences, task_name=None, prompt_name=None, **kwargs):
        """
        Encodes sentences. 
        Matches MTEB EncoderProtocol signature strictly: (sentences, task_name, prompt_name, **kwargs).
        """
        # Retrieve batch_size from kwargs if provided, else use class default
        batch_size = kwargs.get('batch_size', self.batch_size)
        
        all_embeddings = []
        pad_id = self.tokenizer.token_to_id("<pad>")
        if pad_id is None: pad_id = 0

        # Helper to process a single batch of text strings
        def process_batch_texts(text_batch):
            encoded_batch = self.tokenizer.encode_batch(text_batch)
            
            ids = np.full((len(text_batch), self.maxlen), pad_id, dtype=np.int32)
            mask = np.zeros((len(text_batch), self.maxlen), dtype=np.int32)
            
            for j, enc in enumerate(encoded_batch):
                seq = enc.ids[:self.maxlen]
                ids[j, :len(seq)] = seq
                # RoBERTa pad is 1, so we mask where id != pad_id
                mask[j, :len(seq)] = 1
            
            hidden_state = self._embed_fn(jax.device_put(ids), training=False)
            mask_exp = np.expand_dims(mask, -1).astype(np.float32)
            pooled = np.sum(np.asarray(hidden_state) * mask_exp, 1) / np.clip(mask_exp.sum(1), a_min=1e-9, a_max=None)
            
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            return pooled / (norms + 1e-12)

        # Handle MTEB DataLoader vs Standard List
        if isinstance(sentences, list):
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                all_embeddings.append(process_batch_texts(batch))
        else:
            # Assume sentences is a DataLoader that yields batches
            for batch in sentences:
                # FIX: MTEB DataLoaders yield dicts {'column_name': [texts]}, extract the list.
                if isinstance(batch, dict):
                    # Extract the first value (list of texts)
                    batch = next(iter(batch.values()))
                
                all_embeddings.append(process_batch_texts(batch))
                
        return np.vstack(all_embeddings)

# --- Main Functions ---
def main_pretrain():
    """Runs the MLM pre-training loop."""
    config = {
        'num_transformer_blocks': 12, 'maxlen': 1024,
        'embed_dim': 768, 'num_heads': 12, 'feed_forward_dim': 3072, 'batch_size': 32,
        'learning_rate': 1e-4, 'mask_prob': 0.15, 
        'max_tokens_to_process': 5_000_000, 
        'eval_interval': 500, 'eval_steps': 50, 'val_set_size': 2000,
        'checkpoint_interval': 5000, 'checkpoint_dir': './minibert_checkpoints',
        'wandb_project': 'fineweb-bert-combined-run'
    }
    config['checkpoint_dir'] = os.path.abspath(config['checkpoint_dir'])
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    max_iterations = config['max_tokens_to_process'] // (config['batch_size'] * config['maxlen'])
    last_checkpoint_path = ""

    wandb.init(project=config['wandb_project'], config=config, name="phase1_mlm")
    rngs = nnx.Rngs(0)
    
    # 1. Load Data Stream
    print("\n=== Data Loading ===")
    full_dataset = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)
    
    # 2. Load Pretrained Tokenizer (RoBERTa)
    print("Loading pretrained 'roberta-base' tokenizer...")
    tokenizer = Tokenizer.from_pretrained("roberta-base")
    tokenizer.enable_truncation(max_length=config['maxlen'])
    
    config['vocab_size'] = tokenizer.get_vocab_size()
    print(f"Vocab Size: {config['vocab_size']}")

    model = create_model(rngs, config)
    optimizer = nnx.Optimizer(model, optax.adamw(config['learning_rate']))

    print("\n=== Phase 1: MLM Pre-training ===")
    train_dataset = process_dataset_for_mlm(full_dataset.skip(config['val_set_size']), tokenizer, config['maxlen'], config['mask_prob'], config['vocab_size'])
    val_dataset = process_dataset_for_mlm(full_dataset.take(config['val_set_size']), tokenizer, config['maxlen'], config['mask_prob'], config['vocab_size'])
    train_iterator = iter(train_dataset.iter(batch_size=config['batch_size'], drop_last_batch=True))
    val_iterator = iter(val_dataset.iter(batch_size=config['batch_size'], drop_last_batch=True))

    start_time = time.time()
    for step in range(max_iterations):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataset.iter(batch_size=config['batch_size'], drop_last_batch=True))
            batch = next(train_iterator)
        
        sharding = NamedSharding(mesh, P('batch', None))
        sharded_batch = {k: jax.device_put(jnp.array(v), sharding) for k, v in batch.items()}
        loss, model, optimizer = train_step_mlm(model, optimizer, sharded_batch)
        wandb.log({"mlm/train_loss": loss.item()}, step=step)

        if (step + 1) % config['eval_interval'] == 0:
            print(f"MLM Step {step+1}/{max_iterations}, Loss: {loss.item():.4f}")

    path = os.path.join(config['checkpoint_dir'], 'mlm_final')
    checkpointer = orbax.PyTreeCheckpointer()
    _, param_state, _ = nnx.split(model, nnx.Param, ...)
    checkpointer.save(path, item=param_state)
    checkpointer.close()
    last_checkpoint_path = path
    print(f"MLM Pre-training finished. Checkpoint saved at {last_checkpoint_path}")
    
    wandb.finish()
    return last_checkpoint_path, config

def main_contrastive_tuning(mlm_checkpoint_path, config):
    """Runs Contrastive Post-Pretraining using SNLI."""
    print("\n=== Phase 2: Contrastive Post-Pretraining (SimCSE) ===")
    
    config['contrastive_lr'] = 5e-5
    config['contrastive_batch_size'] = 16
    config['contrastive_steps'] = 1000 
    
    wandb.init(project=config['wandb_project'], config=config, name="phase2_contrastive")
    
    rngs = nnx.Rngs(1)
    model = create_model(rngs, config)
    
    # Reload Tokenizer
    tokenizer = Tokenizer.from_pretrained("roberta-base")
    tokenizer.enable_truncation(max_length=config['maxlen'])
    
    print(f"Loading weights from {mlm_checkpoint_path}...")
    checkpointer = orbax.PyTreeCheckpointer()
    _, params_template, _ = nnx.split(model, nnx.Param, ...)
    restored_params = checkpointer.restore(mlm_checkpoint_path, item=params_template)
    nnx.update(model, restored_params)
    
    optimizer = nnx.Optimizer(model, optax.adamw(config['contrastive_lr']))
    
    print("Loading SNLI dataset for pairs...")
    # FIX: Added streaming=True to return an IterableDataset compatible with shuffle(buffer_size) and .iter()
    snli_dataset = load_dataset("snli", split="train", streaming=True)
    contrastive_dataset = process_dataset_for_contrastive(snli_dataset, tokenizer, config['maxlen'])
    train_iterator = iter(contrastive_dataset.iter(batch_size=config['contrastive_batch_size'], drop_last_batch=True))
    
    start_time = time.time()
    for step in range(config['contrastive_steps']):
        try:
            batch = next(train_iterator)
        except StopIteration:
            break
            
        sharding = NamedSharding(mesh, P('batch', None, None))
        sharded_batch = {k: jax.device_put(jnp.array(v), sharding) for k, v in batch.items()}
        
        loss, model, optimizer = train_step_contrastive(model, optimizer, sharded_batch)
        wandb.log({"contrastive/train_loss": loss.item()}, step=step)
        
        if (step + 1) % 100 == 0:
            print(f"Contrastive Step {step+1}/{config['contrastive_steps']}, Loss: {loss.item():.4f}, Time: {time.time()-start_time:.2f}s")
            start_time = time.time()

    path = os.path.join(config['checkpoint_dir'], 'contrastive_final')
    checkpointer = orbax.PyTreeCheckpointer()
    _, param_state, _ = nnx.split(model, nnx.Param, ...)
    checkpointer.save(path, item=param_state)
    checkpointer.close()
    print(f"Contrastive tuning finished. Checkpoint saved at {path}")
    
    return path

def main_eval(checkpoint_path, config):
    """Loads a model checkpoint and runs MTEB evaluation."""
    print(f"\n=== Phase 3: MTEB Evaluation ===")
    rngs = nnx.Rngs(0)
    model = create_model(rngs, config)
    
    # Reload Tokenizer
    tokenizer = Tokenizer.from_pretrained("roberta-base")
    tokenizer.enable_truncation(max_length=config['maxlen'])

    checkpointer = orbax.PyTreeCheckpointer()
    _, params_template, _ = nnx.split(model, nnx.Param, ...)
    restored_params = checkpointer.restore(checkpoint_path, item=params_template)
    nnx.update(model, restored_params)
    
    mteb_model = MiniBERTForMTEB(model=model, tokenizer=tokenizer, maxlen=config['maxlen'])
    tasks = ["STSBenchmark", "Banking77Classification.v2"]
    
    # FIX: Use get_tasks to convert string names to Task objects
    try:
        tasks = get_tasks(tasks=tasks)
    except Exception as e:
        print(f"Warning: Could not load specific tasks {tasks}: {e}")
        # Fallback to STS12 if specific tasks fail
        tasks = get_tasks(tasks=["STS12"])

    evaluation = MTEB(tasks=tasks)
    results = evaluation.run(mteb_model, output_folder="mteb_results", eval_splits=["test"])
    
    print("\n--- MTEB Evaluation Results ---")
    for task_result in results:
        print(f"\nTask: {task_result.task_name}")
        print(json.dumps(task_result.scores, indent=2))

if __name__ == '__main__':
    mlm_ckpt, config = main_pretrain()
    if mlm_ckpt:
        contrastive_ckpt = main_contrastive_tuning(mlm_ckpt, config)
        if contrastive_ckpt:
            main_eval(contrastive_ckpt, config)
    wandb.finish()
