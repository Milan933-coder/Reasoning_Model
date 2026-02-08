import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import os

# --- Configuration Class ---
class Config:
    # Checkpoint Paths
    PHASE1_CKPT = "/kaggle/input/gpt-2-reasoning/pytorch/default/1/phase1_checkpoint (1).pt"
    OUTPUT_DIR = "phase2_checkpoints"
    
    # Model Config
    STUDENT_MODEL = "gpt2-medium"
    STUDENT_DIM = 1024
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_SEQ_LEN = 768
    
    # Gate Locations (Must match Phase 1)
    GATE_LAYERS = [5, 11, 17, 23] 
    
    # Training Config
    BATCH_SIZE = 2
    GATE_THRESHOLD = 0.4 
    
    # Curriculum Config
    STAGE1_EPOCHS = 1
    STAGE2_EPOCHS = 2
    
    # Learning Rates
    LR_STUDENT = 5e-6
    LR_THINK = 1e-4

# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# --- Modules (Required Definitions) ---
class SelfAttentionRewardGate(nn.Module):
    # (Re-defining here for completeness, usually imported from modules.py)
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout)
        )
        self.summary_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.summary_attn = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)
        self.score_mlp = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None: key_padding_mask = (1.0 - attention_mask).bool()
        else: key_padding_mask = None
        
        x = self.ln1(hidden_states)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = hidden_states + attn_out
        
        x2 = self.ln2(x)
        x2 = self.ffn(x2)
        x = x + x2
        
        summary_q = self.summary_query.expand(hidden_states.size(0), -1, -1)
        context, _ = self.summary_attn(summary_q, x, x, key_padding_mask=key_padding_mask)
        return self.score_mlp(context.squeeze(1))

class ThinkingEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.think_vector = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
    def forward(self, hidden_states):
        return hidden_states + self.think_vector

# --- 1. Load Phase 1 Checkpoint ---
print(f"Loading Phase 1 Checkpoint from {Config.PHASE1_CKPT}...")
checkpoint = torch.load(Config.PHASE1_CKPT)

# Load Student
student = GPT2LMHeadModel.from_pretrained(Config.STUDENT_MODEL).to(Config.DEVICE)
# Load weights strictly for transformer, ignore head mismatches if any
student.transformer.load_state_dict(checkpoint['student_state'], strict=False)

tokenizer = GPT2Tokenizer.from_pretrained(Config.STUDENT_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Initialize Gates & Think Vectors
gates = nn.ModuleDict()
think_embeddings = nn.ModuleDict()

for layer_idx in Config.GATE_LAYERS:
    s_idx = str(layer_idx)
    gates[s_idx] = SelfAttentionRewardGate(Config.STUDENT_DIM).to(Config.DEVICE)
    think_embeddings[s_idx] = ThinkingEmbedding(Config.STUDENT_DIM).to(Config.DEVICE)

# Load Gates and Freeze
gates.load_state_dict(checkpoint['gates_state'])
for param in gates.parameters():
    param.requires_grad = False

# Optimizer
optimizer = AdamW([
    {'params': student.parameters(), 'lr': Config.LR_STUDENT}, 
    {'params': think_embeddings.parameters(), 'lr': Config.LR_THINK}
])

# --- Helper: Dynamic Forward Pass with Loops ---
def run_model_with_loops(input_ids, attention_mask):
    # 1. Embeddings
    hidden_states = student.transformer.wte(input_ids) + student.transformer.wpe(torch.arange(input_ids.size(1), device=Config.DEVICE))
    
    # Extended Mask
    extended_mask = attention_mask[:, None, None, :]
    extended_mask = (1.0 - extended_mask) * -10000.0

    # 2. Iterate Blocks
    current_block_idx = 0
    total_blocks = len(student.transformer.h)
    
    while current_block_idx < total_blocks:
        block = student.transformer.h[current_block_idx]
        hidden_states = block(hidden_states, attention_mask=extended_mask)[0]
        
        # Check Gate
        if current_block_idx in Config.GATE_LAYERS:
            gate_key = str(current_block_idx)
            gate_score = gates[gate_key](hidden_states, attention_mask=attention_mask)
            avg_score = gate_score.mean().item()
            
            # LOOP LOGIC (Simplified Unroll)
            if avg_score < Config.GATE_THRESHOLD:
                # Inject Thought
                hidden_states = think_embeddings[gate_key](hidden_states)
                # Re-run Block
                hidden_states = block(hidden_states, attention_mask=extended_mask)[0]
        
        current_block_idx += 1

    hidden_states = student.transformer.ln_f(hidden_states)
    logits = student.lm_head(hidden_states)
    return logits

# --- Helper: Dataset Loader ---
def get_dataset(categories, levels):
    print(f"Preparing Dataset for Categories: {categories} | Levels: {levels}")
    ds_list = []
    
    for cat in categories:
        try:
            # Using EleutherAI mirror as per previous discussion
            ds = load_dataset("EleutherAI/hendrycks_math", cat, split="train")
            ds_list.append(ds)
        except Exception as e:
            print(f"Warning: Could not load {cat}: {e}")
            
    if not ds_list: raise ValueError("No datasets loaded!")
    
    full_ds = concatenate_datasets(ds_list)
    
    # Filter Levels
    full_ds = full_ds.filter(lambda x: x['level'] in levels)
    # Filter Length
    full_ds = full_ds.filter(lambda x: len(x['solution']) < 512)
    
    print(f"Final Dataset Size: {len(full_ds)}")
    return full_ds

def collate_fn(batch):
    texts = [f"Problem: {x['problem']}\nAnswer: {x['solution']}" for x in batch]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=Config.MAX_SEQ_LEN)
    labels = inputs.input_ids.clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    return inputs.input_ids.to(Config.DEVICE), inputs.attention_mask.to(Config.DEVICE), labels.to(Config.DEVICE)

# ==========================================
# TRAINING: STAGE 1 (Foundations)
# ==========================================
print("\n" + "="*30)
print("STARTING STAGE 1: Algebra & Number Theory (Levels 1-3)")
print("="*30)

stage1_cats = ["algebra", "number_theory","counting_and_probability"]
stage1_levels = {'Level 1', 'Level 2', 'Level 3'}
dataset_s1 = get_dataset(stage1_cats, stage1_levels)
loader_s1 = torch.utils.data.DataLoader(dataset_s1, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

for epoch in range(Config.STAGE1_EPOCHS):
    total_loss = 0
    pbar = tqdm(loader_s1, desc=f"Stage 1 Epoch {epoch+1}")
    
    for step, (input_ids, attention_mask, labels) in enumerate(pbar):
        optimizer.zero_grad()
        logits = run_model_with_loops(input_ids, attention_mask)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_description(f"S1 Loss: {loss.item():.4f}")

# Save Stage 1 Intermediate
torch.save(student.state_dict(), f"{Config.OUTPUT_DIR}/stage1_student.pt")
torch.save(think_embeddings.state_dict(), f"{Config.OUTPUT_DIR}/stage1_think.pt")
print("Stage 1 Complete & Saved.")

# ==========================================
# TRAINING: STAGE 2 (Advanced)
# ==========================================
print("\n" + "="*30)
print("STARTING STAGE 2: ALL Topics (Levels 4-5)")
print("="*30)

# Categories: Algebra, Number Theory, AND Counting & Probability
stage2_cats = ["algebra", "number_theory", "counting_and_probability"]
# Levels: Unlock Hardest Difficulty
stage2_levels = {'Level 4', 'Level 5'} 

dataset_s2 = get_dataset(stage2_cats, stage2_levels)
loader_s2 = torch.utils.data.DataLoader(dataset_s2, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

for epoch in range(Config.STAGE2_EPOCHS):
    total_loss = 0
    pbar = tqdm(loader_s2, desc=f"Stage 2 Epoch {epoch+1}")
    
    for step, (input_ids, attention_mask, labels) in enumerate(pbar):
        optimizer.zero_grad()
        logits = run_model_with_loops(input_ids, attention_mask)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_description(f"S2 Loss: {loss.item():.4f}")

# Save Final
torch.save(student.state_dict(), f"{Config.OUTPUT_DIR}/final_student.pt")
torch.save(think_embeddings.state_dict(), f"{Config.OUTPUT_DIR}/final_think.pt")
print("Phase 2 Complete. Final Models Saved.")