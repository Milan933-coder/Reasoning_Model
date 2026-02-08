# train_phase1_kaggle_ddp.py
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Model, GPT2Tokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
class Config:
    STUDENT_MODEL = "gpt2-medium"
    TEACHER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

    STUDENT_DIM = 1024
    TEACHER_DIM = 1536

    GATE_LAYERS = [5, 11, 17, 23]
    LAYER_MAPPING = {5: 6, 11: 12, 17: 18, 23: 27}

    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 8
    LEARNING_RATE = 3e-4
    EPOCHS = 1
    MAX_SEQ_LEN = 512

    SINKHORN_BLUR = 0.05
    SINKHORN_SCALING = 0.7

    OUTPUT_FILE = "phase1_ddp_checkpoint.pt"
    SEED = 42

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_ddp():
    """
    Kaggle-safe DDP init.
    torchrun sets these env vars automatically.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    return local_rank, rank, world_size

# -----------------------------
# Sinkhorn Loss
# -----------------------------
def sinkhorn_loss(x, y, epsilon=0.05, n_iters=5):
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    y = y / (y.norm(dim=-1, keepdim=True) + 1e-8)

    C = 1 - torch.matmul(x, y.transpose(-2, -1))  # [B, N, M]

    B, N, M = C.shape
    f = torch.zeros(B, N, device=x.device)
    g = torch.zeros(B, M, device=x.device)

    for _ in range(n_iters):
        f = -epsilon * torch.logsumexp((g.unsqueeze(1) - C) / epsilon, dim=2)
        g = -epsilon * torch.logsumexp((f.unsqueeze(2) - C) / epsilon, dim=1)

    P = torch.exp((f.unsqueeze(2) + g.unsqueeze(1) - C) / epsilon)
    return torch.sum(P * C) / B

# -----------------------------
# Modules
# -----------------------------
class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)

class SelfAttentionRewardGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.pool = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, attention_mask):
        key_padding_mask = attention_mask == 0
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        q = self.query.expand(x.size(0), -1, -1)
        pooled, _ = self.pool(q, x, x, key_padding_mask=key_padding_mask)
        return self.head(pooled.squeeze(1))

class GateProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.gates = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        for l in Config.GATE_LAYERS:
            self.gates[str(l)] = SelfAttentionRewardGate(Config.STUDENT_DIM)
            self.projectors[str(l)] = Projector(Config.TEACHER_DIM, Config.STUDENT_DIM)

# -----------------------------
# Main Training
# -----------------------------
def main():
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(Config.SEED + rank)

    if rank == 0:
        print(f"DDP running with {world_size} GPUs")

    tokenizer = GPT2Tokenizer.from_pretrained(Config.STUDENT_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("gsm8k", "main", split="train")

    def collate(batch):
        texts = [b["question"] + " " + b["answer"] for b in batch]
        tok = tokenizer(texts, padding=True, truncation=True,
                        max_length=Config.MAX_SEQ_LEN, return_tensors="pt")
        return tok.input_ids, tok.attention_mask

    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE,
                        sampler=sampler, collate_fn=collate)

    teacher = AutoModelForCausalLM.from_pretrained(
        Config.TEACHER_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = GPT2Model.from_pretrained(Config.STUDENT_MODEL).to(device).half()
    gp = GateProjector().to(device)

    student = DDP(student, device_ids=[local_rank])
    gp = DDP(gp, device_ids=[local_rank])

    optimizer = AdamW([
        {"params": student.parameters(), "lr": 1e-5},
        {"params": gp.parameters(), "lr": Config.LEARNING_RATE}
    ])

    for epoch in range(Config.EPOCHS):
        sampler.set_epoch(epoch)
        pbar = tqdm(loader) if rank == 0 else loader

        optimizer.zero_grad()
        for step, (ids, mask) in enumerate(pbar):
            ids = ids.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                t_out = teacher(ids, output_hidden_states=True).hidden_states

            s_out = student(ids, attention_mask=mask, output_hidden_states=True).hidden_states

            loss = 0
            for l in Config.GATE_LAYERS:
                t = gp.module.projectors[str(l)](t_out[Config.LAYER_MAPPING[l] + 1].float())
                s = s_out[l + 1].float()
                d = sinkhorn_loss(s, t)
                target = torch.exp(-d * Config.SINKHORN_SCALING).detach()
                pred = gp.module.gates[str(l)](s, mask)
                loss += nn.MSELoss()(pred.squeeze(), target.expand(pred.size(0))) + 0.5 * d

            (loss / Config.ACCUMULATION_STEPS).backward()

            if (step + 1) % Config.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            if rank == 0:
                pbar.set_description(f"Loss {loss.item():.4f}")

    if rank == 0:
        torch.save({
            "student": student.module.state_dict(),
            "gates_projectors": gp.module.state_dict()
        }, Config.OUTPUT_FILE)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
