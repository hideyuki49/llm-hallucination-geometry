# Runpod: Cache Inspection & Cleanup (Safe Procedure)

## Purpose

Free disk space on Runpod instances by inspecting and (optionally) removing Hugging Face / Transformers / Torch caches.

- **Warning**: This procedure includes destructive commands (`rm -rf`).
Read the “Confirm” steps before executing.

### 1) Check disk usage
```bash
df -h
```

---

### 2) Inspect cache directories (non-destructive)
List Hugging Face cached models
```bash
ls ~/.cache/huggingface/hub | grep models-- | sed 's/models--//'
```
Find large directories under /root (often where caches grow)
```bash
du -h -d 2 /root 2>/dev/null | sort -h | tail -n 30
```
Find large directories under / (broad check)
```bash
du -h -d 2 / 2>/dev/null | sort -h | tail -n 30
```

---

### 3) Decide what to delete

You have two safe options:

- **Option A** (Recommended): Delete only a specific model cache

- **Option B**: Delete all HF/Transformers/Torch caches

---

### 4A) Option A — Delete a specific model cache (recommended)
Locate the exact directory name
```bash
ls -1 ~/.cache/huggingface/hub | grep '^models--'
```
Confirm the target path (IMPORTANT)

Replace <TARGET_DIR> with the exact directory you want to remove.
```bash
TARGET_DIR="$HOME/.cache/huggingface/hub/<TARGET_DIR>"
echo "$TARGET_DIR"
ls -lah "$TARGET_DIR" | head
```
Delete the specific cache directory
```bash
rm -rf "$TARGET_DIR"
```

Example (Meta Llama):
```bash
rm -rf ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct
```

---

### 4B) Option B — Delete all caches (aggressive)
Confirm current cache sizes
```bash
du -sh ~/.cache/huggingface ~/.cache/transformers ~/.cache/torch 2>/dev/null
du -sh /root/.cache/huggingface 2>/dev/null
```
Delete caches
```bash
rm -rf ~/.cache/huggingface
rm -rf /root/.cache/huggingface
rm -rf ~/.cache/torch
rm -rf ~/.cache/transformers
```

---

### 5) Verify cleanup
```bash
df -h
du -sh ~/.cache 2>/dev/null
du -sh /root/.cache 2>/dev/null
```

---

### Notes

- Deleting caches is safe for reproducibility (models will be re-downloaded).

- If your workflow depends on cached weights for speed/cost, prefer Option A.

- Consider pinning models and logging download steps to execution_procedure.txt for full reproducibility.