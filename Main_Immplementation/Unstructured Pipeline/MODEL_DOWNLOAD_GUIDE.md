# How to Download Embedding Model - Solution 2

## Quick Start (Automated)

I've created a script that automates Solution 2 for you:

```bash
./download_embedding_model.sh
```

That's it! The script will:
- ✅ Download all 9 model files individually (more reliable than bulk download)
- ✅ Retry failed downloads automatically (3 attempts per file)
- ✅ Skip already downloaded files if you re-run it
- ✅ Verify all files after download
- ✅ Save to permanent cache: `~/.cache/torch/sentence_transformers/`

**Total download size**: ~91 MB

---

## What the Script Does

1. Creates cache directory: `~/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2`
2. Downloads these files from HuggingFace:
   - `config.json` (612 bytes)
   - `model.safetensors` (90.9 MB) ← The main model
   - `tokenizer_config.json` (350 bytes)
   - `vocab.txt` (232 KB)
   - `special_tokens_map.json` (112 bytes)
   - `tokenizer.json` (466 KB)
   - `sentence_bert_config.json` (53 bytes)
   - `config_sentence_transformers.json` (116 bytes)
   - `modules.json` (229 bytes)

3. Verifies all files are downloaded correctly

---

## If Download Fails

The script has built-in retry logic, but if it still fails:

### Option A: Re-run the Script
```bash
./download_embedding_model.sh
```
Already downloaded files will be skipped!

### Option B: Manual Download (Browser)
1. Visit: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main
2. Download each file listed above
3. Place them in: `~/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2/`

### Option C: Try at Different Time
Network issues may be temporary. Try:
- Different time of day
- Different network connection
- Using mobile hotspot

---

## Verify Model is Cached

After successful download, verify it works:

```bash
source venv/bin/activate
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('✅ Model loaded from cache!')"
```

You should see: `✅ Model loaded from cache!` (no download messages)

---

## Run the Full Pipeline

Once the model is cached:

```bash
./run.sh
```

The pipeline will:
- ✅ Use cached model (no download)
- ✅ Complete all phases including embeddings
- ✅ Work offline (after initial download)

---

## Troubleshooting

### "wget: command not found"
Install wget:
```bash
sudo apt-get update
sudo apt-get install wget
```

### "Permission denied"
Make script executable:
```bash
chmod +x download_embedding_model.sh
```

### Files downloading but pipeline still fails
Clear the cache and re-download:
```bash
rm -rf ~/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2
./download_embedding_model.sh
```

---

## Cache Location

The model is stored at:
```
~/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2/
```

This is a **permanent cache** - once downloaded, it's never downloaded again unless you delete this directory.

---

**Ready to try?** Run:
```bash
./download_embedding_model.sh
```
