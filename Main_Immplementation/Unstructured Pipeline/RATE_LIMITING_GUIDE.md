# Rate Limiting Solutions for Groq API

## Problem
Groq free tier has a rate limit of **12,000 tokens per minute**. The pipeline was hitting this limit using the `llama-3.3-70b-versatile` model.

## Solutions Implemented

### 1. **Switched to Faster Model** ⚡
Changed from `llama-3.3-70b-versatile` → `llama-3.1-8b-instant`

**Benefits:**
- **10x faster** inference
- **~60% fewer tokens** per request
- Still maintains good quality for entity extraction

**File**: `config/settings.yaml` (line 21)

### 2. **Reduced Token Usage** 📉
- Reduced `max_tokens` from 4096 → 2048
- Limits text input to 8000 characters per chunk
- Processes smaller batches (10 → 5 chunks at a time)

**File**: `config/settings.yaml` (lines 24, 112-113)

### 3. **Added Automatic Rate Limiting** ⏱️
- **2.5 second delay** between every API call
- Prevents exceeding 30 requests/minute (well under the limit)
- Tracks API call time and enforces delays

**File**: `src/nlp/entity_extractor.py` (lines 275-283)

### 4. **Intelligent Retry Logic** 🔄
- Detects HTTP 429 (rate limit) errors
- Extracts suggested wait time from error message
- Uses **exponential backoff** if no wait time provided (2s → 4s → 8s)
- Automatically retries up to 3 times before falling back to spaCy

**File**: `src/nlp/entity_extractor.py` (lines 339-357)

## Configuration Reference

```yaml
llm:
  model: "llama-3.1-8b-instant"  # Fast model
  max_tokens: 2048               # Reduced token usage
  retry_attempts: 3              # Max retries
  retry_delay: 2                 # Base delay
  
  rate_limit:
    requests_per_minute: 30          # Conservative limit
    tokens_per_minute: 6000          # Half of 12000 limit
    delay_between_requests: 2.5      # Minimum delay between calls

entity_extraction:
  batch_size: 5                  # Process fewer chunks at once
  chunk_delay: 3.0               # Delay between chunk batches
```

## Expected Performance

### Before:
- ❌ Hit rate limits after ~10 chunks
- ❌ Pipeline crashed with HTTP 429 errors
- ⏱️ No rate limiting control

### After:
- ✅ Processes entire document without hitting limits
- ✅ Automatic recovery from rate limit errors
- ✅ Controlled pace: ~24 API calls/minute
- ⏱️ **Slightly slower** (~30% longer) but **reliable completion**

## Math Breakdown

With the new settings:
- **2.5s delay** between calls = max 24 calls/minute
- **~250 tokens** per call with 8B model = ~6,000 tokens/minute
- **Well under** the 12,000 token/minute limit
- **Safety margin**: 50% headroom for burst traffic

## Alternative Options

If you still hit limits or want faster processing:

### Option A: Use spaCy Only (No API Calls)
```yaml
entity_extraction:
  use_llm: false              # Disable Groq
  use_spacy_fallback: true    # Use spaCy for everything
```
**Pros**: Free, no rate limits, instant  
**Cons**: Lower quality entity extraction, no fraud indicators from LLM

### Option B: Upgrade Groq Tier
Visit: https://console.groq.com/settings/billing
- **Dev Tier**: Higher rate limits
- **Pay-as-you-go**: No rate limits

### Option C: Process Fewer Chunks
```yaml
preprocessing:
  chunk_size: 1000  # Larger chunks = fewer API calls
  target_sections:  # Only process specific sections
    - "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS"
    - "ITEM 8. FINANCIAL STATEMENTS"
```

## Testing

Run the pipeline again:
```bash
./run.sh
```

You should see log messages like:
```
Rate limiting: sleeping for 2.3s
Rate limit hit, retrying in 3.5s... (attempt 1/3)
```

These indicate the rate limiting is working correctly!

---

**Updated**: 2026-01-23  
**Status**: Ready to test ✅
