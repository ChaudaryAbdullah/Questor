# spaCy vs Groq API - Quick Comparison Guide

## Current Setting: **spaCy-Only Mode** ✅

The pipeline is now configured to use **only spaCy** for entity extraction, with **no Groq API calls**.

## How It Works

The entity extractor (`src/nlp/entity_extractor.py`) has three extraction methods that run in sequence:

1. **LLM Extraction** (Groq) - ❌ **DISABLED**
2. **spaCy NER** - ✅ **ACTIVE**
3. **Pattern Matching** (regex) - ✅ **ACTIVE**

## Configuration

**File**: `config/settings.yaml`

```yaml
entity_extraction:
  use_llm: false              # ← Set to false for spaCy-only
  use_spacy_fallback: true    # ← Set to true to enable spaCy
  spacy_model: "en_core_web_lg"
```

## Comparison

| Feature | Groq API (LLM) | spaCy Only |
|---------|----------------|------------|
| **Speed** | Slower (2.5s delay/call) | ⚡ **Instant** |
| **Cost** | API limits (12K tokens/min) | 🆓 **Free** |
| **Rate Limits** | Yes (can hit limits) | ❌ **None** |
| **Setup** | Requires API key | ✅ **Ready** |
| **Quality** | High (understands context) | Good (rule-based) |
| **Entities** | Custom fraud indicators | Standard NER |
| **Relationships** | Extracts relationships | ⚠️ **Limited** |
| **Runtime** | 8-12 minutes | **~3-5 minutes** |

## What spaCy Extracts

spaCy's `en_core_web_lg` model recognizes these entity types:

- **ORG** → Mapped to `COMPANY`
- **PERSON** → Mapped to `PERSON`
- **MONEY** → Mapped to `FINANCIAL`
- **PERCENT** → Mapped to `FINANCIAL`
- **DATE** → Mapped to `TEMPORAL`
- **GPE** → Mapped to `LOCATION`
- **LAW** → Mapped to `REGULATORY`

Plus **regex patterns** for:
- Monetary values ($1.5M, $500K)
- Percentages (15.3%)
- Fiscal years (FY 2023)
- Quarters (Q1 2023)
- SEC items (ITEM 7. MD&A)

## What You'll Miss Without Groq

🔴 **Limitations of spaCy-only:**
- No fraud-specific entity extraction
- Limited relationship extraction
- No semantic understanding of financial terms
- Won't identify hidden subsidiaries or related party transactions
- No confidence scores from LLM context

## Running the Pipeline

No code changes needed! Just run normally:

```bash
./run.sh
```

**Expected runtime**: 3-5 minutes (much faster!)

## Switching Back to Groq

If you want to re-enable Groq API later:

```yaml
entity_extraction:
  use_llm: true   # ← Change back to true
```

## Hybrid Mode (Best of Both)

You can use **both simultaneously** for maximum accuracy:

```yaml
entity_extraction:
  use_llm: true              # Both enabled
  use_spacy_fallback: true   # Both enabled
```

The pipeline will:
1. Try Groq first
2. Use spaCy as backup if Groq fails
3. Merge results from both
4. Keep higher-confidence entities

## Alternative: Sample-Based Processing

Process only a subset of chunks with Groq to stay under limits:

```yaml
entity_extraction:
  use_llm: true
  batch_size: 5
  # Add this to process only first 100 chunks
  max_chunks_llm: 100  # Stop after 100 LLM calls
```

(Note: This requires a small code modification)

## Recommendation

**For development/testing**: Use spaCy-only (current setting)
**For production/final analysis**: Use Groq with rate limiting
**Best accuracy**: Hybrid mode with rate limiting

---

**Current Mode**: spaCy-Only ✅  
**API Calls**: None 🆓  
**Rate Limits**: None ⚡  
**Ready to run**: Yes ✅
