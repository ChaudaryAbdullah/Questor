# 🤖 Fraud Detection Agents

This folder contains **15 specialized financial fraud detection agents** that operate on a single year's financial data extracted from SEC filings. Each agent implements a distinct analytical technique, issues a risk score (0–100), and generates human-readable findings.

---

## Architecture

```
agents/
├── base_agent.py              # Abstract base class + AgentResult dataclass
├── orchestrator.py            # Loads, runs, and combines all agent scores
├── agent_config.yaml          # Per-agent enable/disable, weights, thresholds
│
├── ── Classic Formula Agents ──
├── benfords_law.py            # Digit distribution analysis
├── beneish_mscore.py          # Beneish M-Score (needs YoY data)
├── altman_zscore.py           # Altman Z-Score (bankruptcy risk)
│
├── ── Cash Flow Agents ──
├── cashflow_earnings_agent.py # Cash-flow vs. net income consistency
├── cashflow_composition_agent.py # Is operations the primary cash source?
├── financing_red_flags_agent.py  # Ponzi-like funding pattern detection
│
├── ── Balance Sheet Agents ──
├── debt_anomaly_agent.py      # Extreme leverage & debt structure
├── related_party_agent.py     # Related-party transaction exposure
├── asset_quality_agent.py     # Opaque / hard-to-verify asset concentration
├── negative_equity_agent.py   # Technical insolvency & capital erosion
├── liquidity_crunch_agent.py  # Short-term liquidity stress
│
└── ── Income Statement Agents ──
    ├── expense_padding_agent.py      # Inflated operating expenses
    ├── tax_rate_anomaly_agent.py     # Implausible effective tax rates
    ├── eps_consistency_agent.py      # EPS arithmetic cross-check
    └── depreciation_anomaly_agent.py # Abnormally slow/fast depreciation
```

---

## All 15 Agents at a Glance

| # | Agent Key | File | What It Detects | Needs YoY? |
|---|-----------|------|-----------------|-----------|
| 1 | `benfords_law` | `benfords_law.py` | Digit manipulation in financial numbers | No |
| 2 | `beneish_mscore` | `beneish_mscore.py` | Earnings manipulation via 8-ratio formula | **Yes** |
| 3 | `altman_zscore` | `altman_zscore.py` | Bankruptcy risk / financial distress | No |
| 4 | `cashflow_earnings` | `cashflow_earnings_agent.py` | Accrual ratio — cash vs. book income gap | No |
| 5 | `debt_anomaly` | `debt_anomaly_agent.py` | Extreme D/E ratio, ICR, debt concentration | No |
| 6 | `related_party` | `related_party_agent.py` | Related-party balance vs. liabilities/revenue | No |
| 7 | `expense_padding` | `expense_padding_agent.py` | Expense-to-revenue ratio, gross margin, salary bloat | No |
| 8 | `tax_rate_anomaly` | `tax_rate_anomaly_agent.py` | Implausible ETR — near-zero or extreme | No |
| 9 | `financing_red_flags` | `financing_red_flags_agent.py` | Negative CFO funded by stock issuances / new debt | No |
| 10 | `asset_quality` | `asset_quality_agent.py` | Other-assets and receivables as % of total assets | No |
| 11 | `eps_consistency` | `eps_consistency_agent.py` | Reported EPS vs. net income ÷ shares | No |
| 12 | `negative_equity` | `negative_equity_agent.py` | Negative equity, accumulated deficit erosion | No |
| 13 | `liquidity_crunch` | `liquidity_crunch_agent.py` | Cash ratio, quick ratio, cash burn vs. ending cash | No |
| 14 | `depreciation_anomaly` | `depreciation_anomaly_agent.py` | D&A rate implying 50+ year asset life (inflation) | No |
| 15 | `cashflow_composition` | `cashflow_composition_agent.py` | Operating vs. investing vs. financing cash sources | No |

> **14 of 15 agents work on a single year's data.** Only `beneish_mscore` requires year-over-year comparisons.

---

## How Agents Work

### Base Class (`base_agent.py`)

Every agent inherits from `BaseAgent` and must implement:

```python
class MyAgent(BaseAgent):
    def get_name(self) -> str:
        return "my_agent"   # must match the key in agent_config.yaml

    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        # extract values → compute metrics → return score + findings
        ...

    def is_applicable(self, data: Dict[str, Any]) -> bool:
        # return False if required fields are missing/N/A
        ...
```

### AgentResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` 0–100 | Risk score (100 = maximum fraud risk) |
| `confidence` | `float` 0–1 | How much data was available (1.0 = all fields present) |
| `findings` | `List[str]` | Human-readable explanation of risk factors |
| `metrics` | `Dict` | Raw computed values (ratios, scores, flags) |
| `success` | `bool` | `False` if agent returned "Data not applicable" |

### Data Input

Agents receive a **merged dict** containing:
1. The structured pipeline's model prediction results
2. The raw input JSON (`balance_sheet`, `income_statement`, `cash_flow`)

Each agent uses a defensive `_extract_values()` helper that searches both the flat root dict and all nested section dicts to find required fields. Returns `None` (triggers `is_applicable=False`) if critical fields are missing or `"N/A"`.

---

## Configuration (`agent_config.yaml`)

```yaml
agents:
  altman_zscore:
    enabled: true
    weight: 0.20      # contribution to combined agent score
    safe_zone: 2.99   # agent-specific thresholds
    grey_zone: 1.81

  tax_rate_anomaly:
    enabled: true
    weight: 0.08
```

- **Weight** is normalized across only the agents that successfully ran (so if 10/15 fire, their weights sum to 1.0)
- Set `enabled: false` to disable any agent without code changes
- `min_confidence: 0.3` — agents below this confidence are excluded from the combined score

---

## Orchestrator (`orchestrator.py`)

```python
from agents.orchestrator import AgentOrchestrator, load_agent_config

config = load_agent_config()               # loads agent_config.yaml
orch   = AgentOrchestrator(config)         # registers all 15 agents

results = orch.run_agents(data)            # Dict[str, AgentResult]
combined = orch.calculate_combined_score(results)  # weighted score 0–100
```

### Combined Score Logic
- Agents that returned `success=False` or `confidence < 0.3` are excluded
- Remaining weights are normalized to sum to 1.0
- `combined_score = Σ (agent_score × normalized_weight)`

---

## Adding a New Agent

1. Create `agents/my_agent.py` inheriting `BaseAgent`
2. Register in `orchestrator.py`:
   ```python
   from .my_agent import MyAgent
   self.agents['my_agent'] = MyAgent(self.config.get('agents', {}).get('my_agent', {}))
   ```
3. Add to `agent_config.yaml`:
   ```yaml
   my_agent:
     enabled: true
     weight: 0.10
   ```
