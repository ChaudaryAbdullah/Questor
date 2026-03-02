"""
Asset Quality / Opacity Agent
Flags balance sheets dominated by opaque, hard-to-value, or unverifiable assets.

Fraudsters commonly inflate assets via:
  - Inflating "other assets" (intangibles, goodwill, deferred-tax assets)
  - Overstating accounts receivable (booking revenue not yet earned)
  - Capitalizing expenses as assets (hidden losses)
  - Phantom inventory or fixed assets

Checks:
  1. Other Assets concentration  (other_assets / total_assets)
  2. Receivables concentration   (accounts_receivable / total_assets)
  3. Combined opaque asset ratio ((other + receivables) / total_assets)
  4. Hard asset backing          (fixed_assets_net / total_assets — should be meaningful)
  5. Cash backing                (cash / total_assets — very low is risky)
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentResult


class AssetQualityAgent(BaseAgent):
    """
    Assesses balance sheet asset quality and opacity.

    Required fields:
      - total_assets       (balance_sheet)
      - other_assets       (balance_sheet)
      - accounts_receivable (balance_sheet)

    Optional:
      - cash               (balance_sheet)
      - fixed_assets_net   (balance_sheet)
      - inventory          (balance_sheet)
    """

    # Thresholds
    OTHER_ELEVATED = 0.25    # other_assets > 25% of total — elevated
    OTHER_HIGH     = 0.45    # other_assets > 45% of total — high risk
    OTHER_CRITICAL = 0.65    # other_assets > 65% of total — critical

    RECV_ELEVATED  = 0.30    # receivables > 30% of total — elevated
    RECV_HIGH      = 0.50    # receivables > 50% of total — high risk

    OPAQUE_HIGH    = 0.65    # other + receivables > 65% of total — combined opacity
    OPAQUE_CRIT    = 0.80    # > 80% — critical opacity

    HARD_ASSET_MIN = 0.10    # fixed_assets should be > 10% for most businesses
    CASH_MIN       = 0.02    # cash < 2% of assets → dangerous liquidity

    def get_name(self) -> str:
        return "asset_quality"

    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        vals = self._extract_values(data)
        if vals is None:
            return self._create_result(
                score=0.0, confidence=0.0,
                findings=["Insufficient balance sheet data"],
                metrics={}, success=False, error="Missing asset fields"
            )

        (total_assets, other_assets, receivables,
         cash, fixed_assets, inventory, available) = vals

        def ratio(num):
            return num / total_assets if total_assets > 0 else 0.0

        other_r  = ratio(other_assets)
        recv_r   = ratio(receivables)
        opaque_r = other_r + recv_r
        hard_r   = ratio(fixed_assets)
        cash_r   = ratio(cash)

        findings = []
        sub_scores = []

        # Check 1: Other assets opacity
        if other_r >= self.OTHER_CRITICAL:
            s = min(90.0, 75 + (other_r - self.OTHER_CRITICAL) * 50)
            sub_scores.append(s)
            findings.append(
                f"CRITICAL: Other/intangible assets are {other_r:.1%} of total assets "
                f"(${other_assets:,.0f}) — extremely opaque balance sheet"
            )
        elif other_r >= self.OTHER_HIGH:
            s = 50 + (other_r - self.OTHER_HIGH) / (self.OTHER_CRITICAL - self.OTHER_HIGH) * 25
            sub_scores.append(s)
            findings.append(
                f"HIGH: Other assets dominate balance sheet ({other_r:.1%}, ${other_assets:,.0f}) — "
                f"risk of inflated intangibles or goodwill"
            )
        elif other_r >= self.OTHER_ELEVATED:
            sub_scores.append(20.0)
            findings.append(f"Elevated other-assets ratio ({other_r:.1%}) — monitor for intangible inflation")

        # Check 2: Receivables concentration
        if recv_r >= self.RECV_HIGH:
            s = min(80.0, 55 + (recv_r - self.RECV_HIGH) * 100)
            sub_scores.append(s)
            findings.append(
                f"HIGH: Accounts receivable are {recv_r:.1%} of total assets "
                f"(${receivables:,.0f}) — potential fictitious or uncollectible revenue"
            )
        elif recv_r >= self.RECV_ELEVATED:
            sub_scores.append(25.0)
            findings.append(f"Elevated receivables ratio ({recv_r:.1%}) — possible aggressive revenue recognition")

        # Check 3: Combined opacity
        if opaque_r >= self.OPAQUE_CRIT:
            sub_scores.append(85.0)
            findings.append(
                f"CRITICAL: Combined opaque assets (other + receivables) = {opaque_r:.1%} of total assets "
                f"— less than {1-opaque_r:.0%} of assets are verifiable"
            )
        elif opaque_r >= self.OPAQUE_HIGH:
            sub_scores.append(45.0)
            findings.append(
                f"High combined opacity ratio ({opaque_r:.1%}) — over half of assets are hard to verify"
            )

        # Check 4: Very low hard-asset backing (for non-financial companies)
        if fixed_assets > 0 and hard_r < self.HARD_ASSET_MIN and other_r > 0.3:
            sub_scores.append(30.0)
            findings.append(
                f"Very low fixed asset backing ({hard_r:.1%}) while other assets are high ({other_r:.1%}) "
                f"— balance sheet may be padded with intangibles"
            )

        # Check 5: Cash dangerously low relative to assets
        if cash_r < self.CASH_MIN and total_assets > 1_000_000:
            sub_scores.append(20.0)
            findings.append(
                f"Very low cash backing ({cash_r:.1%} of assets, ${cash:,.0f}) — "
                f"potential liquidity concealment"
            )

        if not findings:
            findings.append(
                f"Asset quality appears reasonable: other assets {other_r:.1%}, "
                f"receivables {recv_r:.1%} of total assets"
            )

        overall_score = max(sub_scores) if sub_scores else 0.0
        confidence = min(1.0, available / 5.0)

        metrics = {
            'total_assets': round(total_assets, 2),
            'other_assets': round(other_assets, 2),
            'accounts_receivable': round(receivables, 2),
            'cash': round(cash, 2),
            'fixed_assets_net': round(fixed_assets, 2),
            'other_assets_ratio': round(other_r, 4),
            'receivables_ratio': round(recv_r, 4),
            'combined_opaque_ratio': round(opaque_r, 4),
            'hard_asset_ratio': round(hard_r, 4),
            'cash_ratio_to_assets': round(cash_r, 4),
        }

        return self._create_result(
            score=overall_score, confidence=confidence,
            findings=findings, metrics=metrics
        )

    def _extract_values(self, data: Dict[str, Any]) -> Optional[tuple]:
        def get(d, *keys):
            for key in keys:
                if key in d and d[key] not in (None, 'N/A', ''):
                    try:
                        return float(d[key]), True
                    except (TypeError, ValueError):
                        pass
            for section in ['balance_sheet', 'income_statement', 'cash_flow', 'financial_data']:
                sub = d.get(section, {})
                if isinstance(sub, dict):
                    for key in keys:
                        if key in sub and sub[key] not in (None, 'N/A', ''):
                            try:
                                return float(sub[key]), True
                            except (TypeError, ValueError):
                                pass
            return 0.0, False

        total_assets, ta_ok = get(data, 'at', 'total_assets', 'balance_sheet_total_assets')
        other,        ot_ok = get(data, 'ivao', 'other_assets', 'balance_sheet_other_assets')
        recv,         rv_ok = get(data, 'rect', 'accounts_receivable', 'balance_sheet_accounts_receivable')
        cash,         _     = get(data, 'che', 'cash', 'balance_sheet_cash')
        fixed,        fx_ok = get(data, 'ppegt', 'fixed_assets_net', 'balance_sheet_fixed_assets_net')
        inventory,    _     = get(data, 'invt', 'inventory', 'balance_sheet_inventory')

        if not ta_ok or total_assets == 0:
            return None
        if not ot_ok and not rv_ok:
            return None

        available = sum([ta_ok, ot_ok, rv_ok, cash > 0, fx_ok, inventory > 0])
        return (total_assets, other, recv, cash, fixed, inventory, available)

    def is_applicable(self, data: Dict[str, Any]) -> bool:
        return self._extract_values(data) is not None
