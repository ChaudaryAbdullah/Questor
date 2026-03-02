"""
Depreciation Anomaly Agent
Detects unusually low depreciation rates — a common way to inflate asset values
and reported profits.

If a company depreciates assets too slowly:
  - Assets are overstated on the balance sheet
  - Expenses are understated → profits are inflated
  - Eventually requires large write-downs (often disclosed simultaneously with fraud)

Core metric:
  depreciation_rate = depreciation_amortization / fixed_assets_net

Industry-agnostic thresholds:
  < 2%  → assets depreciated over 50+ years (implausibly slow)
  2-4%  → slow but possible for real estate or infrastructure
  > 25% → very fast — possible aggressive write-down / cleanup
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentResult


class DepreciationAnomalyAgent(BaseAgent):
    """
    Flags suspiciously low or high depreciation rates.

    Required fields:
      - depreciation_amortization (income_statement)
      - fixed_assets_net          (balance_sheet)

    Optional:
      - total_assets              (balance_sheet)
      - total_revenues            (income_statement)
    """

    DEPR_VERY_SLOW = 0.02    # < 2%  → implausibly slow
    DEPR_SLOW      = 0.04    # < 4%  → slow, watch
    DEPR_FAST      = 0.25    # > 25% → aggressive write-down signal
    DEPR_VERY_FAST = 0.50    # > 50% → extreme

    # D&A / total_assets — general guide across industries
    DA_TO_ASSETS_LOW  = 0.01
    DA_TO_ASSETS_HIGH = 0.15

    def get_name(self) -> str:
        return "depreciation_anomaly"

    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        vals = self._extract_values(data)
        if vals is None:
            return self._create_result(score=0.0, confidence=0.0,
                findings=["Insufficient data: need depreciation and fixed assets"],
                metrics={}, success=False, error="Missing D&A or fixed asset fields")

        da, fixed_assets, total_assets, revenue, available = vals
        findings = []
        sub_scores = []

        # Check 1: Depreciation rate on fixed assets
        if fixed_assets > 0:
            depr_rate = da / fixed_assets
            if depr_rate < self.DEPR_VERY_SLOW:
                sub_scores.append(70.0)
                findings.append(
                    f"CRITICAL: Depreciation rate ({depr_rate:.1%}) implies assets "
                    f"depreciated over {1/depr_rate:.0f} years — likely overstated fixed assets"
                )
            elif depr_rate < self.DEPR_SLOW:
                sub_scores.append(35.0)
                findings.append(
                    f"Slow depreciation rate ({depr_rate:.1%}) — assets depreciated over "
                    f"{1/depr_rate:.0f} years, possible inflation of fixed asset values"
                )
            elif depr_rate > self.DEPR_VERY_FAST:
                sub_scores.append(50.0)
                findings.append(
                    f"Extremely high depreciation rate ({depr_rate:.1%}) — possible "
                    f"aggressive write-down or prior overstatement being corrected"
                )
            elif depr_rate > self.DEPR_FAST:
                sub_scores.append(20.0)
                findings.append(f"High depreciation rate ({depr_rate:.1%}) — faster than typical")
            else:
                findings.append(f"Depreciation rate ({depr_rate:.1%}) is within normal range")

            metrics_depr_rate = round(depr_rate, 4)
        else:
            metrics_depr_rate = None
            # Large D&A with zero/N/A fixed assets
            if da > 0 and total_assets > 0:
                da_to_assets = da / total_assets
                if da_to_assets > self.DA_TO_ASSETS_HIGH:
                    sub_scores.append(30.0)
                    findings.append(
                        f"High D&A-to-assets ratio ({da_to_assets:.1%}) with no fixed assets reported — "
                        f"possible off-balance-sheet asset or intangible write-down"
                    )

        # Check 2: D&A / Total Assets (cross-check)
        if total_assets > 0 and da > 0:
            da_assets = da / total_assets
            if da_assets < self.DA_TO_ASSETS_LOW:
                sub_scores.append(25.0)
                findings.append(
                    f"Very low D&A relative to total assets ({da_assets:.2%}) — "
                    f"possible under-depreciation across all asset classes"
                )

        # Check 3: Zero D&A with significant fixed assets
        if da == 0 and fixed_assets > 1_000_000:
            sub_scores.append(60.0)
            findings.append(
                f"ZERO depreciation reported despite ${fixed_assets:,.0f} in fixed assets — "
                f"possible omission or assets not being depreciated"
            )

        if not findings:
            findings.append("Depreciation policy appears reasonable")

        metrics = {
            'depreciation_amortization': round(da, 2),
            'fixed_assets_net': round(fixed_assets, 2),
            'depreciation_rate': metrics_depr_rate,
            'implied_useful_life_years': round(1 / metrics_depr_rate, 1) if metrics_depr_rate and metrics_depr_rate > 0 else None,
            'da_to_total_assets': round(da / total_assets, 4) if total_assets > 0 else None,
        }
        return self._create_result(score=max(sub_scores) if sub_scores else 0.0,
                                   confidence=min(1.0, available / 3.0),
                                   findings=findings, metrics=metrics)

    def _extract_values(self, data: Dict[str, Any]) -> Optional[tuple]:
        def get(d, *keys):
            for key in keys:
                if key in d and d[key] not in (None, 'N/A', ''):
                    try:
                        return float(d[key]), True
                    except (TypeError, ValueError):
                        pass
            for sec in ['income_statement', 'balance_sheet', 'cash_flow', 'financial_data']:
                sub = d.get(sec, {})
                if isinstance(sub, dict):
                    for key in keys:
                        if key in sub and sub[key] not in (None, 'N/A', ''):
                            try:
                                return float(sub[key]), True
                            except (TypeError, ValueError):
                                pass
            return None, False

        da,     da_ok  = get(data, 'dp', 'depreciation_amortization',
                             'income_statement_depreciation_amortization',
                             'depreciation_amortization_cf', 'cash_flow_depreciation_amortization')
        fixed,  fx_ok  = get(data, 'ppegt', 'fixed_assets_net', 'balance_sheet_fixed_assets_net')
        assets, _      = get(data, 'at', 'total_assets', 'balance_sheet_total_assets')
        rev,    _      = get(data, 'sale', 'total_revenues', 'income_statement_total_revenues')

        if not da_ok:
            return None
        da    = abs(da)
        fixed = abs(fixed) if fixed is not None else 0.0
        assets = assets or 0.0
        rev    = rev    or 0.0
        available = sum([da_ok, fx_ok, assets > 0, rev > 0])
        return (da, fixed, assets, rev, available)

    def is_applicable(self, data: Dict[str, Any]) -> bool:
        return self._extract_values(data) is not None
