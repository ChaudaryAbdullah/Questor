"""
Cash Flow Composition Agent
Checks that operating cash flow is the primary cash source — not financing/investing.

A healthy company's cash flow structure:
  - net_cash_operating > 0  (core business generates cash)
  - net_cash_investing < 0  (investing in growth)
  - net_cash_financing ≈ 0  (modest borrowing/repayment)

Red flag structures:
  1. Investing positive + Operating negative → selling assets to fund losses
  2. Financing >> Operating → debt/equity is the life-blood, not the business
  3. All three negative → company burning cash on every front
  4. Operating cash << Net Income (already partially caught by cashflow_earnings,
     but here we look at the full composition of sources vs uses)
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentResult


class CashFlowCompositionAgent(BaseAgent):
    """
    Analyzes the structure of cash inflows and outflows.

    Required fields (cash_flow section):
      - net_cash_operating
    Optional:
      - net_cash_investing / sale_purchase_fixed_assets
      - net_cash_financing
      - net_change_cash
      - total_revenues (income_statement) — for scale
    """

    def get_name(self) -> str:
        return "cashflow_composition"

    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        vals = self._extract_values(data)
        if vals is None:
            return self._create_result(score=0.0, confidence=0.0,
                findings=["Insufficient data: need net_cash_operating"],
                metrics={}, success=False, error="Missing cash flow fields")

        cfo, cfi, cff, net_change, revenue, available = vals

        findings = []
        sub_scores = []

        total_inflows = max(cfo, 0) + max(cfi or 0, 0) + max(cff or 0, 0)
        total_outflows = abs(min(cfo, 0)) + abs(min(cfi or 0, 0)) + abs(min(cff or 0, 0))

        # Share of cash from operations vs total inflows
        cfo_share = cfo / total_inflows if total_inflows > 0 else None

        # Flag 1: Operations burning cash
        if cfo < 0:
            sub_scores.append(30.0)
            findings.append(f"Negative operating cash flow (${cfo:,.0f}) — core business is cash-negative")

        # Flag 2: Selling assets to cover operating losses
        if cfi is not None and cfi > 0 and cfo < 0:
            sub_scores.append(60.0)
            findings.append(
                f"Positive investing cash flow (${cfi:,.0f}) with negative operating CFO "
                f"(${cfo:,.0f}) — asset sales are funding operating losses"
            )

        # Flag 3: Financing dominates inflows
        if cff is not None and cff > 0:
            if cfo < 0:
                sub_scores.append(50.0)
                findings.append(
                    f"Debt/equity financing (${cff:,.0f}) is substituting for "
                    f"negative operating cash flow (${cfo:,.0f})"
                )
            elif cfo_share is not None and cfo < cff:
                sub_scores.append(25.0)
                findings.append(
                    f"Financing activities (${cff:,.0f}) > operating cash flow (${cfo:,.0f}) — "
                    f"company depends more on external capital than its own business"
                )

        # Flag 4: All three activities negative (burning cash everywhere)
        if cfi is not None and cff is not None and cfo < 0 and cfi < 0 and cff < 0:
            sub_scores.append(75.0)
            findings.append(
                f"ALL cash flow categories negative: operating (${cfo:,.0f}), "
                f"investing (${cfi:,.0f}), financing (${cff:,.0f}) — severe cash crisis"
            )

        # Flag 5: Operating cash fine but net change still negative by huge margin
        if cfo > 0 and net_change is not None and net_change < -abs(cfo) * 2:
            sub_scores.append(35.0)
            findings.append(
                f"Net cash change (${net_change:,.0f}) suggests massive capital outflows "
                f"despite positive operating CFO (${cfo:,.0f})"
            )

        # Flag 6: CFO as % of revenue (very low even if positive)
        if revenue and revenue > 0 and cfo > 0:
            cfo_margin = cfo / revenue
            if cfo_margin < 0.02:
                sub_scores.append(15.0)
                findings.append(
                    f"Operating cash flow margin ({cfo_margin:.1%}) extremely thin relative to revenue — "
                    f"almost no cash being generated per dollar of sales"
                )

        if not findings:
            findings.append(
                f"Cash flow composition healthy: operating (${cfo:,.0f}) is primary source"
            )

        structure = "healthy"
        if cfo < 0 and cfi is not None and cfi > 0: structure = "asset-sale-funded"
        elif cfo < 0 and cff is not None and cff > 0: structure = "debt/equity-funded"
        elif cfo < 0: structure = "loss-making"

        metrics = {
            'net_cash_operating':  round(cfo, 2),
            'net_cash_investing':  round(cfi, 2) if cfi is not None else 'N/A',
            'net_cash_financing':  round(cff, 2) if cff is not None else 'N/A',
            'net_change_cash':     round(net_change, 2) if net_change is not None else 'N/A',
            'cfo_share_of_inflows': round(cfo_share, 4) if cfo_share is not None else None,
            'cash_flow_structure': structure,
        }
        return self._create_result(score=max(sub_scores) if sub_scores else 0.0,
                                   confidence=min(1.0, available / 4.0),
                                   findings=findings, metrics=metrics)

    def _extract_values(self, data: Dict[str, Any]) -> Optional[tuple]:
        def get(d, *keys):
            for key in keys:
                if key in d and d[key] not in (None, 'N/A', ''):
                    try:
                        return float(d[key]), True
                    except (TypeError, ValueError):
                        pass
            for sec in ['cash_flow', 'income_statement', 'balance_sheet', 'financial_data']:
                sub = d.get(sec, {})
                if isinstance(sub, dict):
                    for key in keys:
                        if key in sub and sub[key] not in (None, 'N/A', ''):
                            try:
                                return float(sub[key]), True
                            except (TypeError, ValueError):
                                pass
            return None, False

        cfo, cfo_ok = get(data, 'net_cash_operating', 'cash_flow_net_cash_operating')
        cfi, _      = get(data, 'sale_purchase_fixed_assets', 'net_cash_investing',
                          'cash_flow_sale_purchase_fixed_assets')
        cff, _      = get(data, 'net_cash_financing', 'cash_flow_net_cash_financing')
        net_chg, _  = get(data, 'net_change_cash', 'cash_flow_net_change_cash')
        rev, _      = get(data, 'sale', 'total_revenues', 'income_statement_total_revenues')

        if not cfo_ok:
            return None
        available = sum([cfo_ok, cfi is not None, cff is not None, net_chg is not None, (rev or 0) > 0])
        return (cfo, cfi, cff, net_chg, rev, available)

    def is_applicable(self, data: Dict[str, Any]) -> bool:
        return self._extract_values(data) is not None
