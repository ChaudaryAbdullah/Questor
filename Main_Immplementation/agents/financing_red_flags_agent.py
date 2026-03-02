"""
Financing Red Flags Agent
Detects Ponzi-like funding patterns where the company cannot self-sustain from
operations and instead relies on continuous external capital (stock issuances,
new debt) to survive.

Pattern flags:
  1. Negative operating cash flow + positive stock issuances
     → Company burns cash in ops but raises equity to compensate
  2. Negative operating cash flow + large new borrowings
     → Rolling debt to fund operating losses
  3. Proceeds from stock sales >> operating cash flow
     → Equity issuance is the primary cash source, not the business
  4. Net change in cash is negative despite financing activities
     → Even external capital can't keep up with the burn

All fields from the cash_flow section of the JSON.
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentResult


class FinancingRedFlagsAgent(BaseAgent):
    """
    Detects unsustainable financing patterns.

    Required fields:
      - net_cash_operating       (cash_flow)
      - proceeds_stock_sales     (cash_flow)
      - changes_notes_payable    (cash_flow)

    Optional:
      - net_cash_financing       (cash_flow)
      - net_change_cash          (cash_flow)
      - total_revenues           (income_statement) — for scale context
    """

    def get_name(self) -> str:
        return "financing_red_flags"

    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        vals = self._extract_values(data)
        if vals is None:
            return self._create_result(
                score=0.0, confidence=0.0,
                findings=["Insufficient cash flow data"],
                metrics={}, success=False, error="Missing cash flow fields"
            )

        (cfo, stock_proceeds, notes_change,
         net_financing, net_change_cash, revenue, available) = vals

        findings = []
        sub_scores = []

        # Flag 1: Negative operations + equity issuance
        if cfo < 0 and stock_proceeds > 0:
            ratio = stock_proceeds / abs(cfo) if cfo != 0 else 0
            score = min(80.0, 40 + ratio * 20)
            sub_scores.append(score)
            findings.append(
                f"Operating cash burn (${cfo:,.0f}) funded by stock issuances "
                f"(${stock_proceeds:,.0f}) — equity used to mask operational failure"
            )

        # Flag 2: Negative operations + new debt drawn (notes payable increase)
        if cfo < 0 and notes_change < 0:  # notes_change negative = repaid; positive = new debt
            pass  # paying down debt on negative cfo is even worse — caught by debt_anomaly
        if cfo < 0 and notes_change > abs(cfo) * 0.5:
            sub_scores.append(45.0)
            findings.append(
                f"New borrowings (${notes_change:,.0f}) covering operating cash deficit "
                f"(${cfo:,.0f}) — rolling debt to stay alive"
            )

        # Flag 3: Stock proceeds >> operating cash flow (equity is main cash source)
        if revenue > 0 and stock_proceeds > 0:
            proceeds_to_revenue = stock_proceeds / revenue
            if proceeds_to_revenue > 0.5:
                sub_scores.append(55.0)
                findings.append(
                    f"Stock issuances (${stock_proceeds:,.0f}) equal {proceeds_to_revenue:.0%} "
                    f"of revenue — business is primarily funded by external equity"
                )
            elif proceeds_to_revenue > 0.2:
                sub_scores.append(25.0)
                findings.append(
                    f"Elevated stock issuances ({proceeds_to_revenue:.0%} of revenue) — "
                    f"heavy reliance on equity financing"
                )

        # Flag 4: Even with financing, net cash is declining
        if net_change_cash is not None and net_change_cash < 0:
            if net_financing > 0 and cfo < 0:
                sub_scores.append(50.0)
                findings.append(
                    f"Net cash DECLINING (${net_change_cash:,.0f}) despite positive financing "
                    f"activities — burn rate exceeds all external capital"
                )
            elif net_change_cash < -abs(cfo) * 2:
                sub_scores.append(30.0)
                findings.append(
                    f"Cash position declining sharply (${net_change_cash:,.0f}) — "
                    f"liquidity crisis risk"
                )

        # Flag 5: Positive net income but negative operating cash flow
        # (already handled by cashflow_earnings agent — skip double-count)

        # No red flags found
        if not findings:
            if cfo > 0:
                findings.append(
                    f"Operating cash flow is positive (${cfo:,.0f}) — "
                    f"no dependency on external funding detected"
                )
            else:
                findings.append("Cash flow financing patterns appear normal")

        overall_score = max(sub_scores) if sub_scores else 0.0
        confidence = min(1.0, available / 4.0)

        metrics = {
            'net_cash_operating': round(cfo, 2),
            'proceeds_stock_sales': round(stock_proceeds, 2),
            'notes_payable_change': round(notes_change, 2),
            'net_cash_financing': round(net_financing, 2) if net_financing is not None else 'N/A',
            'net_change_cash': round(net_change_cash, 2) if net_change_cash is not None else 'N/A',
            'cfo_negative': cfo < 0,
            'equity_funded_ops': cfo < 0 and stock_proceeds > 0,
            'flags_triggered': len(sub_scores)
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
            for section in ['cash_flow', 'income_statement', 'balance_sheet', 'financial_data']:
                sub = d.get(section, {})
                if isinstance(sub, dict):
                    for key in keys:
                        if key in sub and sub[key] not in (None, 'N/A', ''):
                            try:
                                return float(sub[key]), True
                            except (TypeError, ValueError):
                                pass
            return None, False

        cfo, cfo_ok = get(data, 'net_cash_operating', 'cash_flow_net_cash_operating')
        stock, stk_ok = get(data, 'proceeds_stock_sales', 'cash_flow_proceeds_stock_sales', 'sstk')
        notes, _ = get(data, 'changes_notes_payable', 'cash_flow_changes_notes_payable')
        if notes is None:
            notes = 0.0
        net_fin, _ = get(data, 'net_cash_financing', 'cash_flow_net_cash_financing')
        net_chg, _ = get(data, 'net_change_cash', 'cash_flow_net_change_cash')
        revenue, _ = get(data, 'sale', 'total_revenues', 'income_statement_total_revenues')
        if revenue is None:
            revenue = 0.0

        if not cfo_ok:
            return None

        stock = stock if stock is not None else 0.0
        available = sum([cfo_ok, stk_ok, notes != 0, net_fin is not None, net_chg is not None])

        return (cfo, stock, notes, net_fin, net_chg, revenue, available)

    def is_applicable(self, data: Dict[str, Any]) -> bool:
        return self._extract_values(data) is not None
