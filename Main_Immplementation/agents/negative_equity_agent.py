"""
Negative Equity / Insolvency Signal Agent
Flags technical insolvency and aggressive capital destruction — strong fraud precursors.

Companies approaching fraud often show:
  - Negative total equity (liabilities exceed assets — technically insolvent)
  - Deeply negative retained earnings (accumulated losses well beyond paid-in capital)
  - Continuing to raise equity while destroying it (repeated dilution)
  - Equity shrinking even as the company reports profits

Checks:
  1. Negative total shareholders equity → technically insolvent
  2. Retained earnings as % of paid-in capital (reconstruction of capital erosion)
  3. Simultaneous large stock issuance + large negative retained earnings
  4. Equity vs liabilities balance (ultra-leveraged)
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentResult


class NegativeEquityAgent(BaseAgent):
    """
    Detects insolvency signals and capital-destruction patterns.

    Required fields:
      - total_shareholders_equity  (balance_sheet)
      - retained_earnings          (balance_sheet)

    Optional:
      - capital_stock              (balance_sheet)
      - additional_paid_in_capital (balance_sheet)
      - total_liabilities          (balance_sheet)
      - proceeds_stock_sales       (cash_flow)
    """

    def get_name(self) -> str:
        return "negative_equity"

    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        vals = self._extract_values(data)
        if vals is None:
            return self._create_result(
                score=0.0, confidence=0.0,
                findings=["Insufficient equity data"],
                metrics={}, success=False, error="Missing equity fields"
            )

        (equity, retained, cap_stock, apic, total_liabilities,
         stock_proceeds, available) = vals

        paid_in_capital = cap_stock + apic  # total invested by shareholders
        findings = []
        sub_scores = []

        # Check 1: Negative total equity → technical insolvency
        if equity < 0:
            deficit = abs(equity)
            score = min(95.0, 70 + deficit / max(1, total_liabilities) * 50)
            sub_scores.append(score)
            findings.append(
                f"CRITICAL: Negative shareholders equity (${equity:,.0f}) — "
                f"company is technically insolvent (liabilities exceed assets)"
            )

        # Check 2: Deeply negative retained earnings
        if retained < 0:
            if paid_in_capital > 0:
                erosion_ratio = abs(retained) / paid_in_capital
                if erosion_ratio > 1.0:
                    sub_scores.append(75.0)
                    findings.append(
                        f"Accumulated deficit (${retained:,.0f}) exceeds ALL paid-in capital "
                        f"(${paid_in_capital:,.0f}) — {erosion_ratio:.1f}× capital destroyed"
                    )
                elif erosion_ratio > 0.5:
                    sub_scores.append(45.0)
                    findings.append(
                        f"Accumulated deficit (${retained:,.0f}) has eroded "
                        f"{erosion_ratio:.0%} of paid-in capital (${paid_in_capital:,.0f})"
                    )
                else:
                    sub_scores.append(20.0)
                    findings.append(
                        f"Negative retained earnings (${retained:,.0f}) — "
                        f"accumulated losses represent {erosion_ratio:.0%} of paid-in capital"
                    )
            else:
                # No paid-in capital context
                if retained < -100_000_000:
                    sub_scores.append(40.0)
                    findings.append(
                        f"Large accumulated deficit (${retained:,.0f}) with minimal paid-in capital"
                    )

        # Check 3: Raising equity while destroying it
        if stock_proceeds > 0 and retained < 0:
            if abs(retained) > stock_proceeds * 0.5:
                sub_scores.append(35.0)
                findings.append(
                    f"Company raised ${stock_proceeds:,.0f} in new equity while carrying "
                    f"${retained:,.0f} accumulated deficit — shareholder dilution cycle"
                )

        # Check 4: Equity is tiny sliver of capital structure
        if total_liabilities > 0 and equity > 0:
            equity_pct = equity / (equity + total_liabilities)
            if equity_pct < 0.05:  # equity < 5% of total capital
                sub_scores.append(50.0)
                findings.append(
                    f"Equity is only {equity_pct:.1%} of total capital structure — "
                    f"extreme leverage leaves almost no buffer"
                )

        if not findings:
            findings.append(
                f"Equity position appears stable: ${equity:,.0f} total equity, "
                f"retained earnings: ${retained:,.0f}"
            )

        overall_score = max(sub_scores) if sub_scores else 0.0
        confidence = min(1.0, available / 4.0)

        metrics = {
            'total_shareholders_equity': round(equity, 2),
            'retained_earnings': round(retained, 2),
            'paid_in_capital': round(paid_in_capital, 2),
            'total_liabilities': round(total_liabilities, 2),
            'is_technically_insolvent': equity < 0,
            'erosion_ratio': round(abs(retained) / paid_in_capital, 3) if paid_in_capital > 0 else None,
            'equity_pct_of_capital': round(
                equity / (equity + total_liabilities), 4
            ) if total_liabilities > 0 and equity > 0 else None
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
            for section in ['balance_sheet', 'cash_flow', 'income_statement', 'financial_data']:
                sub = d.get(section, {})
                if isinstance(sub, dict):
                    for key in keys:
                        if key in sub and sub[key] not in (None, 'N/A', ''):
                            try:
                                return float(sub[key]), True
                            except (TypeError, ValueError):
                                pass
            return 0.0, False

        equity,  eq_ok  = get(data, 'ceq', 'total_shareholders_equity', 'balance_sheet_total_shareholders_equity')
        retained, re_ok = get(data, 're', 'retained_earnings', 'balance_sheet_retained_earnings')
        cap,      _     = get(data, 'capital_stock', 'balance_sheet_capital_stock')
        apic,     _     = get(data, 'additional_paid_in_capital', 'balance_sheet_additional_paid_in_capital')
        liabs,    _     = get(data, 'lt', 'total_liabilities', 'balance_sheet_total_liabilities')
        proceeds, _     = get(data, 'sstk', 'proceeds_stock_sales', 'cash_flow_proceeds_stock_sales')

        if not eq_ok or not re_ok:
            return None

        available = sum([eq_ok, re_ok, cap != 0, apic != 0, liabs != 0, proceeds != 0])
        return (equity, retained, cap, apic, liabs, proceeds, available)

    def is_applicable(self, data: Dict[str, Any]) -> bool:
        return self._extract_values(data) is not None
