"""
Debt Ratio Anomaly Agent
Detects extreme leverage and dangerous debt structures that commonly accompany fraud.

Excessive debt can indicate:
  - Balance sheet manipulation to hide liabilities
  - Aggressive acquisition strategies masking poor operations
  - Liquidity crises necessitating fraudulent reporting

Checks performed:
  1. Debt-to-Equity ratio (> 3.0 → HIGH, > 6.0 → CRITICAL)
  2. Debt Ratio / Total Leverage (> 0.80 → HIGH, > 0.90 → CRITICAL)
  3. Interest Coverage Ratio (< 1.5 → can't cover interest → HIGH risk)
  4. Short-term debt concentration (short-term > 60% of total debt)
  5. Debt-to-Revenue (total liabilities > 5× revenue → extreme)
"""
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentResult


class DebtAnomalyAgent(BaseAgent):
    """
    Flags extreme leverage and dangerous debt structures.

    Required financial fields:
      - total_liabilities
      - total_shareholders_equity
      - total_assets
      - interest_expense (income_statement)
      - profit_loss_before_taxes (EBIT proxy)
      - notes_payable_short (short-term debt)
      - notes_payable_long (long-term debt)
      - total_revenues
    """

    # Thresholds
    D2E_HIGH = 3.0          # Debt-to-Equity HIGH threshold
    D2E_CRITICAL = 6.0      # Debt-to-Equity CRITICAL threshold
    DEBT_RATIO_HIGH = 0.75  # Total debt / Assets HIGH threshold
    DEBT_RATIO_CRIT = 0.90  # Total debt / Assets CRITICAL threshold
    ICR_LOW = 2.0           # Interest Coverage Ratio LOW threshold
    ICR_CRITICAL = 1.0      # Interest Coverage Ratio CRITICAL threshold
    STD_CONCENTRATION = 0.6 # Short-term debt > 60% of total debt
    D2R_MAX = 5.0           # Debt-to-Revenue max tolerance

    def get_name(self) -> str:
        return "debt_anomaly"

    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        """
        Analyze debt structure for anomalies.

        Args:
            data: Financial data dict

        Returns:
            AgentResult with debt anomaly assessment
        """
        values = self._extract_values(data)

        if values is None:
            return self._create_result(
                score=0.0,
                confidence=0.0,
                findings=["Insufficient data: need liabilities and assets to assess debt structure"],
                metrics={},
                success=False,
                error="Missing required debt-related financial fields"
            )

        (total_liabilities, equity, total_assets, interest_expense,
         ebit_proxy, short_debt, long_debt, revenue, available) = values

        # Compute ratios safely
        def safe_div(n, d):
            return n / d if d != 0 else 0.0

        d2e = safe_div(total_liabilities, equity) if equity != 0 else 99.0
        debt_ratio = safe_div(total_liabilities, total_assets)
        icr = safe_div(ebit_proxy, abs(interest_expense)) if interest_expense != 0 else 99.0
        total_debt = short_debt + long_debt
        std_concentration = safe_div(short_debt, total_debt) if total_debt > 0 else 0.0
        d2r = safe_div(total_liabilities, revenue) if revenue > 0 else 0.0

        # Score each check
        scores = []
        flags = []

        # 1. Debt-to-Equity
        d2e_score, d2e_findings = self._check_d2e(d2e)
        scores.append(d2e_score)
        flags.extend(d2e_findings)

        # 2. Debt Ratio
        debt_ratio_score, dr_findings = self._check_debt_ratio(debt_ratio)
        scores.append(debt_ratio_score)
        flags.extend(dr_findings)

        # 3. Interest Coverage
        if interest_expense > 0:
            icr_score, icr_findings = self._check_icr(icr)
            scores.append(icr_score)
            flags.extend(icr_findings)

        # 4. Short-term debt concentration
        if total_debt > 0:
            std_score, std_findings = self._check_std_concentration(std_concentration, total_debt)
            scores.append(std_score)
            flags.extend(std_findings)

        # 5. Debt-to-Revenue
        if revenue > 0:
            d2r_score, d2r_findings = self._check_d2r(d2r)
            scores.append(d2r_score)
            flags.extend(d2r_findings)

        # Combined: weighted average (worst signals dominate)
        if scores:
            overall_score = sum(sorted(scores, reverse=True)[: max(1, len(scores) // 2)]) / max(1, len(scores) // 2)
        else:
            overall_score = 0.0

        confidence = min(1.0, available / 6.0)

        metrics = {
            'debt_to_equity': round(d2e, 3),
            'debt_ratio': round(debt_ratio, 4),
            'interest_coverage_ratio': round(icr, 3) if interest_expense > 0 else 'N/A',
            'short_term_debt_concentration': round(std_concentration, 4),
            'debt_to_revenue': round(d2r, 3),
            'total_liabilities': round(total_liabilities, 2),
            'total_assets': round(total_assets, 2),
            'equity': round(equity, 2),
            'checks_performed': len(scores)
        }

        findings = flags if flags else ["Debt structure is within acceptable ranges"]

        return self._create_result(
            score=overall_score,
            confidence=confidence,
            findings=findings,
            metrics=metrics
        )

    def _extract_values(self, data: Dict[str, Any]) -> Optional[tuple]:
        """Extract debt-related financial values."""

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

        total_liabilities, tl_ok = get(data, 'lt', 'total_liabilities', 'balance_sheet_total_liabilities')
        equity, eq_ok = get(data, 'ceq', 'total_shareholders_equity', 'balance_sheet_total_shareholders_equity')
        total_assets, ta_ok = get(data, 'at', 'total_assets', 'balance_sheet_total_assets')
        interest, int_ok = get(data, 'xint', 'interest_expense', 'income_statement_interest_expense')
        ebit, ebit_ok = get(data, 'ib', 'profit_loss_before_taxes', 'income_statement_profit_loss_before_taxes',
                            'profit_loss_operations', 'income_statement_profit_loss_operations')
        short_debt, sd_ok = get(data, 'dlc', 'notes_payable_short', 'balance_sheet_notes_payable_short')
        long_debt, ld_ok = get(data, 'dltt', 'notes_payable_long', 'balance_sheet_notes_payable_long')
        revenue, rev_ok = get(data, 'sale', 'total_revenues', 'income_statement_total_revenues')

        if not tl_ok or not ta_ok:
            return None

        available = sum([tl_ok, eq_ok, ta_ok, int_ok, ebit_ok, sd_ok, ld_ok, rev_ok])

        return (total_liabilities, equity, total_assets, abs(interest),
                ebit, short_debt, long_debt, revenue, available)

    def _check_d2e(self, d2e: float):
        if d2e < 0:
            return 70.0, [f"NEGATIVE equity (D/E = {d2e:.2f}): company is technically insolvent"]
        elif d2e >= self.D2E_CRITICAL:
            score = min(100.0, 75 + (d2e - self.D2E_CRITICAL) * 5)
            return score, [f"CRITICAL D/E ratio ({d2e:.2f}): extremely high leverage (> {self.D2E_CRITICAL}×)"]
        elif d2e >= self.D2E_HIGH:
            score = 40 + (d2e - self.D2E_HIGH) / (self.D2E_CRITICAL - self.D2E_HIGH) * 35
            return score, [f"High D/E ratio ({d2e:.2f}): above safe threshold of {self.D2E_HIGH}×"]
        return max(0.0, d2e * 5), []

    def _check_debt_ratio(self, debt_ratio: float):
        if debt_ratio >= self.DEBT_RATIO_CRIT:
            score = min(100.0, 80 + (debt_ratio - self.DEBT_RATIO_CRIT) * 200)
            return score, [f"CRITICAL debt ratio ({debt_ratio:.2%}): > {self.DEBT_RATIO_CRIT:.0%} of assets are liabilities"]
        elif debt_ratio >= self.DEBT_RATIO_HIGH:
            score = 40 + (debt_ratio - self.DEBT_RATIO_HIGH) / (self.DEBT_RATIO_CRIT - self.DEBT_RATIO_HIGH) * 40
            return score, [f"High debt ratio ({debt_ratio:.2%}): majority of assets are debt-financed"]
        return debt_ratio * 30, []

    def _check_icr(self, icr: float):
        if icr < self.ICR_CRITICAL:
            return 90.0, [f"CRITICAL: Interest coverage ratio ({icr:.2f}) < 1.0 — company cannot cover interest payments"]
        elif icr < self.ICR_LOW:
            score = 90 - (icr - self.ICR_CRITICAL) / (self.ICR_LOW - self.ICR_CRITICAL) * 50
            return score, [f"Low interest coverage ratio ({icr:.2f}) — at risk of default (below {self.ICR_LOW}× threshold)"]
        return max(0.0, 20 - (icr - self.ICR_LOW) * 2), []

    def _check_std_concentration(self, concentration: float, total_debt: float):
        if concentration > self.STD_CONCENTRATION:
            score = 30 + (concentration - self.STD_CONCENTRATION) / (1 - self.STD_CONCENTRATION) * 40
            return score, [
                f"High short-term debt concentration ({concentration:.1%} of total debt) "
                f"— refinancing risk and rollover risk"
            ]
        return 0.0, []

    def _check_d2r(self, d2r: float):
        if d2r > self.D2R_MAX:
            score = min(80.0, 50 + (d2r - self.D2R_MAX) * 5)
            return score, [f"Debt-to-Revenue ratio ({d2r:.2f}×) is extreme — liabilities greatly exceed annual revenue"]
        elif d2r > 2.0:
            return 20.0, [f"Elevated Debt-to-Revenue ratio ({d2r:.2f}×) — worth monitoring"]
        return 0.0, []

    def is_applicable(self, data: Dict[str, Any]) -> bool:
        return self._extract_values(data) is not None
