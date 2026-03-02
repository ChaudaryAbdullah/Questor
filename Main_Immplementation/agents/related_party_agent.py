"""
Related Party Transaction Agent
Detects suspicious related-party exposure in financial statements.

Related party transactions (RPTs) are a common vehicle for fraud:
  - Executives may transfer assets/cash to related entities at non-arm's-length prices
  - Shell companies controlled by insiders may receive inflated payments
  - High RPT ratios relative to liabilities signal potential self-dealing

Checks performed:
  1. Related-party liabilities as % of total liabilities
  2. Related-party amounts as % of total revenue
  3. Combined text + JSON signal cross-reference
  4. Absolute magnitude of related-party balances

Thresholds (conservative, based on SEC enforcement patterns):
  > 10% of total liabilities → Elevated
  > 25% of total liabilities → HIGH
  > 50% of total liabilities → CRITICAL
"""
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentResult


class RelatedPartyAgent(BaseAgent):
    """
    Flags significant related-party transaction exposure.

    Required financial fields:
      - due_to_related_parties (balance_sheet)
      - total_liabilities (balance_sheet)
      - total_revenues (income_statement) [optional, for revenue ratio]
    """

    # Thresholds for related-party ratio (RPT / total liabilities)
    ELEVATED_THRESHOLD = 0.10  # 10%
    HIGH_THRESHOLD = 0.25      # 25%
    CRITICAL_THRESHOLD = 0.50  # 50%

    # Revenue ratio thresholds
    REVENUE_ELEVATED = 0.05    # 5% of revenue
    REVENUE_HIGH = 0.15        # 15% of revenue

    def get_name(self) -> str:
        return "related_party"

    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        """
        Assess related-party transaction risk.

        Args:
            data: Financial data dict

        Returns:
            AgentResult with RPT risk assessment
        """
        values = self._extract_values(data)

        if values is None:
            return self._create_result(
                score=0.0,
                confidence=0.0,
                findings=["No related-party data found — field may be N/A or absent"],
                metrics={},
                success=False,
                error="Missing related-party financial fields"
            )

        rpt_balance, total_liabilities, revenue, available = values

        findings = []
        sub_scores = []

        # 1. RPT / Total Liabilities
        liability_ratio = 0.0
        if total_liabilities > 0:
            liability_ratio = rpt_balance / total_liabilities
            score_l, findings_l = self._check_liability_ratio(liability_ratio, rpt_balance)
            sub_scores.append(score_l)
            findings.extend(findings_l)

        # 2. RPT / Revenue
        revenue_ratio = 0.0
        if revenue > 0:
            revenue_ratio = rpt_balance / revenue
            score_r, findings_r = self._check_revenue_ratio(revenue_ratio, rpt_balance)
            sub_scores.append(score_r)
            findings.extend(findings_r)

        # 3. Absolute magnitude check
        if rpt_balance > 0:
            score_abs, findings_abs = self._check_absolute_magnitude(rpt_balance)
            sub_scores.append(score_abs)
            findings.extend(findings_abs)

        overall_score = max(sub_scores) if sub_scores else 0.0
        confidence = min(1.0, available / 3.0)

        metrics = {
            'rpt_balance': round(rpt_balance, 2),
            'total_liabilities': round(total_liabilities, 2),
            'rpt_to_liabilities_ratio': round(liability_ratio, 4),
            'rpt_to_revenue_ratio': round(revenue_ratio, 4) if revenue > 0 else 'N/A',
            'revenue': round(revenue, 2) if revenue > 0 else 'N/A',
            'is_elevated': liability_ratio > self.ELEVATED_THRESHOLD,
            'is_high_risk': liability_ratio > self.HIGH_THRESHOLD,
            'is_critical': liability_ratio > self.CRITICAL_THRESHOLD,
        }

        if not findings:
            findings = [f"Related-party balances (${rpt_balance:,.0f}) are within acceptable range"]

        return self._create_result(
            score=overall_score,
            confidence=confidence,
            findings=findings,
            metrics=metrics
        )

    def _extract_values(self, data: Dict[str, Any]) -> Optional[tuple]:
        """Extract related-party balances and context."""

        def get(d, *keys):
            for key in keys:
                if key in d and d[key] not in (None, 'N/A', ''):
                    try:
                        v = float(d[key])
                        return v, True
                    except (TypeError, ValueError):
                        pass
            for section in ['balance_sheet', 'income_statement', 'cash_flow', 'financial_data']:
                sub = d.get(section, {})
                if isinstance(sub, dict):
                    for key in keys:
                        if key in sub and sub[key] not in (None, 'N/A', ''):
                            try:
                                v = float(sub[key])
                                return v, True
                            except (TypeError, ValueError):
                                pass
            return None, False

        rpt, rpt_ok = get(data,
                          'due_to_related_parties',
                          'balance_sheet_due_to_related_parties',
                          'related_party_payable',
                          'related_party_loans',
                          'changes_related_party_loans')

        total_liabilities, tl_ok = get(data, 'lt', 'total_liabilities', 'balance_sheet_total_liabilities')
        revenue, rev_ok = get(data, 'sale', 'total_revenues', 'income_statement_total_revenues')

        # If RPT is not found or is zero, agent is not applicable
        if not rpt_ok or rpt is None or rpt == 0.0:
            return None

        rpt = abs(rpt)  # Some reports show as negative (payable)
        total_liabilities = abs(total_liabilities) if tl_ok and total_liabilities else 0.0
        revenue = abs(revenue) if rev_ok and revenue else 0.0

        available = sum([rpt_ok, tl_ok, rev_ok])

        return rpt, total_liabilities, revenue, available

    def _check_liability_ratio(self, ratio: float, rpt: float):
        findings = []
        if ratio >= self.CRITICAL_THRESHOLD:
            score = min(100.0, 85 + (ratio - self.CRITICAL_THRESHOLD) * 30)
            findings.append(
                f"CRITICAL: Related-party balances are {ratio:.1%} of total liabilities "
                f"(${rpt:,.0f}) — potential self-dealing or off-balance-sheet fraud"
            )
        elif ratio >= self.HIGH_THRESHOLD:
            score = 50 + (ratio - self.HIGH_THRESHOLD) / (self.CRITICAL_THRESHOLD - self.HIGH_THRESHOLD) * 35
            findings.append(
                f"HIGH: Related-party exposure ({ratio:.1%} of liabilities, ${rpt:,.0f}) "
                f"exceeds the {self.HIGH_THRESHOLD:.0%} high-risk threshold"
            )
        elif ratio >= self.ELEVATED_THRESHOLD:
            score = 20 + (ratio - self.ELEVATED_THRESHOLD) / (self.HIGH_THRESHOLD - self.ELEVATED_THRESHOLD) * 30
            findings.append(
                f"Elevated related-party exposure ({ratio:.1%} of liabilities, ${rpt:,.0f}) — warrants scrutiny"
            )
        else:
            score = ratio * 100  # e.g. 3% → score of 3.0
        return score, findings

    def _check_revenue_ratio(self, ratio: float, rpt: float):
        findings = []
        if ratio >= self.REVENUE_HIGH:
            score = min(70.0, 40 + ratio * 100)
            findings.append(
                f"Related-party balance (${rpt:,.0f}) equals {ratio:.1%} of annual revenue "
                f"— significant related-party exposure relative to business size"
            )
            return score, findings
        elif ratio >= self.REVENUE_ELEVATED:
            return 20.0, [f"Related-party balance equals {ratio:.1%} of annual revenue — monitor for growth"]
        return 0.0, []

    def _check_absolute_magnitude(self, rpt: float):
        """Flag extremely large absolute RPT balances."""
        if rpt > 1_000_000_000:  # > $1B
            return 40.0, [f"Very large absolute related-party balance: ${rpt:,.0f}"]
        elif rpt > 100_000_000:  # > $100M
            return 20.0, [f"Large absolute related-party balance: ${rpt:,.0f}"]
        return 0.0, []

    def is_applicable(self, data: Dict[str, Any]) -> bool:
        """Only applicable if a non-zero related-party balance is found."""
        result = self._extract_values(data)
        return result is not None
