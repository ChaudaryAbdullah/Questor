"""
EPS Consistency Agent
Verifies that the reported Earnings Per Share (EPS) is arithmetically consistent
with net income and shares outstanding.

Discrepancies indicate:
  - Manipulation of the share count used in EPS calculation
  - Use of non-GAAP adjustments without clear disclosure
  - Data entry errors or deliberate misstatement

Formula:
  computed_eps = net_income_loss / shares_outstanding
  discrepancy  = |computed_eps - reported_eps| / |reported_eps|

Thresholds:
  discrepancy > 5%  → flag (rounding/methodology difference)
  discrepancy > 15% → HIGH risk
  discrepancy > 30% → CRITICAL (likely manipulation)

Also flags:
  - Positive EPS with negative net income (impossible without manipulation)
  - Negative EPS with positive net income (same)
  - EPS sign mismatch with net income sign
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentResult


class EPSConsistencyAgent(BaseAgent):
    """
    Checks arithmetic consistency between EPS, net income, and share count.

    Required fields:
      - eps                  (income_statement)
      - net_income_loss      (income_statement)
      - shares_outstanding   (income_statement)
    """

    TOLERANCE_LOW  = 0.05   # 5% deviation — possible methodology
    TOLERANCE_HIGH = 0.15   # 15% deviation — suspect
    TOLERANCE_CRIT = 0.30   # 30% deviation — critical

    def get_name(self) -> str:
        return "eps_consistency"

    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        vals = self._extract_values(data)
        if vals is None:
            return self._create_result(
                score=0.0, confidence=0.0,
                findings=["Insufficient data: need eps, net_income_loss, shares_outstanding"],
                metrics={}, success=False, error="Missing EPS fields"
            )

        reported_eps, net_income, shares, available = vals
        findings = []
        sub_scores = []

        computed_eps = net_income / shares if shares != 0 else None

        # Check 1: Sign mismatch (impossible without manipulation)
        if computed_eps is not None:
            income_positive = net_income > 0
            eps_positive    = reported_eps > 0
            if income_positive != eps_positive and net_income != 0 and reported_eps != 0:
                sub_scores.append(90.0)
                findings.append(
                    f"CRITICAL: EPS sign ({'+' if eps_positive else '-'}) contradicts "
                    f"net income sign ({'+' if income_positive else '-'}) — "
                    f"mathematically impossible without manipulation"
                )

        # Check 2: Discrepancy magnitude
        if computed_eps is not None and reported_eps != 0:
            discrepancy = abs(computed_eps - reported_eps) / abs(reported_eps)

            if discrepancy > self.TOLERANCE_CRIT:
                score = min(85.0, 65 + discrepancy * 50)
                sub_scores.append(score)
                findings.append(
                    f"CRITICAL: EPS discrepancy of {discrepancy:.1%} — "
                    f"reported ${reported_eps:.4f} vs computed ${computed_eps:.4f} "
                    f"(net income: ${net_income:,.0f} / shares: {shares:,.0f})"
                )
            elif discrepancy > self.TOLERANCE_HIGH:
                sub_scores.append(50.0)
                findings.append(
                    f"HIGH: EPS discrepancy of {discrepancy:.1%} — "
                    f"reported ${reported_eps:.4f} vs computed ${computed_eps:.4f}"
                )
            elif discrepancy > self.TOLERANCE_LOW:
                sub_scores.append(20.0)
                findings.append(
                    f"Moderate EPS discrepancy ({discrepancy:.1%}) — may reflect diluted "
                    f"share adjustments or non-GAAP calculation (verify disclosure)"
                )
            else:
                findings.append(
                    f"EPS is arithmetically consistent: reported ${reported_eps:.4f}, "
                    f"computed ${computed_eps:.4f} (discrepancy: {discrepancy:.2%})"
                )

        elif computed_eps is None:
            findings.append("Cannot compute EPS — shares outstanding is zero or missing")

        # Check 3: Shares outstanding anomaly
        if shares > 0:
            # Flag extreme share counts
            if shares > 10_000_000_000:  # > 10 billion shares
                sub_scores.append(20.0)
                findings.append(
                    f"Unusually large share count ({shares:,.0f}) — "
                    f"possible share dilution or data error"
                )

        overall_score = max(sub_scores) if sub_scores else 0.0
        confidence = min(1.0, available / 3.0)

        metrics = {
            'reported_eps': round(reported_eps, 6),
            'computed_eps': round(computed_eps, 6) if computed_eps is not None else None,
            'net_income_loss': round(net_income, 2),
            'shares_outstanding': round(shares, 0),
            'discrepancy_pct': round(
                abs(computed_eps - reported_eps) / abs(reported_eps), 4
            ) if computed_eps is not None and reported_eps != 0 else None,
            'sign_consistent': (
                (net_income > 0) == (reported_eps > 0)
                if net_income != 0 and reported_eps != 0 else True
            )
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
            for section in ['income_statement', 'balance_sheet', 'cash_flow', 'financial_data']:
                sub = d.get(section, {})
                if isinstance(sub, dict):
                    for key in keys:
                        if key in sub and sub[key] not in (None, 'N/A', ''):
                            try:
                                return float(sub[key]), True
                            except (TypeError, ValueError):
                                pass
            return None, False

        eps,    eps_ok  = get(data, 'eps', 'income_statement_eps')
        ni,     ni_ok   = get(data, 'ni', 'net_income_loss', 'income_statement_net_income_loss')
        shares, sh_ok   = get(data, 'csho', 'shares_outstanding', 'income_statement_shares_outstanding')

        if not eps_ok or not ni_ok or not sh_ok:
            return None
        if eps is None or ni is None or shares is None:
            return None

        return eps, ni, shares, sum([eps_ok, ni_ok, sh_ok])

    def is_applicable(self, data: Dict[str, Any]) -> bool:
        return self._extract_values(data) is not None
