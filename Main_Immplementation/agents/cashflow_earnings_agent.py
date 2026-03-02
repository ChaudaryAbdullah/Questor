"""
Cash Flow vs Earnings Consistency Agent
Detects earnings manipulation by comparing accrual-based net income to cash-based
operating cash flow.

Core principle: Legitimate earnings are backed by cash. If net income is high but
operating cash flow is low or negative, the company may be using aggressive accruals
to inflate reported earnings.

Key metric — Accrual Ratio:
  accrual_ratio = (net_income - operating_cash_flow) / total_assets
  |accrual_ratio| > 0.1 → suspicious
  |accrual_ratio| > 0.2 → high risk

Also checks: Quality of Earnings (QoE) = operating_cash_flow / net_income
  QoE < 1.0 means cash doesn't back up reported income
"""
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentResult


class CashFlowEarningsAgent(BaseAgent):
    """
    Detects earnings quality issues by comparing net income to operating cash flow.

    Required financial fields:
      - net_income_loss (income_statement)
      - net_cash_operating (cash_flow)
      - total_assets (balance_sheet)
    """

    # Thresholds for the accrual ratio
    LOW_SUSPICION_THRESHOLD = 0.05    # |accrual| > 5% of assets → flag
    HIGH_SUSPICION_THRESHOLD = 0.10   # |accrual| > 10% of assets → high risk
    CRITICAL_THRESHOLD = 0.20         # |accrual| > 20% of assets → critical

    def get_name(self) -> str:
        return "cashflow_earnings"

    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        """
        Compare operating cash flow to net income to assess earnings quality.

        Args:
            data: Financial data dict

        Returns:
            AgentResult with accrual-based risk assessment
        """
        values = self._extract_values(data)

        if values is None:
            return self._create_result(
                score=0.0,
                confidence=0.0,
                findings=["Insufficient data: need net_income, net_cash_operating, and total_assets"],
                metrics={},
                success=False,
                error="Missing required cash flow / earnings fields"
            )

        net_income, cfo, total_assets, available = values

        # Accrual ratio: the core fraud signal
        accruals = net_income - cfo
        accrual_ratio = accruals / total_assets if total_assets != 0 else 0.0

        # Quality of Earnings (only meaningful when net_income > 0)
        if net_income > 0:
            qoe = cfo / net_income
        elif net_income < 0 and cfo < 0:
            qoe = 1.0  # Both negative — consistent (not manipulation)
        else:
            qoe = 0.0  # net_income <= 0 but cfo > 0 or vice versa

        score = self._accrual_to_risk(accrual_ratio, net_income, cfo)
        confidence = min(1.0, available / 3.0)
        findings = self._generate_findings(net_income, cfo, accrual_ratio, qoe)

        metrics = {
            'net_income': round(net_income, 2),
            'operating_cash_flow': round(cfo, 2),
            'accruals': round(accruals, 2),
            'accrual_ratio': round(accrual_ratio, 4),
            'quality_of_earnings': round(qoe, 4) if qoe != 0 else 'N/A',
            'total_assets': round(total_assets, 2),
            'cash_backing_income': cfo > 0 and net_income > 0,
            'income_cfo_divergence': abs(net_income - cfo) > 0.1 * total_assets,
        }

        return self._create_result(
            score=score,
            confidence=confidence,
            findings=findings,
            metrics=metrics
        )

    def _extract_values(self, data: Dict[str, Any]) -> Optional[tuple]:
        """Extract net income, operating cash flow, and total assets."""

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

        net_income, ni_ok = get(data, 'ni', 'net_income_loss', 'income_statement_net_income_loss', 'net_profit_loss')
        cfo, cf_ok = get(data, 'net_cash_operating', 'cash_flow_net_cash_operating')
        total_assets, ta_ok = get(data, 'at', 'total_assets', 'balance_sheet_total_assets')

        available = sum([ni_ok, cf_ok, ta_ok])
        if not ni_ok or not cf_ok:
            return None
        if total_assets == 0:
            total_assets = 1.0  # Avoid division by zero, low confidence anyway

        return net_income, cfo, total_assets, available

    def _accrual_to_risk(self, accrual_ratio: float, net_income: float, cfo: float) -> float:
        """Convert accrual ratio to 0–100 risk score."""
        abs_ratio = abs(accrual_ratio)

        # Base accrual score
        if abs_ratio <= self.LOW_SUSPICION_THRESHOLD:
            accrual_score = abs_ratio / self.LOW_SUSPICION_THRESHOLD * 20  # 0–20
        elif abs_ratio <= self.HIGH_SUSPICION_THRESHOLD:
            ratio = (abs_ratio - self.LOW_SUSPICION_THRESHOLD) / (
                self.HIGH_SUSPICION_THRESHOLD - self.LOW_SUSPICION_THRESHOLD
            )
            accrual_score = 20 + ratio * 30  # 20–50
        elif abs_ratio <= self.CRITICAL_THRESHOLD:
            ratio = (abs_ratio - self.HIGH_SUSPICION_THRESHOLD) / (
                self.CRITICAL_THRESHOLD - self.HIGH_SUSPICION_THRESHOLD
            )
            accrual_score = 50 + ratio * 30  # 50–80
        else:
            accrual_score = min(100.0, 80 + (abs_ratio - self.CRITICAL_THRESHOLD) * 100)

        # Bonus risk: positive income but negative cash flow (classic manipulation)
        bonus = 0.0
        if net_income > 0 and cfo < 0:
            bonus = 20.0  # Earnings without cash — very suspicious
        elif net_income > 0 and 0 < cfo < net_income * 0.3:
            bonus = 10.0  # Cash far below income

        return min(100.0, accrual_score + bonus)

    def _generate_findings(
        self,
        net_income: float,
        cfo: float,
        accrual_ratio: float,
        qoe: float
    ) -> List[str]:
        findings = []

        # Primary signal
        if net_income > 0 and cfo < 0:
            findings.append(
                f"CRITICAL: Positive net income (${net_income:,.0f}) but NEGATIVE operating cash flow "
                f"(${cfo:,.0f}) — earnings not backed by cash"
            )
        elif net_income > 0 and cfo < net_income * 0.3:
            findings.append(
                f"Operating cash flow (${cfo:,.0f}) is less than 30% of net income (${net_income:,.0f}) "
                f"— possible aggressive revenue recognition"
            )
        elif net_income < 0 and cfo > 0:
            findings.append(
                f"Positive operating cash flow (${cfo:,.0f}) despite net loss (${net_income:,.0f}) "
                f"— possible write-down or one-time charges (low manipulation risk)"
            )
        else:
            findings.append(
                f"Net income (${net_income:,.0f}) and operating cash flow (${cfo:,.0f}) are broadly consistent"
            )

        # Accrual ratio assessment
        abs_ratio = abs(accrual_ratio)
        if abs_ratio > self.CRITICAL_THRESHOLD:
            findings.append(
                f"CRITICAL accrual ratio ({accrual_ratio:.3f}): accruals are {abs_ratio*100:.1f}% of total assets"
            )
        elif abs_ratio > self.HIGH_SUSPICION_THRESHOLD:
            findings.append(
                f"High accrual ratio ({accrual_ratio:.3f}): accruals are {abs_ratio*100:.1f}% of total assets"
            )
        elif abs_ratio > self.LOW_SUSPICION_THRESHOLD:
            findings.append(
                f"Moderate accrual ratio ({accrual_ratio:.3f}): monitor for deterioration"
            )

        # Quality of Earnings
        if isinstance(qoe, float):
            if qoe < 0.5 and net_income > 0:
                findings.append(
                    f"Quality of Earnings ratio ({qoe:.2f}) is very low — less than 50% of income is cash-backed"
                )
            elif qoe > 2.0:
                findings.append(
                    f"Quality of Earnings ratio ({qoe:.2f}) is high — cash flow exceeds reported income (positive)"
                )

        return findings if findings else ["Cash flow and earnings are consistent — no manipulation signals detected"]

    def is_applicable(self, data: Dict[str, Any]) -> bool:
        """Applicable if we can extract net income and operating cash flow."""
        result = self._extract_values(data)
        return result is not None
