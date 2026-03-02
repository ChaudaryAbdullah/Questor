"""
Altman Z-Score Agent
Predicts financial distress and bankruptcy probability using the Altman Z-Score model.

The Z-Score uses 5 financial ratios to predict financial distress:
  Z > 2.99  → Safe Zone   (low fraud risk)
  1.81–2.99 → Grey Zone   (moderate risk)
  Z < 1.81  → Distress Zone (high fraud risk)

Formula (public companies):
  Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
"""
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentResult


class AltmanZScoreAgent(BaseAgent):
    """
    Calculates the Altman Z-Score to detect financial distress,
    which is a strong predictor of fraud and earnings manipulation.

    Required financial fields (from balance_sheet / income_statement):
      - total_assets
      - total_current_assets (for working capital)
      - accounts_payable_accrued (current liabilities proxy)
      - retained_earnings
      - profit_loss_operations (EBIT)
      - total_liabilities
      - total_shareholders_equity
      - total_revenues
    """

    # Z-Score thresholds (Altman 1968 for public companies)
    SAFE_ZONE = 2.99
    GREY_ZONE = 1.81

    # Coefficients
    COEFF = {
        'X1': 1.2,   # Working Capital / Total Assets
        'X2': 1.4,   # Retained Earnings / Total Assets
        'X3': 3.3,   # EBIT / Total Assets
        'X4': 0.6,   # Market Value of Equity / Total Liabilities (approx as Book Equity / Total Liabilities)
        'X5': 1.0,   # Revenue / Total Assets
    }

    def get_name(self) -> str:
        return "altman_zscore"

    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        """
        Calculate Altman Z-Score from financial data.

        Args:
            data: Financial data dict with balance_sheet, income_statement, cash_flow sections,
                  or a flat dict of pre-extracted features.

        Returns:
            AgentResult with Z-Score and risk assessment
        """
        ratios = self._extract_ratios(data)

        if ratios is None:
            return self._create_result(
                score=0.0,
                confidence=0.0,
                findings=["Insufficient financial data to calculate Altman Z-Score"],
                metrics={},
                success=False,
                error="Missing required financial fields"
            )

        x1, x2, x3, x4, x5, available = ratios

        # Calculate Z-Score
        z_score = (
            self.COEFF['X1'] * x1 +
            self.COEFF['X2'] * x2 +
            self.COEFF['X3'] * x3 +
            self.COEFF['X4'] * x4 +
            self.COEFF['X5'] * x5
        )

        # Convert Z-Score to risk score (0–100)
        score = self._zscore_to_risk(z_score)

        # Confidence based on how many ratios were fully available
        confidence = available / 5.0

        findings = self._generate_findings(z_score, x1, x2, x3, x4, x5)

        metrics = {
            'z_score': round(z_score, 3),
            'safe_zone_threshold': self.SAFE_ZONE,
            'grey_zone_threshold': self.GREY_ZONE,
            'zone': self._zone_label(z_score),
            'components': {
                'X1_working_capital_to_assets': round(x1, 4),
                'X2_retained_earnings_to_assets': round(x2, 4),
                'X3_ebit_to_assets': round(x3, 4),
                'X4_equity_to_liabilities': round(x4, 4),
                'X5_revenue_to_assets': round(x5, 4),
            },
            'ratios_available': available
        }

        return self._create_result(
            score=score,
            confidence=confidence,
            findings=findings,
            metrics=metrics
        )

    def _extract_ratios(self, data: Dict[str, Any]) -> Optional[tuple]:
        """Extract the 5 Z-Score components from available data."""

        def get(d, *keys, default=0.0):
            """Try multiple keys across nested sections."""
            # Try flat key first
            for key in keys:
                if key in d and d[key] not in (None, 'N/A', ''):
                    try:
                        return float(d[key]), True
                    except (TypeError, ValueError):
                        pass
            # Try nested sections
            for section in ['balance_sheet', 'income_statement', 'cash_flow', 'financial_data']:
                sub = d.get(section, {})
                if isinstance(sub, dict):
                    for key in keys:
                        if key in sub and sub[key] not in (None, 'N/A', ''):
                            try:
                                return float(sub[key]), True
                            except (TypeError, ValueError):
                                pass
            return default, False

        def safe_div(num, denom):
            if denom == 0 or denom is None:
                return 0.0
            return num / denom

        total_assets, ta_ok = get(data, 'at', 'total_assets', 'balance_sheet_total_assets')
        if not ta_ok or total_assets == 0:
            return None

        current_assets, _ = get(data, 'act', 'total_current_assets', 'balance_sheet_total_current_assets')
        current_liab, _ = get(data, 'lct', 'accounts_payable_accrued', 'balance_sheet_accounts_payable_accrued')
        retained_earnings, re_ok = get(data, 're', 'retained_earnings', 'balance_sheet_retained_earnings')
        ebit, ebit_ok = get(data, 'EBIT', 'profit_loss_operations', 'income_statement_profit_loss_operations')
        total_liabilities, tl_ok = get(data, 'lt', 'total_liabilities', 'balance_sheet_total_liabilities')
        equity, eq_ok = get(data, 'ceq', 'total_shareholders_equity', 'balance_sheet_total_shareholders_equity')
        revenue, rev_ok = get(data, 'sale', 'total_revenues', 'income_statement_total_revenues')

        working_capital = current_assets - current_liab

        x1 = safe_div(working_capital, total_assets)
        x2 = safe_div(retained_earnings, total_assets)
        x3 = safe_div(ebit, total_assets)
        x4 = safe_div(equity, total_liabilities) if total_liabilities != 0 else 1.0
        x5 = safe_div(revenue, total_assets)

        available = sum([re_ok, ebit_ok, tl_ok, eq_ok, rev_ok])
        if available < 2:
            return None

        return x1, x2, x3, x4, x5, available

    def _zscore_to_risk(self, z: float) -> float:
        """Convert Z-Score to 0–100 risk score (higher = more risky)."""
        if z >= self.SAFE_ZONE:
            # Safe: risk score 0–20
            return max(0.0, 20.0 - (z - self.SAFE_ZONE) * 5)
        elif z >= self.GREY_ZONE:
            # Grey zone: risk score 20–60
            ratio = (self.SAFE_ZONE - z) / (self.SAFE_ZONE - self.GREY_ZONE)
            return 20.0 + ratio * 40.0
        else:
            # Distress zone: risk score 60–100
            below = self.GREY_ZONE - z
            return min(100.0, 60.0 + below * 15)

    def _zone_label(self, z: float) -> str:
        if z >= self.SAFE_ZONE:
            return "SAFE"
        elif z >= self.GREY_ZONE:
            return "GREY"
        else:
            return "DISTRESS"

    def _generate_findings(self, z, x1, x2, x3, x4, x5) -> List[str]:
        findings = []
        zone = self._zone_label(z)

        if zone == "DISTRESS":
            findings.append(f"Z-Score ({z:.2f}) is in DISTRESS zone (< {self.GREY_ZONE}) — high bankruptcy/fraud risk")
        elif zone == "GREY":
            findings.append(f"Z-Score ({z:.2f}) is in GREY zone ({self.GREY_ZONE}–{self.SAFE_ZONE}) — elevated risk")
        else:
            findings.append(f"Z-Score ({z:.2f}) is in SAFE zone (> {self.SAFE_ZONE}) — low distress risk")

        if x1 < 0:
            findings.append("Negative working capital: current liabilities exceed current assets")
        if x2 < 0:
            findings.append("Negative retained earnings: accumulated deficit detected")
        if x3 < 0:
            findings.append("Negative EBIT: company is operating at a loss")
        if x4 < 0.5:
            findings.append("Low equity-to-liabilities ratio: highly leveraged balance sheet")
        if x5 < 0.3:
            findings.append("Very low asset turnover: assets not generating sufficient revenue")

        return findings if findings else ["Z-Score within normal range"]

    def is_applicable(self, data: Dict[str, Any]) -> bool:
        """Check if key financial fields are present."""
        result = self._extract_ratios(data)
        return result is not None
