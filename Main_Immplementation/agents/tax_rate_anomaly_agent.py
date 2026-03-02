"""
Tax Rate Anomaly Agent
Detects implausible effective tax rates — a classic earnings manipulation signal.

Companies manipulate the tax line to:
  - Inflate reported profits (record phantom tax benefits)
  - Smooth earnings across periods ("cookie jar" reserves)
  - Hide losses (defer recognition)

Effective Tax Rate (ETR) = income_tax_expense / profit_loss_before_taxes

Fraud flags:
  - ETR near 0% on positive income   → income may be inflated / sheltered illegally
  - Very large tax BENEFIT on profit  → aggressive deferred tax asset creation
  - ETR > 70%                         → over-accrued tax = reserves for future manipulation
  - Tax benefit (negative expense) on profitable company → "cookie jar" reserve
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentResult


class TaxRateAnomalyAgent(BaseAgent):
    """
    Analyzes the effective tax rate for manipulation signals.

    Required fields:
      - income_tax_expense   (income_statement)
      - profit_loss_before_taxes (income_statement)
    """

    # Thresholds
    ETR_NEAR_ZERO  = 0.02   # < 2% on profitable company → suspicious
    ETR_LOW        = 0.10   # < 10% on profitable company → watch
    ETR_HIGH       = 0.50   # > 50% → over-accrued, reserve creation risk
    ETR_CRITICAL   = 0.70   # > 70% → extreme over-accrual

    def get_name(self) -> str:
        return "tax_rate_anomaly"

    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        vals = self._extract_values(data)
        if vals is None:
            return self._create_result(
                score=0.0, confidence=0.0,
                findings=["Insufficient data: need income_tax_expense and profit_loss_before_taxes"],
                metrics={}, success=False, error="Missing tax/income fields"
            )

        tax_expense, pre_tax_income, available = vals

        # Effective Tax Rate
        etr = None
        if pre_tax_income != 0:
            etr = tax_expense / pre_tax_income

        findings = []
        sub_scores = []
        confidence = min(1.0, available / 2.0)

        # Case 1: Profitable company with near-zero or negative tax
        if pre_tax_income > 0:
            if tax_expense < 0:
                sub_scores.append(75.0)
                findings.append(
                    f"CRITICAL: Profitable company (pre-tax: ${pre_tax_income:,.0f}) "
                    f"recording a TAX BENEFIT of ${abs(tax_expense):,.0f} — "
                    f"possible 'cookie jar' reserve or inflated deferred tax asset"
                )
            elif etr is not None and etr < self.ETR_NEAR_ZERO:
                sub_scores.append(65.0)
                findings.append(
                    f"Very low ETR ({etr:.1%}) on profitable operations — "
                    f"possible undisclosed tax shelter or income inflation"
                )
            elif etr is not None and etr < self.ETR_LOW:
                sub_scores.append(30.0)
                findings.append(f"Low effective tax rate ({etr:.1%}) — warrants scrutiny of tax disclosures")

        # Case 2: Loss company with large tax expense (over-accrual)
        if pre_tax_income < 0 and tax_expense > 0:
            tax_to_loss_ratio = tax_expense / abs(pre_tax_income)
            if tax_to_loss_ratio > 0.3:
                sub_scores.append(50.0)
                findings.append(
                    f"Tax expense (${tax_expense:,.0f}) recorded on a pre-tax LOSS "
                    f"(${pre_tax_income:,.0f}) — possible reserve manipulation"
                )

        # Case 3: Excessive ETR (over-accrual)
        if etr is not None and pre_tax_income > 0:
            if etr > self.ETR_CRITICAL:
                sub_scores.append(60.0)
                findings.append(
                    f"Extreme ETR ({etr:.1%}) — severe over-accrual suggests "
                    f"reserves being built for future earnings smoothing"
                )
            elif etr > self.ETR_HIGH:
                sub_scores.append(30.0)
                findings.append(f"High ETR ({etr:.1%}) — possible over-accrued tax reserves")

        # Case 4: Zero tax expense in both directions
        if tax_expense == 0 and pre_tax_income != 0:
            sub_scores.append(20.0)
            findings.append(f"Zero tax expense on non-zero pre-tax income — possible omission")

        score = max(sub_scores) if sub_scores else 0.0
        if not findings:
            etr_str = f"{etr:.1%}" if etr is not None else "N/A"
            findings.append(f"Effective tax rate ({etr_str}) is within a plausible range")

        metrics = {
            'income_tax_expense': round(tax_expense, 2),
            'pre_tax_income': round(pre_tax_income, 2),
            'effective_tax_rate': round(etr, 4) if etr is not None else None,
            'profitable': pre_tax_income > 0,
            'tax_benefit_on_profit': pre_tax_income > 0 and tax_expense < 0,
            'suspicion_level': (
                'CRITICAL' if score >= 70 else
                'HIGH' if score >= 50 else
                'ELEVATED' if score >= 25 else 'NORMAL'
            )
        }

        return self._create_result(score=score, confidence=confidence, findings=findings, metrics=metrics)

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
            return 0.0, False

        tax, tax_ok = get(data, 'txt', 'income_tax_expense', 'income_statement_income_tax_expense')
        pretax, pt_ok = get(data, 'ib', 'profit_loss_before_taxes', 'income_statement_profit_loss_before_taxes')

        if not tax_ok or not pt_ok:
            return None
        return tax, pretax, sum([tax_ok, pt_ok])

    def is_applicable(self, data: Dict[str, Any]) -> bool:
        return self._extract_values(data) is not None
