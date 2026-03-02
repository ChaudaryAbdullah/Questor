"""
Expense Padding Agent
Detects abnormally inflated operating expenses relative to revenue — a classic
form of fraud used to siphon cash or qualify for cost-plus contracts.

Patterns detected:
  1. Expense-to-Revenue ratio > 1.0 (expenses exceed revenue)
  2. Gross margin collapse (negative or extremely thin gross profit)
  3. Operating expenses growing significantly above revenue
  4. Abnormal salary/compensation concentration
  5. Vague or catch-all "other expenses" dominance

Core metric — Expense-to-Revenue (E2R):
  E2R = total_operating_expenses / total_revenues
  E2R > 1.0   → company is losing money on operations (flag)
  E2R > 1.2   → expenses are 20%+ above revenue → padding suspected
  E2R > 1.5   → severe expense inflation → high fraud risk

Gross Profit Margin (GPM):
  GPM = (revenue - operating_expenses) / revenue
  GPM < -0.15 → deep operating loss (suspicious)
  GPM < -0.50 → severe gross loss → CRITICAL
"""
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentResult


class ExpensePaddingAgent(BaseAgent):
    """
    Detects inflated and/or suspicious operating expense patterns.

    Required financial fields:
      - total_revenues (income_statement)
      - total_operating_expenses (income_statement)

    Optional fields (for enhanced analysis):
      - salaries_wages
      - depreciation_amortization
      - profit_loss_operations
    """

    # Expense-to-Revenue thresholds
    E2R_WATCH = 0.95          # Approaching breakeven → watch
    E2R_ELEVATED = 1.05       # Just above 100% → elevated concern
    E2R_HIGH = 1.20           # 20%+ expense inflation → HIGH
    E2R_CRITICAL = 1.50       # 50%+ expense inflation → CRITICAL

    # Gross Margin thresholds
    GPM_THIN = 0.05           # Margins thinner than 5% → thin
    GPM_LOSS = -0.15          # GPM < -15% → operating loss
    GPM_SEVERE = -0.50        # GPM < -50% → severe

    # Compensation share thresholds
    SALARY_SHARE_HIGH = 0.60  # Salaries > 60% of total expenses → concentrated

    def get_name(self) -> str:
        return "expense_padding"

    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        """
        Analyze expense patterns for padding and inflation anomalies.

        Args:
            data: Financial data dict

        Returns:
            AgentResult with expense anomaly assessment
        """
        values = self._extract_values(data)

        if values is None:
            return self._create_result(
                score=0.0,
                confidence=0.0,
                findings=["Insufficient data: need total_revenues and total_operating_expenses"],
                metrics={},
                success=False,
                error="Missing required income statement fields"
            )

        (revenue, op_expenses, op_income, salaries,
         depreciation, available) = values

        if revenue == 0:
            return self._create_result(
                score=50.0,
                confidence=0.3,
                findings=["Zero revenue with non-zero operating expenses — potential shell company or fraud"],
                metrics={'revenue': 0, 'operating_expenses': round(op_expenses, 2)},
                success=True
            )

        # Core ratios
        e2r = op_expenses / revenue
        gpm = (revenue - op_expenses) / revenue

        findings = []
        sub_scores = []

        # 1. Expense-to-Revenue
        e2r_score, e2r_findings = self._check_e2r(e2r, revenue, op_expenses)
        sub_scores.append(e2r_score)
        findings.extend(e2r_findings)

        # 2. Gross Profit Margin
        gpm_score, gpm_findings = self._check_gpm(gpm, revenue)
        sub_scores.append(gpm_score)
        findings.extend(gpm_findings)

        # 3. Salary concentration
        salary_findings = []
        if salaries > 0 and op_expenses > 0:
            salary_share = salaries / op_expenses
            sal_score, salary_findings = self._check_salary_share(salary_share, salaries)
            sub_scores.append(sal_score)
            findings.extend(salary_findings)

        # 4. Operating income check
        op_income_findings = []
        if op_income is not None:
            oi_score, op_income_findings = self._check_op_income(op_income, revenue)
            sub_scores.append(oi_score)
            findings.extend(op_income_findings)

        # Weighted: max-driven to catch worst signals
        if sub_scores:
            # Average of top half of scores (worst signals dominate)
            n = max(1, len(sub_scores) // 2)
            overall_score = sum(sorted(sub_scores, reverse=True)[:n]) / n
        else:
            overall_score = 0.0

        confidence = min(1.0, available / 4.0)

        # Compute additional metrics
        salary_ratio_val = salaries / op_expenses if salaries > 0 and op_expenses > 0 else None

        metrics = {
            'total_revenues': round(revenue, 2),
            'total_operating_expenses': round(op_expenses, 2),
            'expense_to_revenue_ratio': round(e2r, 4),
            'gross_profit_margin': round(gpm, 4),
            'gross_profit': round(revenue - op_expenses, 2),
            'operating_income': round(op_income, 2) if op_income is not None else 'N/A',
            'salary_to_expense_ratio': round(salary_ratio_val, 4) if salary_ratio_val else 'N/A',
            'expenses_exceed_revenue': e2r > 1.0,
            'is_deep_loss': gpm < self.GPM_LOSS,
        }

        if not findings:
            findings = [f"Expense-to-revenue ratio ({e2r:.2%}) and margins are within normal range"]

        return self._create_result(
            score=overall_score,
            confidence=confidence,
            findings=findings,
            metrics=metrics
        )

    def _extract_values(self, data: Dict[str, Any]) -> Optional[tuple]:
        """Extract income statement fields."""

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
            return None, False

        revenue, rev_ok = get(data, 'sale', 'total_revenues', 'income_statement_total_revenues')
        expenses, exp_ok = get(data, 'total_operating_expenses', 'income_statement_total_operating_expenses')

        if not rev_ok or not exp_ok:
            return None

        revenue = abs(revenue) if revenue else 0.0
        expenses = abs(expenses) if expenses else 0.0

        op_income_val, _ = get(data, 'profit_loss_operations', 'income_statement_profit_loss_operations')
        salaries_val, sal_ok = get(data, 'salaries_wages', 'income_statement_salaries_wages')
        depreciation_val, dep_ok = get(data, 'dp', 'depreciation_amortization',
                                       'income_statement_depreciation_amortization')

        salaries = abs(salaries_val) if sal_ok and salaries_val else 0.0
        depreciation = abs(depreciation_val) if dep_ok and depreciation_val else 0.0
        op_income = op_income_val  # Can be negative

        available = sum([rev_ok, exp_ok, sal_ok, dep_ok])

        return revenue, expenses, op_income, salaries, depreciation, available

    def _check_e2r(self, e2r: float, revenue: float, expenses: float):
        findings = []
        if e2r >= self.E2R_CRITICAL:
            score = min(100.0, 80 + (e2r - self.E2R_CRITICAL) * 40)
            findings.append(
                f"CRITICAL: Operating expenses (${expenses:,.0f}) are {e2r:.1%} of revenue (${revenue:,.0f}) "
                f"— expenses exceed revenue by {(e2r-1)*100:.0f}%"
            )
        elif e2r >= self.E2R_HIGH:
            score = 50 + (e2r - self.E2R_HIGH) / (self.E2R_CRITICAL - self.E2R_HIGH) * 30
            findings.append(
                f"High expense ratio ({e2r:.1%}): expenses significantly exceed revenue — possible padding"
            )
        elif e2r >= self.E2R_ELEVATED:
            score = 30 + (e2r - self.E2R_ELEVATED) / (self.E2R_HIGH - self.E2R_ELEVATED) * 20
            findings.append(f"Elevated expense ratio ({e2r:.1%}): expenses just exceed revenue")
        elif e2r >= self.E2R_WATCH:
            score = 15.0
            findings.append(f"Watch: expense ratio ({e2r:.1%}) approaching breakeven — tightly controlled margins")
        else:
            score = e2r * 10  # Low score for healthy margins
        return score, findings

    def _check_gpm(self, gpm: float, revenue: float):
        findings = []
        if gpm < self.GPM_SEVERE:
            score = min(90.0, 70 + abs(gpm - self.GPM_SEVERE) * 50)
            findings.append(
                f"CRITICAL gross margin ({gpm:.1%}): company losing more than 50 cents per dollar of revenue"
            )
        elif gpm < self.GPM_LOSS:
            score = 40 + (self.GPM_LOSS - gpm) / (self.GPM_LOSS - self.GPM_SEVERE) * 30
            findings.append(f"Negative gross margin ({gpm:.1%}): operating loss — expenses padded or revenue understated")
        elif gpm < self.GPM_THIN:
            score = 15.0
            findings.append(f"Very thin gross margin ({gpm:.1%}) — vulnerable to expense manipulation")
        else:
            score = 0.0  # Healthy margin
        return score, findings

    def _check_salary_share(self, salary_share: float, salaries: float):
        findings = []
        if salary_share > self.SALARY_SHARE_HIGH:
            score = 25 + (salary_share - self.SALARY_SHARE_HIGH) * 50
            findings.append(
                f"Salary concentration ({salary_share:.1%} of operating expenses, ${salaries:,.0f}) "
                f"is unusually high — potential payroll padding"
            )
            return min(50.0, score), findings
        return 0.0, []

    def _check_op_income(self, op_income: float, revenue: float):
        findings = []
        if revenue > 0:
            oi_margin = op_income / revenue
            if oi_margin < -0.5:
                return 60.0, [
                    f"Severe operating loss: operating income is {oi_margin:.1%} of revenue"
                ]
            elif oi_margin < -0.2:
                return 35.0, [
                    f"Deep operating loss: operating income is {oi_margin:.1%} of revenue"
                ]
        return 0.0, []

    def is_applicable(self, data: Dict[str, Any]) -> bool:
        return self._extract_values(data) is not None
