"""
Liquidity Crunch Agent
Detects dangerously low short-term liquidity.

Checks:
  1. Cash Ratio  = cash / current_liabilities  (< 0.10 → CRITICAL)
  2. Quick Ratio = (cash + receivables) / current_liabilities (< 0.20 → CRITICAL)
  3. Current ratio < 1.0 (current liabilities exceed current assets)
  4. Ending cash < cash burned this period
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentResult


class LiquidityCrunchAgent(BaseAgent):
    CASH_CRIT  = 0.10
    CASH_HIGH  = 0.20
    QUICK_CRIT = 0.20
    QUICK_HIGH = 0.50

    def get_name(self) -> str:
        return "liquidity_crunch"

    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        vals = self._extract_values(data)
        if vals is None:
            return self._create_result(score=0.0, confidence=0.0,
                findings=["Insufficient data: need cash and current liabilities"],
                metrics={}, success=False, error="Missing liquidity fields")

        (cash, current_liab, receivables, current_assets, cfo, net_change_cash, available) = vals
        if current_liab == 0:
            return self._create_result(score=0.0, confidence=0.2,
                findings=["Current liabilities unavailable"], metrics={},
                success=False, error="Zero current liabilities")

        def sr(num, denom): return num / denom if denom else None

        cash_ratio  = sr(cash, current_liab)
        quick_ratio = sr(cash + receivables, current_liab)
        current_r   = sr(current_assets, current_liab) if current_assets else None

        findings = []
        sub_scores = []

        if cash_ratio is not None:
            if cash_ratio < self.CASH_CRIT:
                sub_scores.append(min(90.0, 70 + (self.CASH_CRIT - cash_ratio) / self.CASH_CRIT * 30))
                findings.append(f"CRITICAL: Cash ratio ({cash_ratio:.2f}) — only {cash_ratio*100:.1f}¢ per $1 owed "
                                 f"(${cash:,.0f} vs ${current_liab:,.0f} current liabilities)")
            elif cash_ratio < self.CASH_HIGH:
                sub_scores.append(40.0)
                findings.append(f"Low cash ratio ({cash_ratio:.2f}) — limited immediate liquidity")

        if quick_ratio is not None:
            if quick_ratio < self.QUICK_CRIT:
                sub_scores.append(min(85.0, 65 + (self.QUICK_CRIT - quick_ratio) / self.QUICK_CRIT * 30))
                findings.append(f"CRITICAL: Quick ratio ({quick_ratio:.2f}) — cannot meet short-term obligations")
            elif quick_ratio < self.QUICK_HIGH:
                sub_scores.append(30.0)
                findings.append(f"Low quick ratio ({quick_ratio:.2f}) — tight short-term liquidity")

        if current_r is not None and current_r < 1.0:
            sub_scores.append(40.0)
            findings.append(f"Current ratio ({current_r:.2f}) < 1.0 — current liabilities exceed current assets")

        if net_change_cash is not None and net_change_cash < 0 and cash < abs(net_change_cash):
            sub_scores.append(55.0)
            findings.append(f"Ending cash (${cash:,.0f}) < cash burned this period (${abs(net_change_cash):,.0f})")

        if not findings:
            findings.append(f"Liquidity adequate: cash ratio {cash_ratio:.2f}, quick ratio "
                            f"{quick_ratio:.2f}" if cash_ratio and quick_ratio else "Liquidity appears adequate")

        metrics = {
            'cash': round(cash, 2), 'current_liabilities': round(current_liab, 2),
            'cash_ratio': round(cash_ratio, 4) if cash_ratio else None,
            'quick_ratio': round(quick_ratio, 4) if quick_ratio else None,
            'current_ratio': round(current_r, 4) if current_r else None,
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
            for sec in ['balance_sheet', 'cash_flow', 'income_statement', 'financial_data']:
                sub = d.get(sec, {})
                if isinstance(sub, dict):
                    for key in keys:
                        if key in sub and sub[key] not in (None, 'N/A', ''):
                            try:
                                return float(sub[key]), True
                            except (TypeError, ValueError):
                                pass
            return None, False

        cash,       cash_ok = get(data, 'che', 'cash', 'balance_sheet_cash')
        cur_liab,   cl_ok   = get(data, 'lct', 'accounts_payable_accrued', 'balance_sheet_accounts_payable_accrued')
        recv,       _       = get(data, 'rect', 'accounts_receivable', 'balance_sheet_accounts_receivable')
        cur_assets, _       = get(data, 'act', 'total_current_assets', 'balance_sheet_total_current_assets')
        cfo,        _       = get(data, 'net_cash_operating', 'cash_flow_net_cash_operating')
        net_chg,    _       = get(data, 'net_change_cash', 'cash_flow_net_change_cash')

        if not cash_ok or not cl_ok:
            return None
        recv       = recv       or 0.0
        cur_assets = cur_assets or 0.0
        available  = sum([cash_ok, cl_ok, (recv or 0) > 0, (cur_assets or 0) > 0,
                          cfo is not None, net_chg is not None])
        return (cash, cur_liab, recv, cur_assets, cfo, net_chg, available)

    def is_applicable(self, data: Dict[str, Any]) -> bool:
        return self._extract_values(data) is not None
