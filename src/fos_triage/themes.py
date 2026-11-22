"""Keyword flags for topical issues a UK complaints team is watching.

These are monitoring tags, not supervised labels. A decision can match more
than one theme. Search-hit charts in the EDA use the register's own keyword
counts; these patterns are applied only to the intake text.
"""

from __future__ import annotations

import re

THEME_PATTERNS: dict[str, re.Pattern[str]] = {
    "motor_finance_commission": re.compile(
        r"(?i)\bmotor finance\b|\bcar finance\b|\bdiscretionary commission\b|\bcommission\b.{0,40}\bfinance\b"
    ),
    "app_scam": re.compile(
        r"(?i)authorised push payment|\bapp scam\b|\bpush payment\b|\bimpersonat\w*\b|\bsafe account\b"
    ),
    "packaged_account": re.compile(r"(?i)packaged (?:bank )?account"),
    "section_75": re.compile(r"(?i)section 75|\bs75\b"),
    "irresponsible_lending": re.compile(
        r"(?i)irresponsible lending|unaffordab|affordability|creditworthiness"
    ),
    "fraud_refund": re.compile(
        r"(?i)\bscam\b|\bfraud\b|\bphish\w*\b"
    ),
}


def tag_themes(text: str) -> list[str]:
    if not text:
        return []
    return [name for name, pattern in THEME_PATTERNS.items() if pattern.search(text)]
