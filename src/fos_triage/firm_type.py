"""Map a published business name to a coarse firm type.

The register names the respondent business. Grouping is a transparent
gazetteer plus a few suffix rules, used only for exploratory charts. It is
not a regulatory classification and it is not a model feature: a firm
triaging its own complaints already knows who it is.
"""

from __future__ import annotations

import re

# More specific lists are tested first.
_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "funeral_plan",
        (
            "funeral", "dignity", "golden charter", "golden leaves",
        ),
    ),
    (
        "payments_or_emoney",
        (
            "revolut", "paypal", "wise ", "wise payments", "transferwise",
            "sumup", "worldpay", "checkout.com", "square", "zempler",
            "cashplus", "prepay", "soldo", "curve",
        ),
    ),
    (
        "motor_finance",
        (
            "motor finance", "moneybarn", "motonovo", "close brothers",
            "black horse", "blue motor", "advantage finance", "startline",
            "first response finance", "billing finance", "autolend",
            "oodle", "zuto", "car finance", "vehicle finance",
            "marsh finance", "rating finance", "specialist motor",
            "bmw financial", "volkswagen financial", "mercedes benz financial",
            "mitsubishi hc capital", "rci financial", "toyota financial",
            "honda financial", "ford credit", "blackhorse",
        ),
    ),
    (
        "insurer",
        (
            "insurance", "insurer", "aviva", "axa", "zurich", "allianz",
            "admiral", "direct line", "esure", "lv=", "liverpool victoria",
            "royal & sun", "royal and sun", "ageas", "covea", "nfumutual",
            "nfu mutual", "saga", "hastings", "churchill", "privilege",
            "more than", "rsa", "qbe", "hiscox", "bupa", "vitality",
            "legal & general insurance", "domestic & general",
            "homecare", "british gas insurance", "quotemehappy",
            "inter partner", "qic europe", "society of lloyd", "st andrew",
            "tesco underwriting", "amtrust", "wakam", "national house building",
            "underwriting",
        ),
    ),
    (
        "pension_or_investment",
        (
            "pension", "sipp", "annuity", "hargreaves", "aj bell", "fidelity",
            "quilter", "true potential", "st james", "sjp", "vanguard",
            "nutmeg", "wealthify", "bestinvest", "interactive investor",
            "hlam", "fundsmith", "baillie", "schroders", "aberdeen",
            "reassure", "phoenix", "standard life", "aviva wrap",
            "royal london", "prudential", "m&g", "old mutual",
            "scottish widows", "legal and general", "scottish equitable",
            "st james", "trading 212", "financial administration services",
            "ethical investors", "personal pensions", "wealth management",
        ),
    ),
    (
        "bank_or_building_society",
        (
            "bank", "banking", "building society", "barclays", "hsbc",
            "lloyds", "halifax", "natwest", "nationwide", "santander",
            "starling", "metro bank", "tsb", "virgin money", "co-operative bank",
            "cooperative bank", "tesco bank", "sainsbury's bank", "sainsburys bank",
            "first direct", "ulster bank", "bank of scotland", "coutts",
            "monzo", "chase", "atom bank", "oaknorth", "secure trust bank",
            "clydesdale", "yorkshire bank", "skipton", "coventry building",
            "leeds building", "yorkshire building", "principality",
            "j p morgan", "jp morgan", "national savings", "tesco personal finance",
            "american express",
        ),
    ),
    (
        "consumer_credit",
        (
            "loan", "lending", "credit", "newday", "capital one", "vanquis",
            "aqua", "zopa", "klarna", "clearpay", "laybuy", "oakbrook",
            "118 118", "amigo", "provident", "shorts", "lending stream",
            "drafty", "safety net", "very", "shop direct", "creation",
            "hitachi capital", "novuna", "ratesetter", "funding circle",
            "mbna", "lendable", "quickly finance", "allegiant",
        ),
    ),
    (
        "claims_manager",
        (
            "claims management", "claims ltd", "claim specialists",
            "mis-sold", "missold", "we fight any claim", "charterhouse claims",
            "money redress", "any claim",
        ),
    ),
]


def _contains(name: str, token: str) -> bool:
    token = _normalise_name(token)
    if not token or token == "monzo":
        # Monzo is a bank. It is listed with the banks below; skip any
        # accidental payments-list hit.
        return False
    return re.search(rf"\b{re.escape(token)}\b", name) is not None


def _normalise_name(business: str) -> str:
    name = business.lower().replace("&", " and ")
    name = re.sub(r"[.'’]", " ", name)
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def firm_type(business: str | None) -> str:
    if not business:
        return "unknown"
    name = _normalise_name(business)
    for label, tokens in _RULES:
        if any(_contains(name, token) for token in tokens):
            return label
    if re.search(r"\bmonzo\b", name) or re.search(r"\bstarling\b", name):
        return "bank_or_building_society"
    return "other"


FIRM_TYPE_LABELS = {
    "bank_or_building_society": "Bank or building society",
    "insurer": "Insurer",
    "motor_finance": "Motor finance",
    "consumer_credit": "Consumer credit",
    "payments_or_emoney": "Payments or e-money",
    "pension_or_investment": "Pension or investment",
    "claims_manager": "Claims manager",
    "funeral_plan": "Funeral plan",
    "other": "Other",
    "unknown": "Unknown",
}
