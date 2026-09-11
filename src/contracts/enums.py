from enum import Enum


class DecisionStatus(str, Enum):
    BEST = "BEST"
    ACCEPT = "ACCEPT"
    CAUTION = "CAUTION"
    REJECT = "REJECT"
    NO_BET = "NO_BET"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class IntegrityStatus(str, Enum):
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    EXCLUDED = "EXCLUDED"
