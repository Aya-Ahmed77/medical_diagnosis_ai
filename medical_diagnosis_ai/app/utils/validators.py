"""Small, dependency-free request validation helpers for the Flask APIs."""
from typing import Any, Dict, List, Tuple


class ValidationError(ValueError):
    pass


def require_fields(payload: Dict[str, Any], fields: List[str]) -> None:
    if payload is None:
        raise ValidationError("Request body must be JSON.")
    missing = [f for f in fields if f not in payload or payload[f] in (None, "")]
    if missing:
        raise ValidationError(f"Missing required field(s): {', '.join(missing)}")


def optional_int(payload: Dict[str, Any], field: str) -> Tuple[bool, Any]:
    if field not in payload or payload[field] in (None, ""):
        return True, None
    try:
        return True, int(payload[field])
    except (TypeError, ValueError):
        return False, None
