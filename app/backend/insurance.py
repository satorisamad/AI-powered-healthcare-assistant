# Insurance verification logic

import json
import logging
from typing import Dict

INSURANCE_FILE = "data/insurance_plans.json"

def verify_insurance(provider: str) -> Dict:
    """
    Check if the insurance provider is accepted.
    """
    logging.info(f"[Insurance] Verifying provider: {provider}")
    with open(INSURANCE_FILE, "r", encoding="utf-8") as f:
        plans = json.load(f)["accepted_plans"]
    if provider in plans:
        logging.info(f"[Insurance] Provider accepted: {provider}")
        return {"status": "accepted", "provider": provider}
    logging.info(f"[Insurance] Provider not found: {provider}")
    return {"status": "not_found", "provider": provider}
