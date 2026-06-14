import json
from pathlib import Path

from jsonschema import Draft7Validator


# Repository root:
# scripts/01_setup/validate_schema.py -> repository root
ROOT = Path(__file__).resolve().parents[2]

SCHEMA_PATH = ROOT / "schemas" / "rules_v3.json"


def main() -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")

    with SCHEMA_PATH.open("r", encoding="utf-8") as file:
        schema = json.load(file)

    Draft7Validator.check_schema(schema)

    print(f"Valid JSON Schema: {SCHEMA_PATH.name}")


if __name__ == "__main__":
    main()
