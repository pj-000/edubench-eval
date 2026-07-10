"""Small JSON-schema validator fallback for the Exp27 teacher API schemas.

The project normally uses ``jsonschema``. This fallback covers only the schema
keywords used by the locked Exp27 blind/audit schemas so local API annotation
does not require mutating the system Python environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ValidationError:
    message: str


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def _validate(value: Any, schema: dict[str, Any], path: str) -> Iterable[ValidationError]:
    expected = schema.get("type")
    if expected is not None:
        expected_types = [expected] if isinstance(expected, str) else list(expected)
        if not any(_matches_type(value, item) for item in expected_types):
            yield ValidationError(f"{path} is not of type {expected_types}")
            return

    if "enum" in schema and value not in schema["enum"]:
        yield ValidationError(f"{path} is not one of {schema['enum']}")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                yield ValidationError(f"{path}.{key} is required")
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                yield ValidationError(f"{path} has unexpected properties {extras}")
        for key, item in value.items():
            if key in properties:
                yield from _validate(item, properties[key], f"{path}.{key}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            yield ValidationError(f"{path} has fewer than {schema['minItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                yield from _validate(item, item_schema, f"{path}[{idx}]")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            yield ValidationError(f"{path} is below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            yield ValidationError(f"{path} is above maximum {schema['maximum']}")


class Draft202012Validator:
    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema

    def iter_errors(self, value: Any) -> Iterable[ValidationError]:
        return _validate(value, self.schema, "$")
