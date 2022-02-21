from dataclasses import fields
from enum import Enum
from typing import List, Union
import yahp as hp


HparamField = Union[int, float, bool, str, Enum, List, hp.Hparams]


def resolve_field_name(field: HparamField) -> str:
    if isinstance(field, hp.Hparams):
        if hasattr(field, "name"):
            name = field.name
        else:
            name = get_hparams_name(field, type(field).__name__, [])
    elif isinstance(field, List):
        name = f"[{','.join([resolve_field_name(x) for x in field])}]"
    else:
        name = str(field)
    return name


def get_hparams_name(hparams: hp.Hparams, prefix: str, ignore_fields: List[str]) -> str:
    field_names = [field.name for field in fields(hparams)]
    field_names = [f for f in field_names if f not in ignore_fields]
    field_names = [f for f in field_names if getattr(hparams, f) is not None]
    field_names = [
        f"{f}={resolve_field_name(getattr(hparams, f))}" for f in field_names
    ]
    name = f"{prefix}({','.join(field_names)})"
    return name
