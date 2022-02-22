from dataclasses import fields
from enum import Enum
import hashlib
from typing import List, Union
import yahp as hp


HparamField = Union[int, float, bool, str, Enum, List, hp.Hparams]


def resolve_field_name(field: HparamField) -> str:
    if isinstance(field, hp.Hparams):
        name = (
            field.name
            if hasattr(field, "name")
            else get_hparams_name(field, type(field).__name__, [])
        )
    elif isinstance(field, List):
        name = f"[{','.join([resolve_field_name(x) for x in field])}]"
    else:
        name = str(field)
    return name


def get_hparams_name(hparams: hp.Hparams, prefix: str, ignore_fields: List[str]) -> str:
    field_names = [field.name for field in fields(hparams)]
    # filter out ignore_fields
    field_names = [f for f in field_names if f not in ignore_fields]
    # filter out any fields that are None
    field_names = [f for f in field_names if getattr(hparams, f) is not None]
    field_names = [
        f"{f}={resolve_field_name(getattr(hparams, f))}" for f in field_names
    ]
    name = f"{prefix}({','.join(field_names)})"
    return name


def get_hash(string: str) -> str:
    return hashlib.md5(string.encode("utf-8")).hexdigest()
