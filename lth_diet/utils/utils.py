from __future__ import annotations
import dataclasses
from enum import Enum
import hashlib
from typing import Any, List, Optional
import yahp as hp


def field_val_to_str(field: bool | float | int | str | Enum | List | hp.Hparams) -> str:
    # stringify any field type
    if isinstance(field, hp.Hparams):  # get hparam name
        name = getattr(field, "name", get_hparams_name(field))
    elif isinstance(field, List):  # [stringify,name,of,each,item]
        name = f"[{','.join([field_val_to_str(item) for item in field])}]"
    elif isinstance(field, Enum):  # stringify enum value
        name = field_val_to_str(field.value)
    else:  # stringify name
        name = str(field)
    return name


def get_hparams_name(hparams: hp.Hparams, prefix: Optional[str] = None, ignore_fields: List[str] = []) -> str:
    # Hparams name = prefix(hparams,fields,not,in,ignore_fields,), prefix is type name if unspecified
    field_names = [field.name for field in dataclasses.fields(hparams)]  # field names from dataclass
    field_names = [name for name in field_names if name not in ignore_fields]  # filter ignore_fields
    field_names = [name for name in field_names if getattr(hparams, name) is not None]  # filter Nones
    field_names = [f"{name}={field_val_to_str(getattr(hparams, name))}" for name in field_names]  # make name strings
    if prefix is None:
        prefix = type(hparams).__name__  # set default prefix
    name = f"{prefix}({','.join(field_names)})"  # stringify name
    return name


def get_hash(string: str) -> str:
    return hashlib.md5(string.encode("utf-8")).hexdigest()


def maybe_set_default(value: Optional[Any], default: Any) -> Any:
    return default if value is None else value
