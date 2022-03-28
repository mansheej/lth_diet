from __future__ import annotations
from dataclasses import fields
from enum import Enum
import hashlib
from libcloud.storage.types import ObjectDoesNotExistError
from pathlib import Path
from typing import List, Optional
import yahp as hp
from composer.utils import ObjectStoreProvider


def resolve_field_name(field: int | float | bool | str | Enum | List | hp.Hparams) -> str:
    if isinstance(field, hp.Hparams):
        name = field.name if hasattr(field, "name") else get_hparams_name(field)
    elif isinstance(field, List):
        name = f"[{','.join([resolve_field_name(x) for x in field])}]"
    elif isinstance(field, Enum):
        name = resolve_field_name(field.value)
    else:
        name = str(field)
    return name


def get_hparams_name(
    hparams: hp.Hparams, prefix: Optional[str] = None, ignore_fields: Optional[List[str]] = None
) -> str:
    field_names = [field.name for field in fields(hparams)]
    if ignore_fields is not None:
        field_names = [name for name in field_names if name not in ignore_fields]
    field_names = [name for name in field_names if getattr(hparams, name) is not None]
    field_names = [f"{name}={resolve_field_name(getattr(hparams, name))}" for name in field_names]
    if prefix is None:
        prefix = type(hparams).__name__
    name = f"{prefix}({','.join(field_names)})"
    return name


def get_hash(string: str) -> str:
    return hashlib.md5(string.encode("utf-8")).hexdigest()


def object_exists_in_bucket(object_name: str, object_store: Optional[ObjectStoreProvider]) -> bool:
    if object_store is None:
        return False
    try:
        object_store.get_object_size(object_name)
    except ObjectDoesNotExistError:
        return False
    return True
