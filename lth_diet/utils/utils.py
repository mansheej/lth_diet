from __future__ import annotations

import dataclasses
import hashlib
import os
from enum import Enum
from typing import Any, Callable, List, Optional

import yahp as hp
from composer.core import Event, State
from composer.utils import ObjectStoreProvider, run_directory
from libcloud.storage.types import ObjectDoesNotExistError


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


def save_final(state: State, event: Event) -> bool:
    if event == Event.EPOCH_CHECKPOINT and state.timer >= state.max_duration:
        return True
    return False


def get_object_name(location: str, name: str) -> str:
    return os.path.join(os.environ["OBJECT_STORE_DIR"], location, name)


def get_local_dir(location: str) -> str:
    return os.path.join(run_directory.get_run_directory(), location)


def get_local_path(location: str, name: str) -> str:
    return os.path.join(run_directory.get_run_directory(), location, name)


def save_object(
    object_to_save: Any,
    location: str,
    name: str,
    object_store: ObjectStoreProvider,
    save_fn: Callable[[Any, str], None],
) -> None:
    # make local dir if it doesn't exist
    local_dir = get_local_dir(location)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    # save object to local path
    local_path = get_local_path(location, name)
    save_fn(object_to_save, local_path)
    # upload object to bucket
    object_name = get_object_name(location, name)
    object_store.upload_object(local_path, object_name)
    # delete local copy
    os.remove(local_path)


def load_object(location: str, name: str, object_store: ObjectStoreProvider, load_fn: Callable[[str], Any]) -> Any:
    # loads object location/name from object_store only, no local effects
    object_name = get_object_name(location, name)
    # check object exists
    if not object_exists_in_bucket(object_name, object_store):
        raise ValueError(f"Object {object_name} does not exist is object store")
    # make local location if it doesn't exist
    local_dir = get_local_dir(location)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    # download object to local path, overwrites existing if necessary
    local_path = get_local_path(location, name)
    object_store.download_object(object_name, local_path, overwrite_existing=True)
    # load object and delete local copy
    loaded_object = load_fn(local_path)
    os.remove(local_path)
    return loaded_object


def object_exists_in_bucket(object_name: str, object_store: Optional[ObjectStoreProvider]) -> bool:
    if object_store is None:
        return False
    try:
        object_store.get_object_size(object_name)
    except ObjectDoesNotExistError:
        return False
    return True
