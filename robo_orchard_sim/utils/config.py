# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


"""Configuration class that extends Pydantic's model type.

We use Pydantic's `BaseModel` with additional annotations and methods.
This is the recommended configuration instead of using
`isaaclab.utils.configclass` because it supports Pydantic's
validation and serialization features.
"""

import collections
import dataclasses
import functools
import importlib
import inspect
import sys
import types
import typing
from dataclasses import _MISSING_TYPE
from typing import Annotated, Any, Callable, Mapping, Type

from pydantic import Field, PlainSerializer, PlainValidator, create_model
from robo_orchard_core.utils.config import (
    PYDANTIC_CONFIGCLASS,
    TYPE_LIST,
    CallableConfig as _CallableConfig,
    ClassConfig as _ClassConfig,
    ClassInitFromConfigMixin,  # noqa: F401
    ClassType as _ClassType,  # noqa: F401
    ClassType_co as _ClassType_co,  # noqa: F401
    Config,
    SliceType,
    TorchTensor,  # noqa: F401
    callable_to_string,
    from_json,
    string_to_callable,
)
from robo_orchard_core.utils.logging import LoggerManager
from typing_extensions import TypeVar

logger = LoggerManager().get_child(__name__)


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
V = TypeVar("V")

_CallableSerializer = PlainSerializer(
    lambda x: (callable_to_string(x) if x is not None else None),
    return_type=str,
    when_used="always",
    # when_used="json",
)

ClassType_co = Annotated[
    type[T_co],
    PlainValidator(
        lambda x: string_to_callable(x) if isinstance(x, str) else x
    ),
    _CallableSerializer,
]

ClassType = ClassType_co

CallableType = Annotated[
    Callable[TYPE_LIST, T],
    PlainValidator(
        lambda x: string_to_callable(x) if isinstance(x, str) else x
    ),
    _CallableSerializer,
]


class ClassConfig(_ClassConfig):
    """overwrite ClassConfig in core."""

    class_type: ClassType_co[T_co]


class CallableConfig(_CallableConfig):
    """overwrite CallableConfig in core."""

    func: CallableType[..., T_co]


@dataclasses.dataclass
class FieldDesc:
    field_type: Annotated
    default: Any
    default_factory: Callable


def _isaac_config_validate_from(data: Any) -> Any:
    if isinstance(data, str):
        data = from_json(data, allow_partial=True)
    if isinstance(data, dict):
        assert "__config_type__" in data
        data = data.copy()
        cls = string_to_callable(data.pop("__config_type__"))
        # data.pop("__config_type__")
        if hasattr(cls, "from_dict"):
            return cls.from_dict(data)
        else:
            ret = cls()
            ret.from_dict(data)
            return ret

    return data


def _isaac_config_dump_dict(data: Any) -> dict:
    ret = {}
    ret["__config_type__"] = callable_to_string(type(data))
    assert hasattr(data, "to_dict")
    ret.update(data.to_dict())
    return ret


IsaacConfigType = Annotated[
    T,
    PlainValidator(_isaac_config_validate_from),
    PlainSerializer(_isaac_config_dump_dict, when_used="always"),
]
"""Annotated type for Isaac configuration classes."""


_ConfigType = TypeVar("_ConfigType", bound=Any)
_ConfigBaseType = TypeVar("_ConfigBaseType", bound=Config)


def type_hint_parser(cls: Type[_ConfigType]) -> Type[Any]:
    if cls is collections.abc.Callable:  # type: ignore
        return CallableType  # type: ignore
    elif cls is slice:
        return SliceType  # type: ignore
    elif cls is object:
        return Any  # type: ignore

    hint_origin = typing.get_origin(cls)
    hint_args = typing.get_args(cls)

    if hint_origin is collections.abc.Callable:  # type: ignore
        if len(hint_args) > 1:
            return CallableType[..., hint_args[-1]]  # type: ignore
        else:
            return CallableType  # type: ignore
    elif dataclasses.is_dataclass(cls):
        # just convert and update field_type
        # return OmniConfigType[isaac_configclass2pydantic(cls)]
        return isaac_configclass2pydantic(cls)

    elif hint_origin in (types.UnionType,):
        new_args = (type_hint_parser(arg) for arg in hint_args)
        return typing.Union[tuple(new_args)]  # type: ignore
    elif hint_origin in (typing.List, list):
        if len(hint_args) > 0:
            return typing.List[type_hint_parser(hint_args[0])]
        else:
            return typing.List
    return cls


def isaac_configclass2pydantic(cls: Type[_ConfigType]) -> Type[_ConfigType]:
    """Converts an isaac config class to a Pydantic model.

    This function converts an isaac config class to a Pydantic model.
    It recursively converts the nested classes to Pydantic models.

    Args:
        cls: The omni config class.

    Returns:
        Type[ConfigType | Config]: The Pydantic model.

    """

    def get_globals(obj) -> dict | None:
        """Get the globals dictionary of the class object."""
        obj_globals = None
        module_name = getattr(obj, "__module__", None)
        if module_name:
            module = sys.modules.get(module_name, None)
            if module:
                obj_globals = getattr(module, "__dict__", None)
        unwrap = obj
        if unwrap is not None:
            while True:
                if hasattr(unwrap, "__wrapped__"):
                    unwrap = unwrap.__wrapped__
                    continue
                if isinstance(unwrap, functools.partial):
                    unwrap = unwrap.func
                    continue
                break
            if hasattr(unwrap, "__globals__"):
                obj_globals = unwrap.__globals__
        return obj_globals

    # handle build in class type, or other case, just use it.

    cls_name = cls.__name__
    # qualname is the fully qualified name of the class
    cls_name_full = cls.__qualname__
    module_with_name = cls.__module__ + "." + cls_name_full
    # return the class if it is already registered
    if module_with_name in PYDANTIC_CONFIGCLASS:
        return PYDANTIC_CONFIGCLASS.get(module_with_name)  # type: ignore

    obj_globals = get_globals(cls)

    # patch the module name for some classes ...
    if obj_globals and cls.__module__.startswith("isaaclab.sim.spawners"):
        obj_globals["schemas"] = importlib.import_module(
            "isaaclab.sim.schemas"
        )
        # Fix typo of FixedTendonsPropertiesCfg to FixedTendonPropertiesCfg
        anno = inspect.get_annotations(cls, eval_str=False)
        for k, v in anno.items():
            anno[k] = (
                v.replace(
                    "FixedTendonsPropertiesCfg", "FixedTendonPropertiesCfg"
                )
                if isinstance(v, str)
                else v
            )
        cls.__annotations__ = anno

        if "Usd" not in obj_globals:
            obj_globals["Usd"] = importlib.import_module("pxr.Usd")

    if obj_globals and cls.__module__.startswith("isaaclab.managers"):
        obj_globals["ManagerTermBase"] = importlib.import_module(
            "isaaclab.managers.manager_base"
        ).ManagerTermBase

    if obj_globals and cls.__module__.startswith(
        "isaaclab.sim.spawners.wrappers"
    ):
        obj_globals["Callable"] = Callable

    # assert dataclasses.is_dataclass(cls)
    logger.debug(
        f"dataclass annotations for cls: {cls} "
        f"module: {cls.__module__} "
        f"qualname: {cls.__qualname__} "
        f"annotations: {inspect.get_annotations(cls, eval_str=False, globals=obj_globals)}"  # noqa
    )
    # get all fields and default values from the class
    has_func = False
    has_class_type = False

    fields: Mapping[str, FieldDesc] = {}

    for field, field_type in inspect.get_annotations(
        cls, eval_str=True, globals=obj_globals
    ).items():
        hint_origin = typing.get_origin(field_type)
        hint_args = typing.get_args(field_type)

        # Not handle case:
        # 1. default value is always a callback in omni config class, how to
        #   convert it with converted field_type?
        field_type = type_hint_parser(field_type)
        # special case to handle `func` and `class_type` in omni config class

        if field == "func":
            has_func = True
        elif field == "class_type":
            if typing.get_origin(field_type) not in (typing.Annotated,):
                if isinstance(hint_origin, type):
                    field_type = ClassType_co[hint_args]
                else:
                    assert isinstance(field_type, type)
                    field_type = ClassType_co[field_type]
            has_class_type = True

        fields[field] = FieldDesc(
            field_type=field_type,
            default=cls.__dataclass_fields__[field].default,
            default_factory=cls.__dataclass_fields__[field].default_factory,
        )

    base_class = Config
    if has_func:
        base_class = CallableConfig[cls]
    elif has_class_type:
        base_class = ClassConfig[cls]

    # Old way to create a new class

    # class_body = {
    #     "__annotations__": {k: v[0] for k, v in fields.items()},
    # }
    # class_init_values = {k: v[1]() for k, v in fields.items()}
    # for k, v in class_init_values.items():
    #     if isinstance(v, dataclasses._MISSING_TYPE):
    #         class_init_values[k] = ...
    # class_body.update(class_init_values)
    # if has_func and has_class_type:
    #     raise ValueError(
    #         "The class cannot have both 'func' and 'class_type' fields."
    #     )
    # new_class = types.new_class(
    #     cls_name,
    #     (base_class, cls),
    #     # tuple([base_class, cls]),
    #     # tuple([cls, base_class]),
    #     # (),
    #     exec_body=lambda ns: ns.update(class_body),
    # )
    # print(f"class_body for {cls_name}", class_body)
    # model = pydantic_dataclass(
    #     kw_only=True, config=ConfigDict(arbitrary_types_allowed=True)
    # )(new_class)

    def _patch_missing_field(current_fields: dict):
        new_fields = dict()
        for k, v in current_fields.items():
            if isinstance(v.default, _MISSING_TYPE):
                if isinstance(v.default_factory, _MISSING_TYPE) or isinstance(
                    v.default_factory(), _MISSING_TYPE
                ):
                    new_fields[k] = (v.field_type, ...)
                else:
                    new_fields[k] = (
                        v.field_type,
                        Field(default_factory=v.default_factory),
                    )
            else:
                new_fields[k] = (v.field_type, Field(v.default))

        return new_fields

    new_fields = _patch_missing_field(fields)

    model = create_model(
        cls_name,
        __base__=(base_class, cls),
        **new_fields,  # type: ignore
    )

    if hasattr(cls, "validate"):
        # Override the validate method because isaac config class has a
        # different implementation of the validate method.
        # The isaac config class version of the validate method should be used.
        model.validate = cls.validate

    PYDANTIC_CONFIGCLASS.register(model, name=module_with_name)
    # find all nested class and add them to the model
    for n in PYDANTIC_CONFIGCLASS.keys():
        if (
            n.startswith(f"{module_with_name}.")
            and n.count(".") == module_with_name.count(".") + 1
        ):
            nested_cls_name = n.split(".")[-1]
            setattr(model, nested_cls_name, PYDANTIC_CONFIGCLASS.get(n))

    model.__doc__ = cls.__doc__

    # override the module name
    model.__module__ = cls.__module__
    # unload old class and reload new class
    module = sys.modules[cls.__module__]
    # nested classes are not directly accessible
    if hasattr(module, cls_name):
        delattr(module, cls_name)
        setattr(module, cls_name, model)
    # Update the class in the caller's global namespace
    frame = inspect.currentframe()
    assert frame is not None
    # get caller frame
    if frame.f_back is not None:
        frame = frame.f_back
    if cls_name in frame.f_globals:
        frame.f_globals[cls_name] = model

    return model  # type: ignore
