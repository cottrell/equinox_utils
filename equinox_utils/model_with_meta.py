import inspect
import json
import os
from dataclasses import dataclass
from functools import wraps
from typing import Dict

import equinox as eqx

from .recurse_get_state import get_object_from_module_and_qualname
from .serialization import flavours as _serialization_flavours
from .util import check_identical

_MODEL_FILENAME = 'model.eqx'
_META_FILENAME = 'meta.json'
_SERIALIZE_META_FILENAME = 'serialize_meta.json'


@dataclass
class ModelWithMeta:
    """A holder for equinox model (model) and meta data dictionary (meta).

    NOTE: currently not frozen so you are able to do model.model = train(model.model, ...).

    The (module, qualname, meta) is basically the (func, arg) spec of the maker.
    """

    module: str
    qualname: str
    meta: Dict
    model: eqx.Module

    def __init__(self, default_equinox_serialization_flavour='tree_serialize_leaves', *, meta, model, module, qualname):
        self.meta = meta
        self.model = model
        self.module = module
        self.qualname = qualname
        self.default_equinox_serialization_flavour = default_equinox_serialization_flavour

    def save(self, path, flavour=None):
        os.makedirs(path, exist_ok=True)
        self._save_meta(os.path.join(path, 'meta.json'))
        flavour = self._save_model(os.path.join(path, 'model.eqx'), flavour=flavour)
        serialize_meta = dict(
            serialization_flavour=flavour,
            module=self.module,
            qualname=self.qualname,
        )
        open(os.path.join(path, 'serialize_meta.json'), 'w').write(json.dumps(serialize_meta))

    def _save_model(self, path, *, flavour):
        flavour = flavour or self.default_equinox_serialization_flavour
        writer = _serialization_flavours[flavour]['write']
        writer(self.model, path)
        return flavour

    def _save_meta(self, path):
        json.dump(self.meta, open(path, 'w'))

    @classmethod
    def load(cls, path):
        meta = json.load(open(os.path.join(path, 'meta.json')))
        serialize_meta = json.load(open(os.path.join(path, 'serialize_meta.json')))
        flavour = json.load(open(os.path.join(path, 'serialize_meta.json')))['serialization_flavour']
        reader = _serialization_flavours[flavour]['read']
        path = os.path.join(path, 'model.eqx')
        module = serialize_meta['module']
        qualname = serialize_meta['qualname']
        if flavour == 'tree_serialize_leaves':
            maker_fun = get_object_from_module_and_qualname(module, qualname)
            model = maker_fun(**meta)  # NOTE: remember this returns a model with meta
            reader(path, model=model.model)
            return model
        elif flavour == 'orbax':
            # WARNING: this is experimental and I do not think will not work in general as no recursion on eqx modules?.
            # NOTE: strictly, we only need the equinox class name, not the make function which creates an instance
            # TODO: consider persisting the equinox class name instead of the make function
            maker_fun = get_object_from_module_and_qualname(module, qualname)
            model = maker_fun(**meta)  # NOTE: remember this returns a model with meta
            pytree = reader(path)
            model.model = model.model.__class__(**pytree)
            return model
        else:
            model = reader(path)  # NOTE: this is just an equinox model not mwm
            return cls(meta=meta, model=model, module=module, qualname=qualname)

    def __eq__(self, other):
        check_meta = self.meta == other.meta
        check_model = check_identical(self.model, other.model)
        return check_meta & check_meta


def model_maker(fun):
    """This is a decorator that wraps a function that takes some jsonifiable inputs parameters and returns a model.  The
    wrapped function returns a ModelWithMeta instance."""
    sig = inspect.signature(fun)
    qualname = fun.__qualname__
    module_ = fun.__module__

    @wraps(fun)
    def inner(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        model = fun(*args, **kwargs)
        meta = bound.arguments
        return ModelWithMeta(model=model, meta=meta, qualname=qualname, module=module_)

    return inner
