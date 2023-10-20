from dataclasses import dataclass
import os
import json
import equinox as eqx
from functools import wraps
from typing import Dict
from .util import check_identical
from .serialization import flavours as _serialization_flavours

_MODEL_FILENAME = 'model.eqx'
_META_FILENAME = 'meta.json'
_SERIALIZE_META_FILENAME = 'serialize_meta.json'

@dataclass
class ModelWithMeta:
    """A holder for equinox model (model) and meta data dictionary (meta).
    
    NOTE: currently not frozen so you are able to do model.model = train(model.model, ...)."""
    meta: Dict
    model: eqx.Module

    def __init__(self, default_equinox_serialization_flavour='tree_serialize_leaves'):
        self.default_equinox_serialization_flavour = default_equinox_serialization_flavour

    def save(self, path, flavour=None):
        os.makedirs(path, exist_ok=True)
        self._save_meta(os.path.join(path, 'meta.json'))
        flavour = self._save_model(os.path.join(path, 'model.eqx'), flavour=flavour)
        serialize_meta = dict(serialization_flavour=flavour)
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
        flavour = json.load(os.path.join(path, 'serialize_meta.json'))['serialization_flavour']
        reader = _serialization_flavours[flavour]['reader']
        model = reader(os.path.join(path, 'model.eqx'))
        return cls(meta=meta, model=model)

    def __eq__(self, other):
        check_meta = self.meta == other.meta
        check_model = check_identical(self.model, other.model)
        return check_meta & check_meta


def model_maker(fun):
    """This is a decorator that wraps a function that takes some jsonifiable inputs parameters and returns a model.  The
    wrapped function returns a ModelWithMeta instance."""
    sig = inspect.signature(fun)

    @wraps(fun)
    def inner(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        model = fun(*args, **kwargs)
        meta = bound.arguments
        return ModelWithMeta(model=model, meta=meta)

    return inner

