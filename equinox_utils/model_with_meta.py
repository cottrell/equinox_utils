import inspect
import io
import json
import logging
import os
from dataclasses import dataclass
from functools import wraps
from typing import Dict

import equinox as eqx

from .recurse_get_state import get_object_from_module_and_qualname, recurse_get_state
from .serialization import flavours as _serialization_flavours
from .util import check_identical

_MODEL_FILENAME = 'model.eqx'
_META_FILENAME = 'meta.json'
_SERIALIZE_META_FILENAME = 'serialize_meta.json'

logging.getLogger().setLevel(logging.DEBUG)


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

    def __init__(self, default_equinox_serialization_flavour='tree_serialise_leaves', *, meta, model, module, qualname):
        self.meta = meta
        self.model = model
        self.module = module
        self.qualname = qualname
        self.default_equinox_serialization_flavour = default_equinox_serialization_flavour

    def serialize_to_dict(self, flavour=None, **kwargs):
        """For in-memory serialization. Returns a dictionary of bytes corresponding to the files that would be written.

        Only intended to work for tree_serialise_leaves and recurse_get_state flavours."""
        out = dict()
        out[_META_FILENAME] = json.dumps(self.meta).encode()
        buf = io.BytesIO()
        flavour = self._save_model(buf, flavour=flavour, **kwargs)
        buf.seek(0)
        out[_MODEL_FILENAME] = buf.read()
        out[_SERIALIZE_META_FILENAME] = json.dumps(
            dict(
                serialization_flavour=flavour,
                module=self.module,
                qualname=self.qualname,
            )
        ).encode()
        return out

    def save(self, path, flavour=None, **kwargs):
        # TODO: consider using serialize to dict as intermediary to avoid logic twice
        os.makedirs(path, exist_ok=True)
        self._save_meta(os.path.join(path, _META_FILENAME))
        flavour = self._save_model(os.path.join(path, _MODEL_FILENAME), flavour=flavour, **kwargs)
        serialize_meta = dict(
            serialization_flavour=flavour,
            module=self.module,
            qualname=self.qualname,
        )
        open(os.path.join(path, _SERIALIZE_META_FILENAME), 'w').write(json.dumps(serialize_meta))

    def _save_model(self, path, *, flavour, **kwargs):
        flavour = flavour or self.default_equinox_serialization_flavour
        assert flavour in _serialization_flavours, f'unknown flavour {flavour}'
        writer = _serialization_flavours[flavour]['write']
        writer(self.model, path, **kwargs)
        return flavour

    def _save_meta(self, path):
        json.dump(self.meta, open(path, 'w'))

    @classmethod
    def load(cls, path):
        # TODO: consider using deserialize from dict as intermediary to avoid logic twice
        meta = json.load(open(os.path.join(path, _META_FILENAME)))
        serialize_meta = json.load(open(os.path.join(path, _SERIALIZE_META_FILENAME)))
        flavour = serialize_meta['serialization_flavour']
        reader = _serialization_flavours[flavour]['read']
        path = os.path.join(path, _MODEL_FILENAME)
        module = serialize_meta['module']
        qualname = serialize_meta['qualname']
        if flavour == 'tree_serialise_leaves':
            maker_fun = get_object_from_module_and_qualname(module, qualname)
            model = maker_fun(**meta)  # NOTE: remember this returns a model with meta
            model.model = reader(path, model=model.model)
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
        elif flavour == 'recurse_get_state':
            model = reader(path)  # NOTE: this is just an equinox model not mwm
            return cls(meta=meta, model=model, module=module, qualname=qualname)
        else:
            raise Exception(f'unknown flavour {flavour}')

    @classmethod
    def deserialize_from_dict(cls, d_in):
        meta = json.loads(d_in[_META_FILENAME])
        serialize_meta = json.loads(d_in[_SERIALIZE_META_FILENAME])
        flavour = serialize_meta['serialization_flavour']
        reader = _serialization_flavours[flavour]['read']
        module = serialize_meta['module']
        qualname = serialize_meta['qualname']
        buf = io.BytesIO(d_in[_MODEL_FILENAME])
        if flavour == 'tree_serialise_leaves':
            maker_fun = get_object_from_module_and_qualname(module, qualname)
            model = maker_fun(**meta)  # NOTE: remember this returns a model with meta
            model.model = reader(buf, model=model.model)
            return model
        elif flavour == 'orbax':
            raise Exception('not implemented')
            # # WARNING: this is experimental and I do not think will not work in general as no recursion on eqx modules?.
            # # NOTE: strictly, we only need the equinox class name, not the make function which creates an instance
            # # TODO: consider persisting the equinox class name instead of the make function
            # maker_fun = get_object_from_module_and_qualname(module, qualname)
            # model = maker_fun(**meta)  # NOTE: remember this returns a model with meta
            # pytree = reader(path)
            # model.model = model.model.__class__(**pytree)
            # return model
        elif flavour == 'recurse_get_state':
            model = reader(buf)  # NOTE: this is just an equinox model not mwm
            return cls(meta=meta, model=model, module=module, qualname=qualname)
        else:
            raise Exception(f'unknown flavour {flavour}')

    def __eq__(self, other):
        if set(self.__dataclass_fields__) != set(other.__dataclass_fields__):
            intersection = set(self.__dataclass_fields__) & set(other.__dataclass_fields__)
            missing_left = set(self.__dataclass_fields__) - set(other.__dataclass_fields__)
            missing_right = set(other.__dataclass_fields__) - set(self.__dataclass_fields__)
            logging.debug(
                f'fields do not match. missing_left: {missing_left}, missing_right: {missing_right}, intersection: {intersection}'
            )
            return False
        for k in self.__dataclass_fields__:
            if k == 'model':
                continue
            if not getattr(self, k) == getattr(other, k):
                logging.debug(f'fields do not match: {k} {getattr(self, k)} != {getattr(other, k)}')
                return False
        # TODO: I would like this one but it does not work due to ... classses (not class instances and not being equal
        # for some reason ... something to do with dynamic import vs static?)
        # check_model = check_identical(self.model, other.model)
        check_model = check_identical(recurse_get_state(self.model), recurse_get_state(other.model))
        # NOTE: why not use model.model.__eq__ ??? ... see test 'shared' there are problematic cases of functions etc.
        # arguably equinox __eq__ could be fixed but probably it should be strict there.
        # check_model = self.model == other.model
        if not check_model:
            logging.debug(f'models do not match: {self.model} != {other.model}')
        return check_model


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
        # NOTE: if we do not manually pop kwargs we end up with 'kwargs' in the meta instead of the items
        if 'kwargs' in bound.arguments:
            bound.arguments.update(bound.arguments.pop('kwargs'))
        model = fun(*args, **kwargs)
        meta = bound.arguments
        return ModelWithMeta(model=model, meta=meta, qualname=qualname, module=module_)

    return inner
