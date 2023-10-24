"""NOTE: experimental."""
import dataclasses
import importlib
import io
import lzma
import pickle
from base64 import b64decode, b64encode
from types import FunctionType

import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import json_util as json


def recurse_get_state(x):
    """custom recursion for eqx.Module detection"""
    if isinstance(x, eqx.Module):
        return {'module': {x.__class__.__module__: {x.__class__.__qualname__: recurse_get_state(x.__getstate__())}}}
    elif isinstance(x, eqx.nn._shared.SharedNode):
        # NOTE: this is just None state I think
        return {'module': {x.__class__.__module__: {x.__class__.__qualname__: recurse_get_state(x.__getstate__())}}}
    elif isinstance(x, dict):
        return {
            'dict': {k: recurse_get_state(v) for k, v in x.items() if not isinstance(k, str) or not k.startswith('__')}
        }
    elif isinstance(x, list):
        return [recurse_get_state(v) for v in x]
    elif isinstance(x, tuple):
        return tuple(recurse_get_state(v) for v in x)
    else:
        return x


def recurse_diff(x, y):
    assert (type(x) == type(y)) or {type(x), type(y)}.issubset({tuple, list}), f'expected same type got {type(x)} and {type(y)}'
    if isinstance(x, eqx.Module):
        x = recurse_get_state(x)
        y = recurse_get_state(y)
    if isinstance(x, dict):
        assert set(x.keys()) == set(y.keys())
        return {k: recurse_diff(x[k], y[k]) for k in x.keys()}
    elif isinstance(x, list):
        return [recurse_diff(x_, y_) for x_, y_ in zip(x, y)]
    elif isinstance(x, tuple):
        return tuple(recurse_diff(x_, y_) for x_, y_ in zip(x, y))
    else:
        return x == y


def init_from_state_params(class_, params):
    if dataclasses.is_dataclass(class_):
        module = object.__new__(class_)
        fieldnames = {f.name for f in dataclasses.fields(class_)}
        if params is None:
            assert len(fieldnames) == 0
        else:
            assert set(params.keys()) == fieldnames
            for key, value in params.items():
                object.__setattr__(module, key, value)
        return module
    else:
        if params is None:
            # NOTE: this is only for ShareNode case I think. It is *not* a dataclass or a eqx.Module
            obj = class_()
        else:
            obj = class_(**params)  # NOTE: I do not think ever hitting this case right now
        return obj



def get_object_from_module_and_qualname(module_name, qualname):
    module = importlib.import_module(module_name)
    obj = module
    for attr in qualname.split('.'):
        obj = getattr(obj, attr)
    return obj


def reconstitute_from_root(params):
    out = None
    if isinstance(params, dict):
        assert len(params) == 1
        k, v = list(params.items())[0]
        if k == 'module':
            assert len(v) == 1
            module, v = list(v.items())[0]
            assert len(v) == 1
            qualname, v = list(v.items())[0]
            class_ = get_object_from_module_and_qualname(module, qualname)
            params_ = reconstitute_from_root(v)
            out = init_from_state_params(class_, params_)
        elif k == 'dict':
            out = {k_: reconstitute_from_root(v_) for k_, v_ in v.items()}
        else:
            raise Exception(f'unknown key {k}')
    elif isinstance(params, list):
        out = [reconstitute_from_root(v) for v in params]
    elif isinstance(params, tuple):
        out = tuple(reconstitute_from_root(v) for v in params)
    else:
        out = params
    return out


def reconstitute(params):
    module = reconstitute_from_root(params)
    return module
    if len(module) == 1:
        return module[list(module.keys())[0]]


def io_helper(f_save):
    def inner(x):
        fout = io.BytesIO()
        f_save(fout, x)
        fout.seek(0)
        return fout.read()

    return inner


np_save = io_helper(np.save)
jnp_save = io_helper(jnp.save)


def maybe_json_loads(x):
    if isinstance(x, str):
        if x in ('True', 'False'):
            return bool(x)
        else:
            return json.loads(x)
    return x


serializers_deserializers = {
    'np_save': {'write': lambda x: b64encode(np_save(x)).decode(), 'read': lambda x: np.load(io.BytesIO(b64decode(x)))},
    'jnp_save': {
        'write': lambda x: b64encode(jnp_save(x)).decode(),
        'read': lambda x: jnp.load(io.BytesIO(b64decode(x))),
    },
    'np_save_xz_b64': {
        'write': lambda x: b64encode(lzma.compress(np_save(x))).decode(),
        'read': lambda x: np.load(io.BytesIO(lzma.decompress(b64decode(x)))),
    },
    'jnp_save_xz_b64': {
        'write': lambda x: b64encode(lzma.compress(jnp_save(x))).decode(),
        'read': lambda x: jnp.load(io.BytesIO(lzma.decompress(b64decode(x)))),
    },
    # NOTE: these are pretty awful now as they are not even really json anymore with this str: prefix thing
    'np_tolist': {'write': lambda x: x.tolist(), 'read': lambda x: np.array(maybe_json_loads(x))},
    'jnp_tolist': {'write': lambda x: x.tolist(), 'read': lambda x: jnp.array(maybe_json_loads(x))},
    'pickle': {'write': lambda x: b64encode(pickle.dumps(x)).decode(), 'read': lambda x: pickle.loads(b64decode(x))},
    'cloudpickle': {
        'write': lambda x: b64encode(cloudpickle.dumps(x)).decode(),
        'read': lambda x: cloudpickle.loads(b64decode(x)),
    },
}

_array_flavours = ['tolist', 'save', 'save_xz_b64']


def params_to_jsonifiable(params, array_flavour='tolist', allow_pickle_fallback=False):
    f"""
    Dict of params to something that shoudl be jsonifiable. Arrays handled according to array_flavour.

    - array_flavour: one of {_array_flavours}
    """

    assert array_flavour in _array_flavours
    def inner(x):
        if isinstance(x, jax.Array):
            key = {'tolist': 'jnp_tolist', 'save': 'jnp_save', 'save_xz_b64': 'jnp_save_xz_b64'}[array_flavour]
            fun = serializers_deserializers[key]['write']
            return f'{key}:{fun(x)}'
        elif isinstance(x, np.ndarray):
            key = {'tolist': 'np_tolist', 'save': 'np_save', 'save_xz_b64': 'np_save_xz_b64'}[array_flavour]
            fun = serializers_deserializers[key]['write']
            return f'{key}:{fun(x)}'
        elif isinstance(x, FunctionType):
            # NOTE: bad but just for functions not sure what else to do here
            fun = serializers_deserializers['cloudpickle']['write']
            return f'cloudpickle:{fun(x)}'
        else:
            try:
                json.dumps(x)
            except TypeError:
                if allow_pickle_fallback:
                    fun = serializers_deserializers['cloudpickle']['write']
                    return f'cloudpickle:{fun(x)}'
                else:
                    raise Exception(
                        f'failed to json.dumps(x) for x={x}, set allow_pickle_fallback=True if you want to use cloudpickle'
                    )
            return x  # f'json:{x}'

    return jax.tree_map(inner, params)


def jsonifiable_to_params(jsonifiable):
    def inner(x):
        if not isinstance(x, str):
            return x
        args = x.split(':', 1)
        if len(args) == 1:
            return x
        key, val = args
        fun = serializers_deserializers[key]['read']
        return fun(val)

    return jax.tree_map(inner, jsonifiable)


def save_model_state(model, buf, array_flavour='tolist'):
    """Save a model to a "json" file with arrays encoded according to array_flavour.

    - array_flavour: one of 'tolist', 'save', 'save_xz_b64'"""
    params = recurse_get_state(model)
    jsonifiable = params_to_jsonifiable(params)
    json.dump(jsonifiable, buf)


def load_model_state(filename):
    jsonifiable = json.load(open(filename))
    params = jsonifiable_to_params(jsonifiable)
    model = reconstitute(params)
    return model
