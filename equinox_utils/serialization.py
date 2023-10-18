import contextlib
import inspect
from functools import wraps
from dataclasses import dataclass
import dataclasses
import hashlib
import importlib

# begin serialization lib
import io
import json
import lzma
import os
import pickle
import tempfile
from base64 import b64decode, b64encode
from types import FunctionType
from typing import Dict

import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int

# NOTE: see https://github.com/patrick-kidger/equinox/issues/535


def example_pytreedef_works(model):
    t = recurse_get_state(model)
    leaves, treedef = jax.tree_util.tree_flatten(t)

    # NOTE: this fails if use on model directly
    zdef = treedef.serialize_using_proto()

    # NOTE: can we do a registry_pytree_node for this to make it work?
    # It seems classes and functions can appear as leaves do not come through well.
    # zdata = leaves.serialize_somehow() # ???


# ## experiment with tree registry (didn't work don't understand)
# def flatten_module(container):
#     flat_contents = container.__getstate__()
#     aux_data = (container.__module__, container.__class__.__qualname__)
#     return flat_contents, aux_data
# 
# def unflatted_module(aux_data, flat_contents):
#     module, qualname = aux_data
#     class_ = get_object_from_module_and_qualname(module, qualname)
#     asdf
#     # params = reconstitute_from_root(module)
#     # return init_from_state_params(class_, params)
#     return 
# 
# jax.tree_util.register_pytree_node(eqx.Module, flatten_module, unflatted_module)
# 
# ##


def get_hash_from_params(params):
    # NOTE: dict of tuples are not supported in json, convert to nested dicts instead.
    text = json.dumps(params, sort_keys=True)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@contextlib.contextmanager
def temp_then_atomic_move(path, dir=False, tempdir=None, prefix=".tmp_"):
    """Safe atomic writes (then move) to path. Will fail if target exists."""
    if dir:
        temp = tempfile.mkdtemp(prefix=prefix, dir=tempdir)
    else:
        temp = tempfile.mktemp(prefix=prefix, dir=tempdir)
    yield temp
    os.rename(temp, path)


def save_model_state_to_path(model, path, array_flavour='tolist', tempdir=None):
    """Save to path/<hash_of_params>/params.json. This is useful in case you want to store meta data logs etc."""
    params = recurse_get_state(model)
    jsonifiable = params_to_jsonifiable(params)
    param_hash = get_hash_from_params(jsonifiable)
    hashpath = os.path.join(path, param_hash)
    os.makedirs(path, exist_ok=True)
    filename = None
    with temp_then_atomic_move(hashpath, dir=True, tempdir=tempdir) as tempdir:
        filename = os.path.join(tempdir, 'params.json')
        print(f'saving to {filename}')
        fout = open(filename, 'w')
        json.dump(jsonifiable, fout)
        fout.close()
    return dict(hashpath=hashpath)


def save_model_state(model, filename, array_flavour='tolist'):
    """
    Save a model to a "json" file with arrays encoded according to array_flavour.

    - array_flavour: one of 'tolist', 'save', 'save_xz_b64'
    """
    params = recurse_get_state(model)
    jsonifiable = params_to_jsonifiable(params)
    param_hash = get_hash_from_params(jsonifiable)
    print(f'saving to {filename}')
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    json.dump(jsonifiable, open(filename, 'w'))


def load_model_state(filename):
    jsonifiable = json.load(open(filename))
    params = jsonifiable_to_params(jsonifiable)
    model = reconstitute(params)
    return model


def io_helper(f_save):
    def inner(x):
        fout = io.BytesIO()
        f_save(fout, x)
        fout.seek(0)
        return fout.read()

    return inner


np_save = io_helper(np.save)
jnp_save = io_helper(jnp.save)


def _maybe_json_loads(x):
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
    'np_tolist': {'write': lambda x: x.tolist(), 'read': lambda x: np.array(_maybe_json_loads(x))},
    'jnp_tolist': {'write': lambda x: x.tolist(), 'read': lambda x: jnp.array(_maybe_json_loads(x))},
    'pickle': {'write': lambda x: b64encode(pickle.dumps(x)).decode(), 'read': lambda x: pickle.loads(b64decode(x))},
    'cloudpickle': {
        'write': lambda x: b64encode(cloudpickle.dumps(x)).decode(),
        'read': lambda x: cloudpickle.loads(b64decode(x)),
    },
}


def params_to_jsonifiable(params, array_flavour='tolist'):
    """
    Dict of params to something that shoudl be jsonifiable. Arrays handled according to array_flavour.

    - array_flavour: one of 'tolist', 'save', 'save_xz_b64'
    """

    # NOTE: probably awful just do something for now ... look for someone to have done something sane on the jax side
    # that isn't pickle. Likely the equinox pattern with some way to get at the typing would be fine.
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
                fun = serializers_deserializers['cloudpickle']['write']
                return f'cloudpickle:{fun(x)}'
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


# end serialization lib


def recurse_get_state(x):
    # NOTE: this is a somewhat custom recursion due to eqx.Module detection
    if isinstance(x, eqx.Module):
        # return {'module': {(x.__class__.__module__, x.__class__.__qualname__): recurse_get_state(x.__getstate__())}}
        # NOTE: some libraries like msgpack do not allow non-string dictionary keys so let's just are MORE NESTING
        return {'module': {x.__class__.__module__: {x.__class__.__qualname__: recurse_get_state(x.__getstate__())}}}
    elif isinstance(x, dict):
        # TODO: review this, symptom was in diffrax test got
        # dict_keys(['t0', 't1', 'ts', 'ys', 'interpolation', 'stats', 'result', 'solver_state', 'controller_state',
        # 'made_jump', '__doc__', '__annotations__', '__module__'])
        # comment out and uncomment below two lines to see error in test_diffrax
        # return {'dict': {k: recurse_get_state(v) for k, v in x.items() if not k.startswith('__')}}
        return {'dict': {k: recurse_get_state(v) for k, v in x.items() if not isinstance(k, str) or not k.startswith('__')}}
        # return {'dict': {k: recurse_get_state(v) for k, v in x.items()}}
    elif isinstance(x, list):
        return [recurse_get_state(v) for v in x]
    elif isinstance(x, tuple):
        return tuple(recurse_get_state(v) for v in x)
    else:
        return x


def init_from_state_params(class_, params):
    module = object.__new__(class_)
    fieldnames = {f.name for f in dataclasses.fields(class_)}
    if params is None:
        assert len(fieldnames) == 0
    else:
        assert set(params.keys()) == fieldnames
        for key, value in params.items():
            object.__setattr__(module, key, value)
    return module


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


# TESTS


TEST_SERIALIZATION = True


def serialization_test_fun(params):
    """params comes from recurse_get_state"""
    for array_flavour in ['tolist', 'save', 'save_xz_b64']:
        jsonifiable = params_to_jsonifiable(params, array_flavour=array_flavour)
        string_ = json.dumps(jsonifiable)
        jsonifiable_ = json.loads(string_)
        check = check_identical(jsonifiable, jsonifiable_)
        params_ = jsonifiable_to_params(jsonifiable_)
        check = check_identical(params, params_)
        if not check:
            return
        assert check_identical(params, params_), f'array_flavour={array_flavour} failed'


def tuple_to_list(tree):
    if isinstance(tree, tuple):
        return [tuple_to_list(elem) for elem in tree]
    elif isinstance(tree, list):
        return [tuple_to_list(elem) for elem in tree]
    elif isinstance(tree, dict):
        return {key: tuple_to_list(value) for key, value in tree.items()}
    else:
        return tree


def check_identical(tree1, tree2):
    def compare_elements(x, y):
        if isinstance(x, FunctionType):
            return x.__code__.co_code == y.__code__.co_code
        else:
            return jnp.all(x == y)

    comparison_tree = jax.tree_map(compare_elements, tree1, tree2)

    all_identical = all(jax.tree_util.tree_flatten(comparison_tree)[0])
    return all_identical


def check_identical_with_debug(tree1, tree2):
    disagreements = []

    def compare_elements(x, y):
        if isinstance(x, FunctionType):
            identical = x.__code__.co_code == y.__code__.co_code
        else:
            identical = jnp.all(x == y)

        if not identical:
            disagreements.append((x, y))

        return identical

    comparison_tree = jax.tree_map(compare_elements, tree1, tree2)
    all_identical = all(jax.tree_util.tree_flatten(comparison_tree)[0])
    print(f"all_identical: {all_identical}")

    if not all_identical:
        print("Disagreeing elements:")
        for x, y in disagreements:
            print(f"x: {x}, y: {y}")

    return all_identical


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))


class Another(eqx.Module):
    layers: list

    def __init__(self, n, in_size, out_size, key):
        self.layers = [Linear(in_size, out_size, key) for _ in range(n)]


def test_simple():
    key = jax.random.PRNGKey(0)
    in_size = 12
    out_size = 3
    n = 5
    a = Another(n, in_size, out_size, key)
    params = recurse_get_state(a)
    b = reconstitute(params)
    assert check_identical(a, b), f'failed'
    if TEST_SERIALIZATION:
        serialization_test_fun(params)


class Func(eqx.Module):
    func: FunctionType

    def __init__(self):
        self.func = lambda x: x


def test_func():
    a = Func()
    params = recurse_get_state(a)
    b = reconstitute(params)
    assert check_identical(a, b), f'failed'
    if TEST_SERIALIZATION:
        serialization_test_fun(params)


def test_lineax():
    from lineax import CG, GMRES, LU, QR, SVD, BiCGStab, Diagonal, NormalCG, Triangular, Tridiagonal

    for module_ in [BiCGStab, CG, GMRES, NormalCG]:
        a = module_(atol=1e-3, rtol=1e-4)
        params = recurse_get_state(a)
        b = reconstitute(params)
        assert check_identical(a, b), f'{module_} failed'
        if TEST_SERIALIZATION:
            serialization_test_fun(params)
    for module_ in [Diagonal, LU, QR, SVD, Triangular, Tridiagonal]:
        a = module_()
        params = recurse_get_state(a)
        b = reconstitute(params)
        assert check_identical(a, b), f'{module_} failed'
        if TEST_SERIALIZATION:
            serialization_test_fun(params)


def test_diffrax():
    from diffrax import Dopri5, ODETerm, diffeqsolve

    def f(t, y, args):
        return -y

    term = ODETerm(f)
    solver = Dopri5()
    y0 = jnp.array([2.0, 3.0])
    a = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=y0)
    params = recurse_get_state(a)
    b = reconstitute(params)
    assert check_identical(a, b), f'diffrax failed'
    if TEST_SERIALIZATION:
        serialization_test_fun(params)


class Model_stateful(eqx.Module):
    norm1: eqx.nn.BatchNorm
    spectral_linear: eqx.nn.SpectralNorm[eqx.nn.Linear]
    norm2: eqx.nn.BatchNorm
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.norm1 = eqx.nn.BatchNorm(input_size=3, axis_name="batch")
        self.spectral_linear = eqx.nn.SpectralNorm(
            layer=eqx.nn.Linear(in_features=3, out_features=32, key=key1),
            weight_name="weight",
            key=key2,
        )
        self.norm2 = eqx.nn.BatchNorm(input_size=32, axis_name="batch")
        self.linear1 = eqx.nn.Linear(in_features=32, out_features=32, key=key3)
        self.linear2 = eqx.nn.Linear(in_features=32, out_features=3, key=key4)

    def __call__(self, x, state):
        x, state = self.norm1(x, state)
        x, state = self.spectral_linear(x, state)
        x = jax.nn.relu(x)
        x, state = self.norm2(x, state)
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x, state


def test_stateful():
    # from https://docs.kidger.site/equinox/examples/stateful/
    key = jax.random.PRNGKey(0)
    a = Model_stateful(key=key)
    params = recurse_get_state(a)
    b = reconstitute(params)
    assert check_identical(a, b), f'stateful failed'
    # TODO: NOTE: abolish tuples as they are not json serializable round trip
    params = tuple_to_list(params)
    if TEST_SERIALIZATION:
        serialization_test_fun(params)


class LanguageModel_shared(eqx.Module):
    shared: eqx.nn.Shared

    def __init__(self, *, key):
        embedding = eqx.nn.Embedding(num_embeddings=3, embedding_size=4, key=key)
        linear = eqx.nn.Linear(in_features=4, out_features=3, key=key)
        # These two weights will now be tied together.
        where = lambda embed_and_lin: embed_and_lin[1].weight
        get = lambda embed_and_lin: embed_and_lin[0].weight
        self.shared = eqx.nn.Shared((embedding, linear), where, get)

    def __call__(self, tokens: Int[Array, "sequence"]):
        # Expand back out so we can evaluate these layers.
        embedding, linear = self.shared()
        assert embedding.weight is linear.weight  # same parameter!
        # Now go ahead and evaluate your language model.
        values = jax.vmap(embedding)(tokens)
        # ...  # other layers, probably
        return jax.vmap(linear)(values)


def test_shared():
    # from https://docs.kidger.site/equinox/api/nn/shared/
    key = jax.random.PRNGKey(0)
    a = LanguageModel_shared(key=key)
    params = recurse_get_state(a)
    b = reconstitute(params)
    assert check_identical(a, b), f'stateful failed'


def test_all():
    test_simple()
    test_func()
    test_diffrax()
    test_lineax()
    test_stateful()
    test_shared()


def get_summary_info(model):
    """An alternative repr useful for initial debugging"""
    import pandas as pd

    def get_info(v):
        info = dict()
        info['type'] = type(v).__name__
        if isinstance(v, (jax.Array, np.ndarray, float)):
            info['dtype'] = v.dtype.name if hasattr(v, 'dtype') else None
            info['shape'] = np.shape(v)
            info['size'] = np.size(v)
            info['nancount'] = np.isnan(v).sum()
            info['zerocount'] = np.size(v) - np.count_nonzero(v)
            info['min'] = np.min(v).item()
            info['max'] = np.max(v).item()
        return info

    d_ = {jax.tree_util.keystr(k): get_info(v) for k, v in jax.tree_util.tree_leaves_with_path(model)}
    return pd.DataFrame(d_).T


# NOTE: currently not frozen so you can do model.model = train(model.model, ...)
@dataclass
class ModelWithMeta:
    meta: Dict
    model: eqx.Module

    def save(self, path):
        # TODO: handle buf saving via zipfile or something or separate method
        os.makedirs(path, exist_ok=True)
        self._save_meta(os.path.join(path, 'meta.json'))
        self._save_model(os.path.join(path, 'model.eqx'))

    def _save_model(self, path):
        save_model_state(self.model, path)

    def _save_meta(self, path):
        json.dump(self.meta, open(path, 'w'))

    @classmethod
    def load(cls, path):
        meta = json.load(open(os.path.join(path, 'meta.json')))
        model = load_model_state(os.path.join(path, 'model.eqx'))
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

