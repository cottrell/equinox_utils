import contextlib
import inspect
from functools import wraps
from dataclasses import dataclass
import dataclasses
import hashlib
import importlib

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
TEST_SERIALIZATION = True


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


