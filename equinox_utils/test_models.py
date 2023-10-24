import pytest
from types import FunctionType
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

from .model_with_meta import model_maker, ModelWithMeta
from .recurse_get_state import _array_flavours


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


class Func(eqx.Module):
    func: FunctionType

    def __init__(self):
        self.func = lambda x: x


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


@model_maker
def make_model(*, seed=0, flavour, **kwargs):
    key = jax.random.PRNGKey(seed)
    if flavour == 'simple':
        in_size = 12
        out_size = 3
        n = 5
        model = Another(n, in_size, out_size, key)
    elif flavour == 'shared':
        # from https://docs.kidger.site/equinox/api/nn/shared/
        model = LanguageModel_shared(key=key)
    elif flavour == 'stateful':
        # from https://docs.kidger.site/equinox/examples/stateful/
        model = Model_stateful(key=key)
    elif flavour.startswith('lineax-'):
        from lineax import CG, GMRES, LU, QR, SVD, BiCGStab, Diagonal, NormalCG, Triangular, Tridiagonal

        module_ = flavour.split('-')[1]
        module_ = locals()[module_]
        if module_ in [BiCGStab, CG, GMRES, NormalCG]:
            model = module_(atol=1e-3, rtol=1e-4)
        elif module_ in [Diagonal, LU, QR, SVD, Triangular, Tridiagonal]:
            model = module_()
        else:
            raise ValueError(f'unknown module {module_}')
    elif flavour == 'diffrax':
        from diffrax import Dopri5, ODETerm, diffeqsolve

        def f(t, y, args):
            return -y

        term = ODETerm(f)
        solver = Dopri5()
        y0 = jnp.array([2.0, 3.0])
        model = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=y0)
    elif flavour == 'func':
        model = Func()
    else:
        raise ValueError(f'unknown flavour {flavour}')
    return model


flavours = [
    'simple',
    'shared',
    'stateful',
    'lineax-CG',
    'lineax-GMRES',
    'lineax-LU',
    'lineax-QR',
    'lineax-SVD',
    'lineax-BiCGStab',
    'lineax-Diagonal',
    'lineax-NormalCG',
    'lineax-Triangular',
    'lineax-Tridiagonal',
    'diffrax',
    'func',
]

def _helper(*, model_flavour, serialization_flavour, **kwargs):
    model = make_model(flavour=model_flavour, seed=2, something_else=dict(a=1, b=2, c=['here', 'is', 'more']))
    filename = tempfile.mktemp(prefix='equinox_util_test_')
    print(f'flavour={model_flavour} serialization_flavour={serialization_flavour} {kwargs}, saving to {filename}')
    model.save(filename, flavour=serialization_flavour, **kwargs)
    model_ = ModelWithMeta.load(filename)
    assert model_ == model, f'flavour={model_flavour} {kwargs} failed eq check'

marks = {
    'stateful': pytest.param('stateful', marks=pytest.mark.xfail(reason="stateful not working yet")),
    'lineax-LU': pytest.param('stateful', marks=pytest.mark.xfail(reason="some lineax not working yet")),
    'lineax-QR': pytest.param('stateful', marks=pytest.mark.xfail(reason="some lineax not working yet")),
    'lineax-Triangular': pytest.param('stateful', marks=pytest.mark.xfail(reason="some lineax not working yet")),
    'lineax-Tridiagonal': pytest.param('stateful', marks=pytest.mark.xfail(reason="some lineax not working yet")),
}
flavours_with_marks = [marks.get(x, x) for x in flavours]

@pytest.mark.parametrize("model_flavour", flavours_with_marks)
def test_serialization_tree_serialize_leaves(model_flavour):
    serialization_flavour = 'tree_serialize_leaves'
    _helper(model_flavour=model_flavour, serialization_flavour=serialization_flavour)


@pytest.mark.parametrize("model_flavour", flavours_with_marks)
@pytest.mark.parametrize("array_flavour", _array_flavours)
def test_serialization_recurse_get_state(model_flavour, array_flavour):
    _helper(model_flavour=model_flavour, serialization_flavour='recurse_get_state', array_flavour=array_flavour)

# def test_all():
#     for model_flavour in flavours:
#         model = make_model(flavour=model_flavour, seed=2, something_else=dict(a=1, b=2, c=['here', 'is', 'more']))
# 
#         filename = tempfile.mktemp(prefix='equinox_util_test_')
#         print(f'flavour={model_flavour} serialization_flavour={serialization_flavour} saving to {filename}')
#         model.save(filename, flavour=serialization_flavour)
#         model_ = ModelWithMeta.load(filename)
#         assert model_ == model, f'flavour={model_flavour} failed eq check'
# 
#         serialization_flavour = 'recurse_get_state'
#         for array_flavour in _array_flavours:
#             filename = tempfile.mktemp(prefix='equinox_util_test_')
#             print(f'flavour={model_flavour} serialization_flavour={serialization_flavour} array_flavour={array_flavour} saving to {filename}')
#             model.save(filename, flavour=serialization_flavour, array_flavour=array_flavour)
#             model_ = ModelWithMeta.load(filename)
#             assert model_ == model, f'flavour={model_flavour} array_flavour={array_flavour} failed eq check'