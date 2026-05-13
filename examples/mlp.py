import equinox as eqx
import jax

import equinox_utils as eu


@eu.model_maker
def make_model(*, in_size=7, out_size=1, width_size=8, depth=5, seed=0, extra_stuff=None):
    if extra_stuff is None:
        extra_stuff = {"a": 1, "b": 2}
    key = jax.random.PRNGKey(seed=seed)
    model = eqx.nn.MLP(in_size, out_size, width_size, depth, key=key)
    return model


key = jax.random.PRNGKey(seed=0)
X = jax.random.normal(key, (10, 7))

model = make_model(seed=1)
