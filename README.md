# equinox_utils

<ins>***Experimental***</ins> utilities for working with [equinox](github.com/patrick-kidger/equinox).

```shell
python -m build
pip install -e .  # develop install
pip install .
```


## tests

```shell
pytest .
```

## monkey patches

```shell
~/.ipython/profile_default/startup$ cat zzz_monkeypatches.py
import equinox_utils.monkey_patches
```


```python
In [10]: model
Out[10]:
Solution(
  t0=f32[],
  t1=f32[],
  ts=f32[1],
  ys=f32[1,2],
  interpolation=None,
  stats={
    'max_steps':
    4096,
    'num_accepted_steps':
    i32[],
    'num_rejected_steps':
    i32[],
    'num_steps':
    i32[]
  },
  result=i32[],
  solver_state=None,
  controller_state=None,
  made_jump=None
)

In [11]: model.summary_info
Out[11]:
                                   type    dtype   shape size nancount zerocount       min       max
.t0                           ArrayImpl  float32      ()    1        0         1       0.0       0.0
.t1                           ArrayImpl  float32      ()    1        0         0       1.0       1.0
.ts                           ArrayImpl  float32    (1,)    1        0         0       1.0       1.0
.ys                           ArrayImpl  float32  (1, 2)    2        0         0  0.735759  1.103638
.stats['max_steps']                 int      NaN     NaN  NaN      NaN       NaN       NaN       NaN
.stats['num_accepted_steps']  ArrayImpl    int32      ()    1        0         0        10        10
.stats['num_rejected_steps']  ArrayImpl    int32      ()    1        0         1         0         0
.stats['num_steps']           ArrayImpl    int32      ()    1        0         0        10        10
.result                       ArrayImpl    int32      ()    1        0         1         0         0
```

## `ModelWithMeta` from `@model_maker` example

This pattern gives you a holder class for (non-trainable) meta data and the (trainable) equinox model.
```python
import equinox_util as eu 

@eu.model_maker
def make_model(*, in_size=7, out_size=1, width_size=8, depth=5, seed=0, extra_stuff=dict(a=1, b=2)):
    key = jax.random.PRNGKey(seed=0)
    model = eqx.nn.MLP(in_size, out_size, width_size, depth, key=key)
    return model
```

Usage

```python
In [56]: import jax

In [57]: import equinox_utils as eu

In [58]: import examples.mlp as em

In [59]: import jax

In [60]: mwm = em.make_model()

In [61]: mwm.__dataclass_fields__.keys()
Out[61]: dict_keys(['module', 'qualname', 'meta', 'model'])

In [62]: mwm.meta
Out[62]:
{'in_size': 7,
 'out_size': 1,
 'width_size': 8,
 'depth': 5,
 'seed': 0,
 'extra_stuff': {'a': 1, 'b': 2}}

In [63]: mwm.model.summary_info
Out[63]:
                         type    dtype   shape size nancount zerocount       min       max
.layers[0].weight   ArrayImpl  float32  (8, 7)   56        0         0  -0.37626  0.357937
.layers[0].bias     ArrayImpl  float32    (8,)    8        0         0  -0.31719  0.231203
.layers[1].weight   ArrayImpl  float32  (8, 8)   64        0         0  -0.34055  0.345468
.layers[1].bias     ArrayImpl  float32    (8,)    8        0         0 -0.291942  0.348188
.layers[2].weight   ArrayImpl  float32  (8, 8)   64        0         0 -0.346929  0.351709
.layers[2].bias     ArrayImpl  float32    (8,)    8        0         0 -0.204486  0.347938
.layers[3].weight   ArrayImpl  float32  (8, 8)   64        0         0 -0.352663  0.352085
.layers[3].bias     ArrayImpl  float32    (8,)    8        0         0 -0.262711  0.316399
.layers[4].weight   ArrayImpl  float32  (8, 8)   64        0         0 -0.337944  0.346138
.layers[4].bias     ArrayImpl  float32    (8,)    8        0         0 -0.295722  0.338768
.layers[5].weight   ArrayImpl  float32  (1, 8)    8        0         0 -0.302109  0.257613
.layers[5].bias     ArrayImpl  float32    (1,)    1        0         0   0.29676   0.29676
.activation        custom_jvp      NaN     NaN  NaN      NaN       NaN       NaN       NaN
.final_activation    function      NaN     NaN  NaN      NaN       NaN       NaN       NaN

In [64]: mwm.save('model.eqx')

In [65]: mwm2 = eu.load_model('model.eqx')

In [66]: mwm == mwm2
Out[66]: True

In [67]: key = jax.random.PRNGKey(seed=0)
    ...: X = jax.random.normal(key, (10, 7))

In [68]: jax.vmap(mwm.model)(X).shape
Out[68]: (10, 1)

In [69]: jax.vmap(mwm.model)(X)
Out[69]:
Array([[0.35647807],
       [0.35181946],
       [0.35584906],
       [0.35654053],
       [0.3513621 ],
       [0.35666227],
       [0.35779208],
       [0.35553125],
       [0.3535732 ],
       [0.3556212 ]], dtype=float32)

In [70]: jax.vmap(mwm2.model)(X)
Out[70]:
Array([[0.35647807],
       [0.35181946],
       [0.35584906],
       [0.35654053],
       [0.3513621 ],
       [0.35666227],
       [0.35779208],
       [0.35553125],
       [0.3535732 ],
       [0.3556212 ]], dtype=float32)
```
