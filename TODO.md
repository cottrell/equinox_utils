---
title: equinox_utils TODO
created_date: '2026-05-13'
agent: Codex
session_id: 019e208d-ca44-77c2-9577-c6a674868eda
---
# equinox_utils TODO

Notes from reviewing TIY model training and serialization.

## Keep

`ModelWithMeta` and `model_maker` are still useful ideas. The important pattern
is:

- separate model leaves from JSON-serializable metadata;
- record the maker `module` and `qualname`;
- reconstruct the model skeleton from metadata before loading leaves;
- keep a small provenance file beside model weights.

This is especially useful for agent-led work, where implicit context is easy to
lose.

## Modernize

Prefer current Equinox primitives for normal Equinox models:

- `eqx.tree_serialise_leaves`
- `eqx.tree_deserialise_leaves`
- `eqx.filter_eval_shape` when constructing a load skeleton without allocating
  unnecessary arrays

The old `recurse_get_state` path should remain a fallback for awkward objects,
not the default.

Consider adding a clearer artifact layout:

```text
artifact_dir/
  model.eqx
  meta.json
  serialize_meta.json
  diagnostics.json
```

Where `serialize_meta.json` includes at least:

- serialization format and version;
- maker module and qualname;
- Equinox/JAX/equinox_utils versions if cheap to record;
- creation timestamp;
- optional agent/session provenance.

## Possible improvements

- Add a `save_artifact(...)` / `load_artifact(...)` API that is intentionally
  boring and documented as the recommended path.
- Keep `ModelWithMeta.save/load` as wrappers around that API.
- Add explicit format versions to `serialize_meta.json`.
- Add a small compatibility test that writes an artifact and loads it in a fresh
  Python process.
- Add an example showing the official Equinox "hyperparameters plus leaves"
  pattern and how `ModelWithMeta` wraps it.
- Add optional `diagnostics.json` support without coupling this package to any
  project-specific training loop.
- Review whether `tree_serialise_leaves` should be the default flavour
  everywhere now.
- Keep Orbax out of the default path for now. Add an Orbax backend only if there
  is a concrete need for optimizer-state checkpointing, checkpoint managers,
  async saves, sharded arrays, or distributed training.

## Current Orbax wrapper status

`equinox_utils/serialization.py` currently has:

```python
def write_equinox_via_orbax(model, path):
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, model)

def read_equinox_via_orbax(path, **kwargs):
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(path)
```

Treat this as not production-ready.

Reasons:

- Orbax is not currently an installed dependency in this project.
- The wrapper does not take or use an Equinox skeleton/template on restore.
- It does not save optimizer/train state, so it is not a real checkpoint helper.
- It does not integrate with `ModelWithMeta` reconstruction the way
  `tree_serialise_leaves` does.
- `ModelWithMeta.load(..., flavour='orbax')` tries to reconstruct with
  `model.model.__class__(**pytree)`, which is unlikely to be robust for normal
  Equinox modules.
- There is no active test coverage for the Orbax flavour.

If adding Orbax properly, design it as a checkpoint backend for a train state:

```text
checkpoint item:
  model
  optimizer_state
  step
  rng
  trainer_state
```

Keep final fitted-model artifacts on the simpler
`ModelWithMeta-like manifest + eqx.tree_serialise_leaves` path unless there is a
specific reason to use Orbax.

## TIY lesson

TIY currently has ad hoc `.npz`, `.eqx`, lineage JSON, and prediction artifacts.
That is not the same as a loadable model artifact. A small improved
`equinox_utils` artifact API could be reused there, but should stay general:
model metadata and Equinox leaves in; project-specific run metadata stays
outside.
