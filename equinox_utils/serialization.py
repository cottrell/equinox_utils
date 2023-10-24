import os
from contextlib import contextmanager
from pathlib import Path

import equinox as eqx


@contextmanager
def _path_or_buf(path_or_buf, mode='wb'):
    if isinstance(path_or_buf, (str, Path)):
        with open(str(path_or_buf), mode) as buf:
            yield buf
    elif hasattr(path_or_buf, 'write'):
        yield path_or_buf
    else:
        raise ValueError("Invalid type for path_or_buf")


def write_equinox_via_tree_serialize_leaves(model, path_or_buf):
    assert isinstance(model, eqx.Module), f'expected model to be an equinox Module, got {type(model)}'
    with _path_or_buf(path_or_buf) as buf:
        eqx.tree_serialise_leaves(buf, model)


def read_equinox_via_tree_serialize_leaves(path_or_buf, model, **kwargs):
    # NOTE: this is **not** an in-place as I previously though
    return eqx.tree_deserialise_leaves(path_or_buf, model)


def write_equinox_via_recurse_get_state(model, path_or_buf, **kwargs):
    from .recurse_get_state import save_model_state

    with _path_or_buf(path_or_buf, mode='w') as buf:
        save_model_state(model, buf, **kwargs)


def read_equinox_via_recurse_get_state(path_or_buf, **kwargs):
    from .recurse_get_state import load_model_state

    return load_model_state(path_or_buf)


# WARNING: orbax method is experimental work and basically probably does not work. Needs some recursive eqx stuff ...


def write_equinox_via_orbax(model, path):
    """NOTE: orbax only supports paths not buf."""
    assert isinstance(model, eqx.Module), f'expected model to be an equinox Module, got {type(model)}'
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    path = os.path.abspath(path)
    checkpointer.save(path, model)


def read_equinox_via_orbax(path, **kwargs):
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    path = os.path.abspath(path)
    return checkpointer.restore(path)


flavours = {
    'recurse_get_state': {'write': write_equinox_via_recurse_get_state, 'read': read_equinox_via_recurse_get_state},
    'orbax': {'write': write_equinox_via_orbax, 'read': read_equinox_via_orbax},
    'tree_serialize_leaves': {
        'write': write_equinox_via_tree_serialize_leaves,
        'read': read_equinox_via_tree_serialize_leaves,
    },
}
