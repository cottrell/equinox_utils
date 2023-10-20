from contextlib import contextmanager
import equinox as eqx
from pathlib import Path


@contextmanager
def _path_or_buf(path_or_buf):
    if isinstance(path_or_buf, (str, Path)):
        with open(str(path_or_buf), 'wb') as buf:
            yield buf
    elif hasattr(path_or_buf, 'write'):
        yield path_or_buf
    else:
        raise ValueError("Invalid type for path_or_buf")

def write_equinox_via_tree_serialize_leaves(model, path_or_buf):
    assert isinstance(model, eqx.Module), f'expected model to be an equinox Module, got {type(model)}'
    with _path_or_buf(path_or_buf) as buf:
        eqx.tree_serialise_leaves(buf, model)

def read_equinox_via_tree_serialize_leaves(model, path_or_buf):
    pass


def write_equinox_via_recurse_get_state(model, path_or_buf):
    from .recurse_get_state import save_model_state
    with _path_or_buf(path_or_buf) as buf:
        save_model_state(model, buf)

def read_equinox_via_recurse_get_state(path_or_buf):
    pass

def write_equinox_via_orbax(model, path):
    """NOTE: orbax only supports paths not buf."""
    assert isinstance(model, eqx.Module), f'expected model to be an equinox Module, got {type(model)}'
    import orbax.checkpoint as ocp
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, model)

def read_equinox_via_orbax(model, path):
    pass


flavours = {
    'recurse_get_state': {'write': write_equinox_via_recurse_get_state, 'read': read_equinox_via_recurse_get_state},
    'orbax': {'write': write_equinox_via_orbax, 'read': read_equinox_via_orbax},
    'tree_serialize_leaves': {'write': write_equinox_via_tree_serialize_leaves, 'read': read_equinox_via_tree_serialize_leaves},

}
    
