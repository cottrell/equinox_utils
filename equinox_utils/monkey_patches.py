import jax
import numpy as np


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


def monkey_patch_equinox():
    import equinox as eqx

    eqx.Module.summary_info = property(get_summary_info)
    # eqx.Module.save_model_state = save_model_state


monkey_patch_equinox()
