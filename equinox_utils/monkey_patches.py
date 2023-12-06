from .summary_info import get_info, get_summary_info


def monkey_patch_equinox():
    import equinox as eqx

    from .serialization import write_equinox_via_recurse_get_state as save_model_state

    eqx.Module.summary_info = property(get_summary_info)
    eqx.Module.info = property(get_info)
    eqx.Module.save_model_state = save_model_state
