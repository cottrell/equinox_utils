import contextlib
import hashlib
import json
import os
import tempfile
from types import FunctionType

import jax
import jax.numpy as jnp


def tuple_to_list(tree):
    if isinstance(tree, tuple):
        return [tuple_to_list(elem) for elem in tree]
    elif isinstance(tree, list):
        return [tuple_to_list(elem) for elem in tree]
    elif isinstance(tree, dict):
        return {key: tuple_to_list(value) for key, value in tree.items()}
    else:
        return tree


# def check_identical(tree1, tree2):
#     def compare_elements(x, y):
#         if isinstance(x, FunctionType):
#             return x.__code__.co_code == y.__code__.co_code
#         else:
#             return jnp.all(x == y)
# 
#     comparison_tree = jax.tree_map(compare_elements, tree1, tree2)
# 
#     all_identical = all(jax.tree_util.tree_flatten(comparison_tree)[0])
#     return all_identical


def check_identical(tree1, tree2):
    leaves1, treedef1 = jax.tree_util.tree_flatten(tree1)
    leaves2, treedef2 = jax.tree_util.tree_flatten(tree2)

    if treedef1 != treedef2:
        return False

    def compare_elements(x, y):
        check = False
        if isinstance(x, FunctionType):
            check = x.__code__.co_code == y.__code__.co_code
        else:
            check = jnp.all(x == y)
        # if not check:
        #     breakpoint()
        return check

    all_identical = all(map(compare_elements, leaves1, leaves2))
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


@contextlib.contextmanager
def temp_then_atomic_move(path, dir=False, tempdir=None, prefix=".tmp_"):
    """Safe atomic writes (then move) to path. Will fail if target exists."""
    if dir:
        temp = tempfile.mkdtemp(prefix=prefix, dir=tempdir)
    else:
        temp = tempfile.mktemp(prefix=prefix, dir=tempdir)
    yield temp
    os.rename(temp, path)


def get_hash_from_params(params):
    # NOTE: dict of tuples are not supported in json, convert to nested dicts instead.
    text = json.dumps(params, sort_keys=True)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
