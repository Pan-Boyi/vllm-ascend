# SPDX-License-Identifier: Apache-2.0
"""The registry entry, and the guard that stands behind it.

Registering the architecture is one line. What that line has to be worth is
that everything reached through it has been adapted -- and the adaptations are
rebinds of upstream names, which is exactly the kind of thing that can be
undone by a later import with nothing to show for it.
"""

import importlib

import pytest

from vllm_ascend.models.deepseek_v4_1 import caches, model
from vllm_ascend.models.deepseek_v4_1.platform_patches import CLASS_REBINDS

#: The architectures entry in the checkpoint's config.json.
ARCH = "DeepseekV41ForCausalLM"
TARGET = "vllm_ascend.models.deepseek_v4_1.model:AscendDeepseekV41ForCausalLM"


def test_the_architecture_is_registered_to_the_ascend_class():
    import inspect

    import vllm_ascend.models as models

    source = inspect.getsource(models.register_model)
    assert f'"{ARCH}"' in source
    assert TARGET in source


def test_the_registered_path_resolves():
    """A registry entry is a string; nothing checks it until a model loads."""
    module_name, _, class_name = TARGET.partition(":")
    resolved = getattr(importlib.import_module(module_name), class_name)
    assert resolved is model.AscendDeepseekV41ForCausalLM


def test_it_subclasses_upstream_rather_than_forking_it():
    """V4.0's Ascend port forks upstream's model into 1352 lines. This one does
    not, and the reason is which failure is preferred: a rebind that drifts
    raises and names itself, a forked model that drifts quietly computes
    something else."""
    from vllm.models.deepseek_v4_1.nvidia.vl_model import DeepseekV41ForCausalLM

    assert issubclass(model.AscendDeepseekV41ForCausalLM, DeepseekV41ForCausalLM)

    # Only the constructor is overridden, to run the guard. Read from the
    # class body rather than vars(): the class picks up __subclasshook__,
    # _abc_impl and friends from the ABC and Protocol machinery upstream
    # inherits, none of which this file wrote.
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(model))
    body = next(node for node in tree.body
                if isinstance(node, ast.ClassDef)
                and node.name == "AscendDeepseekV41ForCausalLM")
    defined = [n.name for n in body.body if isinstance(n, ast.FunctionDef)]
    assert defined == ["__init__"], (
        f"the entry point defines {defined}; anything beyond __init__ is a "
        "behaviour change living somewhere nobody looks for one")


def test_importing_the_entry_point_applies_the_patches():
    """The ordering the whole scheme rests on: the target module lives inside
    the package, so there is no path that reaches the model without them."""
    model._assert_patches_applied()


@pytest.mark.parametrize("module_name,upstream_name,ascend_name", CLASS_REBINDS)
def test_the_guard_catches_each_rebind_being_undone(module_name, upstream_name,
                                                    ascend_name):
    """Every rebind, not just one.

    Each is silent in its own way when missing: an unpatched selector builds
    the CUDA attention class, an unpatched engram table asks for 188.83 GiB of
    device memory, an unpatched compressor cache has no get_kv_cache_shape.
    """
    module = importlib.import_module(module_name)
    original = getattr(module, upstream_name)
    setattr(module, upstream_name, type("Restored", (), {}))
    try:
        with pytest.raises(RuntimeError) as exc:
            model._assert_patches_applied()
        assert upstream_name in str(exc.value)
        assert ascend_name in str(exc.value)
    finally:
        setattr(module, upstream_name, original)

    model._assert_patches_applied()   # and passes again once restored


def test_the_guard_reads_the_bindings_not_the_record_of_applying_them():
    """apply_all's return value says what was true at import.

    The failure being guarded against is something re-importing an upstream
    module afterwards and restoring its original names -- which leaves that
    record saying everything is fine. So the guard has to evaluate the
    bindings, not consult a list.
    """
    import inspect

    source = inspect.getsource(model._assert_patches_applied)
    assert "APPLIED_PATCHES" not in source
    assert "import_module" in source and "getattr" in source


def test_the_engram_table_is_the_ascend_one():
    """The rebind that decides whether the model fits at all: 188.83 GiB of
    engram tables, 39.8% of the checkpoint, read from a host mapping."""
    import vllm.models.deepseek_v4_1.common.engram as engram_mod

    assert engram_mod.ParallelEngramEmbedding is caches.AscendEngramEmbedding
    mro = [c.__name__ for c in caches.AscendEngramEmbedding.__mro__[:3]]
    assert mro == ["AscendEngramEmbedding", "AscendParallelEngramEmbedding",
                   "ParallelEngramEmbedding"], (
        "the mixin must precede upstream's class or its _storage and lookup "
        f"lose the MRO; got {mro}")
