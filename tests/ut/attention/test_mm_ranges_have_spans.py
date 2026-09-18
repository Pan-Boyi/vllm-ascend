# SPDX-License-Identifier: Apache-2.0
"""A container is not its contents.

``mm_req_doc_ranges`` is built for *every* request whenever the checkpoint is
a mm-prefix LM, and a request with no image contributes an empty list under
its own index. So a text-only prefill on V4.1-Flash arrives as ``{0: []}`` --
which is true. Testing the dict rather than its contents put every text-only
prefill through the bidirectional-vision index path, which supplies an
explicit index list, which puts the operator in ori-only sparse mode.

The values below are read from the checkpoint and from the config class, not
recalled: if the vision tower is ever removed from this checkpoint the trigger
disappears and the test that pins it should say so out loud.
"""

import pytest

from vllm_ascend.attention.dsa_v1 import mm_ranges_have_spans


TEXT_ONLY_ONE_REQUEST = {0: []}
TEXT_ONLY_FOUR_REQUESTS = {0: [], 1: [], 2: [], 3: []}
ONE_IMAGE_AMONG_FOUR = {0: [], 1: [(12, 40)], 2: [], 3: []}


def test_the_shape_that_caused_this_is_true_as_a_dict():
    """The premise. If this ever stops holding, the bug is gone by itself."""
    assert bool(TEXT_ONLY_ONE_REQUEST), (
        "a dict of empty span lists is falsy now, so the condition this "
        "replaces was never wrong and this whole file is pinning nothing")


@pytest.mark.parametrize("ranges", [TEXT_ONLY_ONE_REQUEST,
                                    TEXT_ONLY_FOUR_REQUESTS])
def test_no_spans_is_not_a_vision_batch(ranges):
    assert mm_ranges_have_spans(ranges) is False


@pytest.mark.parametrize("ranges", [ONE_IMAGE_AMONG_FOUR,
                                    {0: [(0, 3)]},
                                    {7: [(1, 2), (9, 20)]}])
def test_one_real_span_anywhere_is_a_vision_batch(ranges):
    assert mm_ranges_have_spans(ranges) is True


def test_absent_or_empty_is_false():
    assert mm_ranges_have_spans(None) is False
    assert mm_ranges_have_spans({}) is False


def test_a_non_mapping_is_reported_rather_than_reinterpreted():
    """A list of range lists would have no .values(); guessing its shape here
    is how a gate starts meaning something different from what it says."""
    assert mm_ranges_have_spans([[], []]) is True


def test_the_checkpoint_still_arms_this_path():
    """vision_n_layers > 0 is what makes a text request take the vision path.

    Read from the checkpoint rather than asserted as a remembered 32: the
    trigger is "a multimodal checkpoint serving a text-only request", and if
    the checkpoint stops being multimodal the trigger is gone.
    """
    import json
    import pathlib

    config = pathlib.Path(
        "/workspace/models/DeepSeek-V4.1-Flash/config.json")
    if not config.exists():
        pytest.skip(f"{config} is not present on this host")
    vision = json.loads(config.read_text()).get("vision_config") or {}
    assert vision.get("num_hidden_layers", 0) > 0, (
        "this checkpoint no longer carries a vision tower, so "
        "DeepseekV41Config leaves is_mm_prefix_lm False, the runner builds no "
        "mm_req_doc_ranges, and the condition this guards is unreachable")
