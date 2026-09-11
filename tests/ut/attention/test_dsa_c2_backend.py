"""AscendDSAC2Backend's block sizes, and why C4 cannot stand in for it.

The list [64, 128, 256] is not a preference. AscendMLAAttentionSpec recovers the
physical page by dividing the ratio back out of block_size, so a spec's block_size
is the physical page size times the ratio, and that product is what
select_common_block_size is handed. These tests recompute the product from the
DSV4 block-size table rather than restating the literal, so they fail if either
the table or the backend moves without the other.

The second test is the one that matters. Reusing AscendDSAC4Backend for ratio 2
works at cache block_size 64 and 128 and fails only at 32 -- it passes on the
default and breaks on the small one, which is the shape of a bug nobody's test
run catches. Pinning that asymmetry here keeps the "just reuse C4" shortcut from
looking viable to the next reader.

Card-free: block-size lists and the DSV4 table are plain class/module data.
"""

from vllm.v1.worker.utils import select_common_block_size

from tests.ut.base import TestBase
from vllm_ascend.attention.dsa_v1 import (
    AscendDSABackend,
    AscendDSAC2Backend,
    AscendDSAC4Backend,
    AscendDSASWABackend,
)
from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

RATIO = 2


def _logical_sizes(ratio: int) -> list[int]:
    """Spec block sizes a cache at this ratio can be asked for, smallest first."""
    return sorted({row[0][0] * ratio for row in DSV4_BLOCK_SIZES.values()})


class TestAscendDSAC2Backend(TestBase):
    def test_lists_exactly_the_sizes_ratio_2_can_produce(self):
        self.assertEqual(AscendDSAC2Backend.get_supported_kernel_block_sizes(), _logical_sizes(RATIO))
        self.assertEqual(AscendDSAC2Backend.get_name(), "ASCEND_DSA_C2")

    def test_every_ratio_2_spec_size_is_taken_verbatim(self):
        """Case 1 must hit: a case-2 fallback silently allocates a different page size."""
        for want in _logical_sizes(RATIO):
            with self.subTest(spec_block_size=want):
                self.assertEqual(select_common_block_size(want, [AscendDSAC2Backend]), want)

    def test_c4_is_not_a_substitute_and_fails_only_on_the_small_block(self):
        """The reason C2 exists: C4 covers ratio 2 at 64 and 128 but not at 32."""
        accepted, rejected = [], []
        for block_size in sorted(DSV4_BLOCK_SIZES):
            want = DSV4_BLOCK_SIZES[block_size][0][0] * RATIO
            try:
                got = select_common_block_size(want, [AscendDSAC4Backend])
            except ValueError:
                rejected.append(block_size)
                continue
            (accepted if got == want else rejected).append(block_size)
        self.assertEqual(rejected, [32], f"expected C4 to fail only on cache block_size 32, got {rejected}")
        self.assertEqual(accepted, [64, 128], f"expected C4 to pass on 64 and 128, got {accepted}")

    def test_ratio_1_needs_no_new_backend(self):
        """Stops someone adding a C1 class by symmetry with C2."""
        for backend in (AscendDSABackend, AscendDSASWABackend):
            with self.subTest(backend=backend.get_name()):
                supported = backend.get_supported_kernel_block_sizes()
                for want in _logical_sizes(1):
                    self.assertIn(want, supported)

    def test_ratio_0_is_swallowed_silently_not_rejected(self):
        """Ratio 0 does not fail loudly -- it quietly takes the LARGEST page.

        A ratio-0 (sliding-window) layer computes a spec block_size of 0. That is not
        rejected: `select_common_block_size`'s case 2 asks `0 % supported == 0`, which
        holds for every candidate, so it returns the largest supported size and the
        planner allocates pages of that size for a cache that wanted none.

        This is why mapping ratio 0 to 1 before the spec is built is load-bearing rather
        than tidy: without it there is no exception to notice, only a wrong page size.
        Pinned here so that if upstream ever starts rejecting 0, we find out from this
        test instead of from a silently changed allocation.
        """
        self.assertEqual(_logical_sizes(0), [0])
        for backend in (AscendDSAC2Backend, AscendDSAC4Backend):
            with self.subTest(backend=backend.get_name()):
                largest = max(backend.get_supported_kernel_block_sizes())
                self.assertEqual(select_common_block_size(0, [backend]), largest)
