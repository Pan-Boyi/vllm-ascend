Structural trap: `tools/` on `sys.path[0]`
==========================================

Running `python tools/<x>.py` puts `tools/` at `sys.path[0]`, where any
package in here **shadows a stdlib module of the same name**. `tools/bisect/`
shadows the stdlib `bisect`, which `random` imports, which `tempfile`
imports, which `torch` imports -- so the script cannot start at all.

This has bitten twice (the upstream-substitution audit, then the L0 scale
gate). Twice is a structural property, not forgetfulness, and a guard at the
top of each script relies on the memory that already failed.

`tests/ut/test_tools_do_not_shadow_stdlib.py` fails if any directory here
shares a name with a stdlib module that is imported during interpreter or
torch start-up.
