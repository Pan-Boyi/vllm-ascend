# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ask whether an aclnn operator path is actually WALKABLE -- not whether a name exists.

Why this module exists
----------------------
A custom Ascend operator only runs if four separate things line up, and each of
them can be absent while the other three look perfect:

1. the torch schema is registered (``torch.ops._C_ascend.<name>``);
2. the aclnn entry points ``aclnn<OpType>`` and ``aclnn<OpType>GetWorkspaceSize``
   resolve through the same search order ``GetOpApiFuncAddr`` uses
   (``csrc/aclnn_torch_adapter/op_api_common.h``);
3. they resolve out of the operator package WE ship, not out of some other
   library that happens to export the same name;
4. the package actually contains a kernel for the operator, i.e. the OpType is
   in its ops-info registry.

Checking only (1) is the mistake this module was written to stop. ``csrc``
registers every torch schema UNCONDITIONALLY, so (1) passes for operators that
were never compiled for this SoC -- a guard built on it can never fail. And (3)
is worse than (2): CANN's own ``libopapi.so`` exports some of the same names
with DIFFERENT SIGNATURES, so a name-existence check reads exactly the same
whether the operator is installed or whether a stranger answered to its name.

Three failure shapes, measured -- cover all three or you ship a false green
-------------------------------------------------------------------------
The same class of defect ("our package does not supply this operator") shows up
in three different ways depending on the operator. A guard written for one of
them gives a FALSE GREEN on the other two. Measured on this machine, 2026-09-20:

  A. NOBODY ANSWERS        -- e.g. ``AddRmsNormBias``, ``TransposeKvCacheByBlock``.
     Neither our vendor package nor CANN exports the symbol.
     Caught by: level (2).

  B. A STRANGER ANSWERS, WITH A DIFFERENT SIGNATURE -- e.g. ``RmsNormDynamicQuant``.
     Our package lacks it; CANN's ``libopapi.so`` exports the same name with 8
     parameters while ``EXEC_NPU_CMD`` passes 12. dlsym succeeds, ``TORCH_CHECK``
     passes, nothing downstream compares signatures.
     Caught by: level (3) provenance, and independently by ``expected_arg_count``.

  C. A STRANGER ANSWERS, WITH THE **SAME** SIGNATURE -- e.g.
     ``InplacePartialRotaryMul``. CANN 9.2 exports it with the same 5 parameters
     we pass. ``expected_arg_count`` HAS NO DISCRIMINATING POWER HERE.
     Caught by: level (3) provenance ONLY -> such call sites must pass
     ``require_vendor_package=True``.

  D. (a fourth, seen on ``StoreKVBlock``) HALF-BUILT PACKAGE -- symbol, proto and
     tiling are present, the kernel .o and the ops-info entry are not, because
     those are built per SoC. Levels (1)-(3) all pass.
     Caught by: level (4) only. The failure otherwise moves to TILING and
     arrives as a message that never says anything is missing.

Template for adding a new operator: decide which of A-D can happen to it, and
do not rely on a level that cannot see that shape. When unsure, require
provenance -- it is the only level that separates C from "installed".
⚠ Exception: do NOT require provenance for operators that are SUPPOSED to come
from CANN (there are ~10 of those); requiring it there is a false kill.

What each level can and cannot say
----------------------------------
* (1) is exact.
* (2) is exact: dlopen/dlsym on the same libraries, in the same order, as the
  C++ side. It is done here with libc's ``dlopen``/``dlsym`` and ``RTLD_LAZY``
  -- ctypes' ``CDLL`` forces ``RTLD_NOW``, which would refuse libraries the
  C++ side loads fine.
* (3) is exact about WHERE the symbol came from. It is NOT able to say anything
  about the signature of what it found: a C shared object carries no arity.
  Callers that need the signature must read a declaration (see
  ``read_aclnn_arg_count``) and are told, in the negative case, that they were
  not able to.
* (4) is a three-valued answer -- installed / not installed / unknown -- and
  the unknown case is reported as unknown rather than folded into either side.
  It is judged on the ops-info registry KEY, which is literally the OpType
  (``OP_ADD(X)`` emits key ``X``), so no name transform is involved. Kernel
  DIRECTORY names are deliberately not used as the criterion: they are
  snake_cased and some carry an ``_apt`` suffix, so deriving one from an OpType
  would be a guess.

Everything here is host-side and read-only: no device, no runtime, no acl init.
"""

import ctypes
import functools
import json
import os
from dataclasses import dataclass, field

from vllm.logger import logger

# Library names, verbatim from op_api_common.h's GetOpApiLibName /
# GetCustOpApiLibName.
_CUST_OPAPI_LIB = "libcust_opapi.so"
_OPAPI_LIB = "libopapi.so"

# Where a symbol was found. The order is GetOpApiFuncAddr's search order.
ORIGIN_CUSTOM_OPP_PATH = "ASCEND_CUSTOM_OPP_PATH vendor package"
ORIGIN_DEFAULT_VENDOR = "ASCEND_OPP_PATH/vendors package"
ORIGIN_CANN_BUILTIN = "CANN built-in libopapi.so"
ORIGIN_UNRESOLVED = "not found in any searched library"

# The two origins that mean "our own operator package answered".
_VENDOR_ORIGINS = (ORIGIN_CUSTOM_OPP_PATH, ORIGIN_DEFAULT_VENDOR)


@functools.lru_cache(maxsize=1)
def _libdl():
    """Bind libc's dlopen/dlsym/dlerror once.

    ctypes.CDLL is not usable here: CPython ORs RTLD_NOW into the mode
    unconditionally, while the C++ side uses RTLD_LAZY. A library that the
    production path loads happily could then be reported as unloadable, and a
    guard that reports a false "missing" is worse than no guard.
    """
    libc = ctypes.CDLL(None)
    libc.dlopen.restype = ctypes.c_void_p
    libc.dlopen.argtypes = [ctypes.c_char_p, ctypes.c_int]
    libc.dlsym.restype = ctypes.c_void_p
    libc.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    libc.dlerror.restype = ctypes.c_char_p
    return libc


def environment_fingerprint() -> tuple[str, ...]:
    """The env vars every answer here depends on, read at CALL time.

    Results below are memoised on this tuple rather than on the name alone.
    A cache keyed only on the operator name would freeze the first answer into
    a constant, and the first answer is taken before anything is certain --
    ``bootstrap_custom_op_env`` prepends to ASCEND_CUSTOM_OPP_PATH while the C
    extension imports, so "the paths at import time" and "the paths at call
    time" are different values.
    """
    return tuple(
        os.environ.get(name, "")
        for name in (
            "ASCEND_CUSTOM_OPP_PATH",
            "ASCEND_OPP_PATH",
            "ASCEND_HOME_PATH",
            "ASCEND_TOOLKIT_HOME",
        )
    )


def _dlopen(path: str):
    libc = _libdl()
    libc.dlerror()  # clear
    handle = libc.dlopen(path.encode(), os.RTLD_LAZY)
    if not handle:
        err = libc.dlerror()
        return None, (err.decode(errors="replace") if err else "dlopen returned NULL")
    return handle, None


def _dlsym(handle, symbol: str) -> bool:
    libc = _libdl()
    libc.dlerror()
    return bool(libc.dlsym(ctypes.c_void_p(handle), symbol.encode()))


def _real_path(path: str) -> str:
    """op_api_common.h's real_path: empty string when the path does not exist."""
    try:
        resolved = os.path.realpath(path)
    except OSError:
        return ""
    return resolved if os.path.exists(resolved) else ""


def custom_opp_package_roots() -> list[str]:
    """The ASCEND_CUSTOM_OPP_PATH entries, in order. Read at call time.

    Not at import time: ``bootstrap_custom_op_env`` (vllm_ascend/utils.py)
    PREPENDS the in-tree vendor package to this variable while the C extension
    is being imported, so a value snapshotted at import of this module would be
    the wrong one, and would be wrong in the direction of looking cleaner.
    """
    raw = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    if not raw:
        return []
    return [entry for entry in raw.split(":") if entry]


def default_vendor_package_roots() -> list[str]:
    """The $ASCEND_OPP_PATH/vendors entries named by config.ini's load_priority.

    Mirrors get_default_custom_lib_path() in op_api_common.h, including the
    detail that an unreadable or load_priority-less config.ini yields nothing.
    """
    opp_path = os.environ.get("ASCEND_OPP_PATH")
    if not opp_path:
        return []
    vendors = os.path.join(opp_path, "vendors")
    config = _real_path(os.path.join(vendors, "config.ini"))
    if not config or not os.path.isfile(config):
        return []
    line = ""
    try:
        with open(config, encoding="utf-8", errors="replace") as handle:
            for candidate in handle:
                if candidate.startswith("load_priority="):
                    line = candidate.strip()
                    break
    except OSError:
        return []
    if not line:
        return []
    names = [name for name in line[len("load_priority="):].split(",") if name]
    roots = []
    for name in names:
        root = _real_path(os.path.join(vendors, name))
        if root:
            roots.append(root)
    return roots


def aclnn_search_order() -> list[tuple[str, str, str]]:
    """(origin, package_root, library_path) triples, in GetOpApiFuncAddr order.

    package_root is "" for the CANN built-in entry: it is not a vendor package
    and has no ops-info registry of ours to look in.
    """
    order: list[tuple[str, str, str]] = []
    for root in custom_opp_package_roots():
        lib = _real_path(os.path.join(root, "op_api", "lib", _CUST_OPAPI_LIB))
        if lib:
            order.append((ORIGIN_CUSTOM_OPP_PATH, root, lib))
    for root in default_vendor_package_roots():
        lib = _real_path(os.path.join(root, "op_api", "lib", _CUST_OPAPI_LIB))
        if lib:
            order.append((ORIGIN_DEFAULT_VENDOR, root, lib))
    # The C++ side dlopens the bare soname and lets the loader find it.
    order.append((ORIGIN_CANN_BUILTIN, "", _OPAPI_LIB))
    return order


@dataclass(frozen=True)
class SymbolResolution:
    """Where one aclnn symbol resolved, and what was searched to find out."""

    symbol: str
    origin: str
    package_root: str
    library: str
    searched: tuple[tuple[str, str], ...] = ()
    load_errors: tuple[tuple[str, str], ...] = ()

    @property
    def found(self) -> bool:
        return self.origin != ORIGIN_UNRESOLVED


def resolve_aclnn_symbol(symbol: str) -> SymbolResolution:
    """dlopen/dlsym the way GetOpApiFuncAddr does, and say WHERE it landed."""
    return _resolve_aclnn_symbol_cached(symbol, environment_fingerprint())


@functools.cache
def _resolve_aclnn_symbol_cached(symbol: str, _fingerprint: tuple[str, ...]) -> SymbolResolution:
    # Memoised because a call site can run this once per decode step, and the
    # handles opened below are never dlclose()d -- on purpose, the production
    # path keeps them open too, but an uncached probe would take a fresh
    # reference on every token.
    searched: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []
    for origin, root, lib in aclnn_search_order():
        searched.append((origin, lib))
        handle, err = _dlopen(lib)
        if handle is None:
            errors.append((lib, err or "unknown dlopen failure"))
            continue
        if _dlsym(handle, symbol):
            return SymbolResolution(
                symbol=symbol,
                origin=origin,
                package_root=root,
                library=lib,
                searched=tuple(searched),
                load_errors=tuple(errors),
            )
    return SymbolResolution(
        symbol=symbol,
        origin=ORIGIN_UNRESOLVED,
        package_root="",
        library="",
        searched=tuple(searched),
        load_errors=tuple(errors),
    )


@dataclass(frozen=True)
class AclnnApiResolution:
    """Both entry points of one aclnn API. EXEC_NPU_CMD needs both."""

    api: str
    entry: SymbolResolution
    workspace: SymbolResolution

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(r.symbol for r in (self.entry, self.workspace) if not r.found)

    @property
    def resolved(self) -> bool:
        return not self.missing

    @property
    def origin(self) -> str:
        if not self.resolved:
            return ORIGIN_UNRESOLVED
        return self.entry.origin

    @property
    def split(self) -> bool:
        """True when the two entry points came out of DIFFERENT libraries.

        GetOpApiFuncAddr resolves them independently, so this is possible, and
        it is the worst shape of all: half of one implementation, half of
        another, with no error anywhere.
        """
        return self.resolved and self.entry.library != self.workspace.library

    @property
    def package_root(self) -> str:
        return self.entry.package_root

    @property
    def library(self) -> str:
        return self.entry.library

    @property
    def searched(self) -> tuple[tuple[str, str], ...]:
        return self.entry.searched


def resolve_aclnn_api(api: str) -> AclnnApiResolution:
    return AclnnApiResolution(
        api=api,
        entry=resolve_aclnn_symbol(api),
        workspace=resolve_aclnn_symbol(api + "GetWorkspaceSize"),
    )


# --------------------------------------------------------------------------
# Level (4): does the package that answered actually contain a kernel?
# --------------------------------------------------------------------------

KERNEL_INSTALLED = "installed"
KERNEL_ABSENT = "absent"
KERNEL_UNKNOWN = "unknown"


@dataclass(frozen=True)
class KernelRegistration:
    """Three-valued: the unknown case is reported, not folded into either side."""

    op_type: str
    package_root: str
    state: str
    registries: tuple[str, ...] = ()
    registries_with_key: tuple[str, ...] = ()
    reason: str = ""


def _ops_info_registries(package_root: str) -> list[str]:
    """Every ops-info registry in an operator package (AI Core and AICPU)."""
    found = []
    aic = os.path.join(package_root, "op_impl", "ai_core", "tbe", "config")
    if os.path.isdir(aic):
        for soc in sorted(os.listdir(aic)):
            soc_dir = os.path.join(aic, soc)
            if not os.path.isdir(soc_dir):
                continue
            for name in sorted(os.listdir(soc_dir)):
                if name.endswith("ops-info.json"):
                    found.append(os.path.join(soc_dir, name))
    aicpu = os.path.join(package_root, "op_impl", "cpu", "config", "cust_aicpu_kernel.json")
    if os.path.isfile(aicpu):
        found.append(aicpu)
    return found


def _registry_has_op_type(path: str, op_type: str) -> bool:
    """The registry key IS the OpType -- exact string, no transform."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return False
    if isinstance(data, dict):
        if op_type in data:
            return True
        # AICPU registries nest the op list one level down.
        for value in data.values():
            if isinstance(value, dict) and op_type in value:
                return True
            if isinstance(value, list) and op_type in value:
                return True
    return False


def kernel_registration(op_type: str, package_root: str) -> KernelRegistration:
    return _kernel_registration_cached(op_type, package_root, environment_fingerprint())


@functools.cache
def _kernel_registration_cached(
    op_type: str, package_root: str, _fingerprint: tuple[str, ...]
) -> KernelRegistration:
    """Is ``op_type`` registered in ``package_root``'s ops-info registries?

    ABSENT only when at least one registry was readable and NONE of them has
    the key. "I could not read any registry" is UNKNOWN, never ABSENT: an
    absence that is really an unreadable instrument reads identically to a real
    absence, and only one of the two is a defect.
    """
    if not package_root:
        return KernelRegistration(
            op_type=op_type,
            package_root=package_root,
            state=KERNEL_UNKNOWN,
            reason=(
                "the symbol did not come from an operator package (no package root), "
                "so there is no ops-info registry of ours to consult"
            ),
        )
    registries = _ops_info_registries(package_root)
    if not registries:
        return KernelRegistration(
            op_type=op_type,
            package_root=package_root,
            state=KERNEL_UNKNOWN,
            reason=f"no ops-info registry found under {package_root}",
        )
    with_key = tuple(path for path in registries if _registry_has_op_type(path, op_type))
    return KernelRegistration(
        op_type=op_type,
        package_root=package_root,
        state=KERNEL_INSTALLED if with_key else KERNEL_ABSENT,
        registries=tuple(registries),
        registries_with_key=with_key,
        reason="",
    )


# --------------------------------------------------------------------------
# Signatures. Only a DECLARATION can answer this; a .so cannot.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AclnnArgCount:
    api: str
    count: int | None
    header: str = ""
    params: tuple[str, ...] = ()
    reason: str = ""

    @property
    def known(self) -> bool:
        return self.count is not None


def _split_top_level(text: str) -> list[str]:
    parts, depth, current = [], 0, []
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return [part.strip() for part in parts if part.strip()]


def _parse_arg_count(text: str, token: str) -> tuple[int, tuple[str, ...]] | None:
    """Count the operator arguments of ``<token>(...)`` in a header.

    The last two parameters of every aclnn *GetWorkspaceSize declaration are
    ``workspaceSize`` and ``executor``; EXEC_NPU_CMD appends those (and the
    stream) itself, so they are not part of what a call site passes.
    """
    index = 0
    while True:
        index = text.find(token, index)
        if index < 0:
            return None
        after = index + len(token)
        while after < len(text) and text[after] in " \t\n":
            after += 1
        if after >= len(text) or text[after] != "(":
            index = after
            continue
        depth, end = 0, after
        for end in range(after, len(text)):
            if text[end] == "(":
                depth += 1
            elif text[end] == ")":
                depth -= 1
                if depth == 0:
                    break
        else:
            return None
        params = _split_top_level(text[after + 1:end])
        names = tuple(part.split()[-1].lstrip("*") for part in params if part.split())
        trailing = 0
        for name in reversed(names):
            if name in ("workspaceSize", "executor"):
                trailing += 1
            else:
                break
        if trailing == 0:
            # Not a GetWorkspaceSize declaration shape; keep looking.
            index = end
            continue
        return len(params) - trailing, names[: len(params) - trailing]


def read_aclnn_arg_count(api: str, include_roots: tuple[str, ...] = ()) -> AclnnArgCount:
    return _read_aclnn_arg_count_cached(api, tuple(include_roots), environment_fingerprint())


@functools.cache
def _read_aclnn_arg_count_cached(
    api: str, include_roots: tuple[str, ...], _fingerprint: tuple[str, ...]
) -> AclnnArgCount:
    """How many operator arguments does the INSTALLED declaration of ``api`` take?

    Found by CONTENT, not by file name: the header is located by searching for
    the exact token ``<api>GetWorkspaceSize`` inside ``aclnnop`` include trees.
    Deriving ``aclnn_rms_norm_dynamic_quant.h`` from ``aclnnRmsNormDynamicQuant``
    would be a guess about a naming convention, and a guess that misses reads
    exactly like "the header is not installed".

    Returns count=None with a reason when no declaration could be read. Callers
    must say that they could not tell, not pretend they checked.
    """
    token = api + "GetWorkspaceSize"
    roots: list[str] = []
    for root in include_roots:
        if root:
            roots.append(root)
    for var in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH"):
        base = os.environ.get(var)
        if base:
            roots.append(os.path.join(base, "include"))
    seen, headers = set(), []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if os.path.basename(dirpath) != "aclnnop" and "aclnnop" not in dirpath.split(os.sep):
                continue
            for name in filenames:
                if not name.endswith(".h"):
                    continue
                path = os.path.join(dirpath, name)
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                headers.append(path)
    if not headers:
        return AclnnArgCount(
            api=api,
            count=None,
            reason=(
                "no aclnnop include directory was found under "
                f"{roots or ['(nothing to search: ASCEND_HOME_PATH/ASCEND_TOOLKIT_HOME/ASCEND_OPP_PATH unset)']}"
            ),
        )
    for path in sorted(headers):
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError:
            continue
        if token not in text:
            continue
        parsed = _parse_arg_count(text, token)
        if parsed is None:
            continue
        count, names = parsed
        return AclnnArgCount(api=api, count=count, header=path, params=names)
    return AclnnArgCount(
        api=api,
        count=None,
        reason=(
            f"searched {len(headers)} aclnnop header(s) and none declares {token}; "
            "the installed CANN does not publish a declaration for this API"
        ),
    )


# --------------------------------------------------------------------------
# The guard.
# --------------------------------------------------------------------------


@dataclass
class OpAvailability:
    """Everything the four levels found, plus which one refuses."""

    torch_name: str
    aclnn_api: str
    op_type: str
    torch_namespace: str = ""
    op: object = None
    resolution: AclnnApiResolution | None = None
    kernel: KernelRegistration | None = None
    arity: AclnnArgCount | None = None
    unchecked: list[str] = field(default_factory=list)


def _searched_text(resolution: AclnnApiResolution) -> str:
    if not resolution.searched:
        return "(nothing: no operator package and no libopapi.so were reachable)"
    return "; ".join(f"{origin} -> {lib}" for origin, lib in resolution.searched)


def torch_op_namespaces(torch_module, torch_npu_module):
    """(name, namespace) pairs in the order the call sites try them."""
    return (
        ("torch.ops._C_ascend", getattr(torch_module.ops, "_C_ascend", None)),
        ("torch.ops.custom", getattr(torch_module.ops, "custom", None)),
        ("torch_npu", torch_npu_module),
    )


def find_torch_op(torch_module, torch_npu_module, op_name: str):
    """Level (1). Returns (op, namespace_name) or (None, "")."""
    for label, namespace in torch_op_namespaces(torch_module, torch_npu_module):
        if namespace is None:
            continue
        op = getattr(namespace, op_name, None)
        if op is not None:
            return op, label
    return None, ""


def check_aclnn_op(
    torch_module,
    torch_npu_module,
    *,
    torch_name: str,
    aclnn_api: str,
    op_type: str,
    expected_arg_count: int | None = None,
    require_vendor_package: bool = False,
) -> tuple[OpAvailability, str]:
    """Run every level that CAN be run here. Returns (report, failure_message).

    ``failure_message`` is "" when nothing conclusive went wrong. It never says
    "everything is fine": levels that could not be judged are listed in
    ``report.unchecked`` and the callers log them.
    """
    report = OpAvailability(torch_name=torch_name, aclnn_api=aclnn_api, op_type=op_type)

    op, namespace = find_torch_op(torch_module, torch_npu_module, torch_name)
    report.torch_namespace = namespace
    report.op = op
    if op is None:
        return report, (
            f"torch op '{torch_name}' is NOT REGISTERED in torch.ops._C_ascend, "
            f"torch.ops.custom or torch_npu.\n"
            f"  ATTRIBUTION: the vllm-ascend C extension. Either "
            f"`import vllm_ascend.vllm_ascend_C` failed (look for an earlier "
            f"'Failed to register custom ops' warning) or this build's "
            f"csrc/torch_binding.cpp has no ops.def(\"{torch_name}\", ...).\n"
            f"  This says NOTHING about the CANN operator package: torch schemas are "
            f"registered unconditionally, so a registered name is not evidence that the "
            f"operator is installed, and this branch is not evidence that it is not.\n"
            f"  WHO TO ASK: whoever builds vllm-ascend."
        )

    if namespace != "torch.ops._C_ascend":
        # The op came from torch_npu (or the never-exercised torch.ops.custom
        # namespace). Those resolve their own aclnn entry points; our search
        # order and our ops-info registry say nothing about them.
        report.unchecked.append(
            f"aclnn resolution and kernel registration were NOT checked: "
            f"'{torch_name}' was provided by {namespace}, not by vllm-ascend's "
            f"_C_ascend extension, so {aclnn_api}/{op_type} is not the symbol it calls."
        )
        return report, ""

    resolution = resolve_aclnn_api(aclnn_api)
    report.resolution = resolution

    if not resolution.resolved:
        return report, (
            f"torch op '{torch_name}' IS registered, but its aclnn entry point(s) "
            f"{', '.join(resolution.missing)} DO NOT RESOLVE in any library that "
            f"GetOpApiFuncAddr searches.\n"
            f"  searched, in order: {_searched_text(resolution)}\n"
            f"  ATTRIBUTION: the CANN OPERATOR PACKAGE on this machine. {op_type} was "
            f"not compiled into it -- check the operator's AddConfig SoC list and the "
            f"COMPILED_OPS selection of the package build.\n"
            f"  This is NOT a torch registration problem and NOT a Python problem; "
            f"torch registration already passed one line above.\n"
            f"  WHO TO ASK: whoever builds/installs the operator package "
            f"(ASCEND_CUSTOM_OPP_PATH / opp vendors).\n"
            f"  Without this guard the call would have reached EXEC_NPU_CMD and died "
            f"there with 'aclnnX or aclnnXGetWorkspaceSize not in libopapi.so, or "
            f"libopapi.so not found', which names the wrong library and sends you to "
            f"the wrong person."
        )

    if resolution.split:
        return report, (
            f"torch op '{torch_name}' resolves, but its two aclnn entry points came out "
            f"of DIFFERENT libraries: {resolution.entry.symbol} from "
            f"{resolution.entry.library} ({resolution.entry.origin}) and "
            f"{resolution.workspace.symbol} from {resolution.workspace.library} "
            f"({resolution.workspace.origin}).\n"
            f"  ATTRIBUTION: two operator packages are on the search path and each "
            f"supplies half of this API. Nothing downstream would report this; the call "
            f"would size a workspace with one implementation and run another.\n"
            f"  WHO TO ASK: whoever set ASCEND_CUSTOM_OPP_PATH / the opp vendors "
            f"load_priority on this machine."
        )

    if require_vendor_package and resolution.origin == ORIGIN_CANN_BUILTIN:
        arity = read_aclnn_arg_count(aclnn_api)
        report.arity = arity
        if arity.known and expected_arg_count is not None and arity.count == expected_arg_count:
            arity_line = (
                f"  signature: the installed declaration ({arity.header}) takes "
                f"{arity.count} operator arguments, which MATCHES the {expected_arg_count} "
                f"this call site passes. Matching arity is not proof of the same "
                f"operator -- argument ORDER and MEANING are not checked by anything "
                f"here -- so the mismatch of provenance above still stands."
            )
        elif arity.known:
            arity_line = (
                f"  signature: the installed declaration ({arity.header}) takes "
                f"{arity.count} operator arguments "
                f"({', '.join(arity.params) or 'unnamed'}), while this call site passes "
                f"{expected_arg_count}. Calling it would push "
                f"{expected_arg_count} arguments at a {arity.count}-argument ABI."
            )
        else:
            arity_line = (
                f"  signature: COULD NOT BE DETERMINED -- {arity.reason}. A shared "
                f"object carries no arity, so this check needs a declaration and there "
                f"is none to read. Treat the argument count as UNKNOWN, not as matching."
            )
        return report, (
            f"torch op '{torch_name}' resolves, but NOT from an operator package of "
            f"ours: {aclnn_api} came from {resolution.library} ({resolution.origin}).\n"
            f"  searched, in order: {_searched_text(resolution)}\n"
            f"  ATTRIBUTION: WHAT YOU GOT IS PROBABLY CANN'S OWN OPERATOR OF THE SAME "
            f"NAME, WITH A DIFFERENT SIGNATURE. {op_type} is an operator vllm-ascend "
            f"ships in csrc; it is absent from every operator package on this search "
            f"path, so dlsym fell through to CANN's built-in library, which exports the "
            f"same symbol. dlsym succeeds, the TORCH_CHECK in EXEC_NPU_CMD passes, and "
            f"nothing downstream compares signatures.\n"
            f"{arity_line}\n"
            f"  WHO TO ASK: whoever builds the operator package -- {op_type} needs to be "
            f"compiled for this SoC. Do NOT read the successful dlsym as 'it is "
            f"installed'."
        )

    kernel = kernel_registration(op_type, resolution.package_root)
    report.kernel = kernel

    if kernel.state == KERNEL_ABSENT:
        return report, (
            f"torch op '{torch_name}' resolves and {aclnn_api} was found in "
            f"{resolution.library}, so every name-based check passes -- but the package "
            f"that supplied it has NO KERNEL for {op_type}.\n"
            f"  {op_type} is absent from all {len(kernel.registries)} ops-info "
            f"registr{'y' if len(kernel.registries) == 1 else 'ies'} of "
            f"{kernel.package_root}: {', '.join(kernel.registries)}\n"
            f"  ATTRIBUTION: a HALF-BUILT operator package. aclnn stubs, prototypes and "
            f"tiling are built SoC-independently; kernels are built per SoC. When a SoC "
            f"is dropped from the operator's AddConfig list, the symbol still ships and "
            f"the kernel does not.\n"
            f"  WHO TO ASK: whoever builds the operator package. The name being "
            f"exported is NOT evidence the operator is installed.\n"
            f"  Without this guard the failure would have moved to operator selection / "
            f"TILING and arrived as 'call {aclnn_api} failed, detail: ...' -- a message "
            f"that never says anything is missing."
        )

    if kernel.state == KERNEL_UNKNOWN:
        report.unchecked.append(
            f"kernel registration for {op_type} is UNKNOWN: {kernel.reason}. "
            f"The aclnn symbols resolved from {resolution.library}, but this run could "
            f"not confirm that a kernel for this SoC exists."
        )
    else:
        report.unchecked.append(
            f"{op_type} is registered in {', '.join(kernel.registries_with_key)}, but "
            f"which SoC this process will actually run on was not determined here (no "
            f"device was touched), and the resolved library's argument order/meaning "
            f"is not verifiable from a shared object."
        )
    return report, ""


def require_aclnn_op(
    torch_module,
    torch_npu_module,
    *,
    torch_name: str,
    aclnn_api: str,
    op_type: str,
    feature: str,
    expected_arg_count: int | None = None,
    require_vendor_package: bool = False,
):
    """Raise unless the path is walkable. Returns the torch op when it is.

    The three refusals this can produce are deliberately different texts
    pointing at three different people: the vllm-ascend build, the operator
    package build, and whoever configured the operator package search path.
    """
    report, failure = check_aclnn_op(
        torch_module,
        torch_npu_module,
        torch_name=torch_name,
        aclnn_api=aclnn_api,
        op_type=op_type,
        expected_arg_count=expected_arg_count,
        require_vendor_package=require_vendor_package,
    )
    if failure:
        raise RuntimeError(f"{feature} cannot run on this installation.\n  {failure}")
    for note in report.unchecked:
        logger.warning_once("%s: %s", feature, note)
    return report.op
