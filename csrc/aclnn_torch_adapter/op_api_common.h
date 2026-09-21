// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OP_API_COMMON_ADAPTER
#define OP_API_COMMON_ADAPTER

#include <fstream>
// 🔴 显式列出来，不靠传递包含：DSV41-ACL4BIT-DESC-PROD 这块用到
//    ostringstream/mutex/set，而**探针那一块会被 revert 掉**，
//    今天是它把这些头顺带带进来的。靠传递包含的话，回退的那一刻
//    这块才第一次编不过 —— 而那时正好是没人预期会有编译错误的时候。
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <torch/types.h>
#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <functional>
#include <type_traits>
#include <vector>

#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "NPUBridge.h"
#include "NPUStorageImpl.h"

#define NPU_NAME_SPACE at_npu::native
using namespace at;

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;

typedef aclTensor *(*_aclCreateTensor)(
    const int64_t *view_dims, uint64_t view_dims_num, aclDataType data_type,
    const int64_t *stride, int64_t offset, aclFormat format,
    const int64_t *storage_dims, uint64_t storage_dims_num, void *tensor_data);
typedef aclScalar *(*_aclCreateScalar)(void *value, aclDataType data_type);
typedef aclIntArray *(*_aclCreateIntArray)(const int64_t *value, uint64_t size);
typedef aclFloatArray *(*_aclCreateFloatArray)(const float *value,
                                               uint64_t size);
typedef aclBoolArray *(*_aclCreateBoolArray)(const bool *value, uint64_t size);
typedef aclTensorList *(*_aclCreateTensorList)(const aclTensor *const *value,
                                               uint64_t size);

typedef int (*_aclDestroyTensor)(const aclTensor *tensor);
typedef int (*_aclDestroyScalar)(const aclScalar *scalar);
typedef int (*_aclDestroyIntArray)(const aclIntArray *array);
typedef int (*_aclDestroyFloatArray)(const aclFloatArray *array);
typedef int (*_aclDestroyBoolArray)(const aclBoolArray *array);
typedef int (*_aclDestroyTensorList)(const aclTensorList *array);

constexpr int kHashBufSize = 8192;
constexpr int kHashBufMaxSize = kHashBufSize + 1024;
extern thread_local char g_hashBuf[kHashBufSize];
extern thread_local int g_hashOffset;

#ifdef MMCV_WITH_XLA
#define DEVICE_TYPE at_npu::key::NativeDeviceType
#else
#define DEVICE_TYPE c10::DeviceType::PrivateUse1
#endif

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_) \
  _(at::ScalarType::Byte, ACL_UINT8)                \
  _(at::ScalarType::Char, ACL_INT8)                 \
  _(at::ScalarType::Short, ACL_INT16)               \
  _(at::ScalarType::Int, ACL_INT32)                 \
  _(at::ScalarType::Long, ACL_INT64)                \
  _(at::ScalarType::Half, ACL_FLOAT16)              \
  _(at::ScalarType::Float, ACL_FLOAT)               \
  _(at::ScalarType::Double, ACL_DOUBLE)             \
  _(at::ScalarType::ComplexHalf, ACL_COMPLEX32)     \
  _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)    \
  _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)  \
  _(at::ScalarType::Bool, ACL_BOOL)                 \
  _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)        \
  _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)       \
  _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)       \
  _(at::ScalarType::BFloat16, ACL_BF16)             \
  _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)     \
  _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)     \
  _(at::ScalarType::Bits1x8, ACL_DT_UNDEFINED)      \
  _(at::ScalarType::Bits2x4, ACL_DT_UNDEFINED)      \
  _(at::ScalarType::Bits4x2, ACL_DT_UNDEFINED)      \
  _(at::ScalarType::Bits8, ACL_DT_UNDEFINED)        \
  _(at::ScalarType::Bits16, ACL_DT_UNDEFINED)       \
  _(at::ScalarType::Float8_e5m2, ACL_FLOAT8_E5M2)   \
  _(at::ScalarType::Float8_e4m3fn, ACL_FLOAT8_E4M3FN) \
  _(at::ScalarType::Float8_e5m2fnuz, ACL_DT_UNDEFINED) \
  _(at::ScalarType::Float8_e4m3fnuz, ACL_DT_UNDEFINED) \
  _(at::ScalarType::UInt16, ACL_UINT16)             \
  _(at::ScalarType::UInt32, ACL_UINT32)             \
  _(at::ScalarType::UInt64, ACL_UINT64)             \
  _(at::ScalarType::UInt1, ACL_UINT1)               \
  _(at::ScalarType::UInt2, ACL_DT_UNDEFINED)        \
  _(at::ScalarType::UInt3, ACL_DT_UNDEFINED)        \
  _(at::ScalarType::UInt4, ACL_DT_UNDEFINED)        \
  _(at::ScalarType::UInt5, ACL_DT_UNDEFINED)        \
  _(at::ScalarType::UInt6, ACL_DT_UNDEFINED)        \
  _(at::ScalarType::UInt7, ACL_DT_UNDEFINED)        \
  _(at::ScalarType::Int1, ACL_DT_UNDEFINED)         \
  _(at::ScalarType::Int2, ACL_DT_UNDEFINED)         \
  _(at::ScalarType::Int3, ACL_DT_UNDEFINED)         \
  _(at::ScalarType::Int4, ACL_INT4)                 \
  _(at::ScalarType::Int5, ACL_DT_UNDEFINED)         \
  _(at::ScalarType::Int6, ACL_DT_UNDEFINED)         \
  _(at::ScalarType::Int7, ACL_DT_UNDEFINED)         \
  _(at::ScalarType::Float8_e8m0fnu, ACL_FLOAT8_E8M0) \
  _(at::ScalarType::Float4_e2m1fn_x2, ACL_FLOAT4_E2M1) \
  _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)    \
  _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable
    [static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
        AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

#define GET_OP_API_FUNC(apiName) \
  reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

#define MEMCPY_TO_BUF(data_expression, size_expression)               \
  if (g_hashOffset + (size_expression) > kHashBufSize) {              \
    g_hashOffset = kHashBufMaxSize;                                   \
    return;                                                           \
  }                                                                   \
  memcpy(g_hashBuf + g_hashOffset, data_expression, size_expression); \
  g_hashOffset += size_expression;

bool IsOpInputBaseFormat(const at::Tensor &tensor)
{
    if (!tensor.is_privateuseone()) {
        return true;
    }
    const auto format = vllm_ascend::NPUBridge::GetNpuStorageImplDesc(tensor).npu_format_;
    return (format == ACL_FORMAT_ND) || (format == ACL_FORMAT_NCHW) || (format == ACL_FORMAT_NHWC) ||
        (format == ACL_FORMAT_NCDHW);
}

static std::vector<std::string> split_str(std::string s, const std::string &del)
{
    int end = s.find(del);
    std::vector<std::string> path_list;
    while (end != -1) {
        path_list.push_back(s.substr(0, end));
        s.erase(s.begin(), s.begin() + end + 1);
        end = s.find(del);
    }
    path_list.push_back(s);
    return path_list;
}

static bool is_file_exist(const std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX) {
        return false;
    }
    return (access(path.c_str(), F_OK) == 0) ? true : false;
}

inline  std::string real_path(const std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX) {
        return "";
    }
    char realPath[PATH_MAX] = {0};
    if (realpath(path.c_str(), realPath) == nullptr) {
        return "";
    }
    return std::string(realPath);
}

inline std::vector<std::string> get_custom_lib_path()
{
    char *ascend_custom_opppath = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    std::vector<std::string> custom_lib_path_list;

    if (ascend_custom_opppath == nullptr) {
        return std::vector<std::string>();
    }

    std::string ascend_custom_opppath_str(ascend_custom_opppath);
    // split string with ":"
    custom_lib_path_list = split_str(ascend_custom_opppath_str, ":");
    if (custom_lib_path_list.empty()) {
        return std::vector<std::string>();
    }
    for (auto &it : custom_lib_path_list) {
        it = it + "/op_api/lib/";
    }

    return custom_lib_path_list;
}

inline std::vector<std::string> get_default_custom_lib_path()
{
    char *ascend_opp_path = std::getenv("ASCEND_OPP_PATH");
    std::vector<std::string> default_vendors_list;

    if (ascend_opp_path == nullptr) {
        return std::vector<std::string>();
    }

    std::string vendors_path(ascend_opp_path);
    vendors_path = vendors_path + "/vendors";
    std::string vendors_config_file = real_path(vendors_path + "/config.ini");
    if (vendors_config_file.empty()) {
        return std::vector<std::string>();
    }

    if (!is_file_exist(vendors_config_file)) {
        return std::vector<std::string>();
    }

    std::ifstream ifs(vendors_config_file);
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find("load_priority=") == 0) {
            break;
        }
    }
    std::string head = "load_priority=";
    line.erase(0, head.length());

    // split string with ","
    default_vendors_list = split_str(line, ",");
    if (default_vendors_list.empty()) {
        return std::vector<std::string>();
    }
    for (auto &it : default_vendors_list) {
        it = real_path(vendors_path + "/" + it + "/op_api/lib/");
    }

    return default_vendors_list;
}

const std::vector<std::string> g_custom_lib_path = get_custom_lib_path();
const std::vector<std::string> g_default_custom_lib_path = get_default_custom_lib_path();

inline const char *GetOpApiLibName(void) { return "libopapi.so"; }

inline const char *GetCustOpApiLibName(void) { return "libcust_opapi.so"; }

inline void *GetOpApiFuncAddrInLib(void *handler, const char *libName,
                                   const char *apiName) {
  auto funcAddr = dlsym(handler, apiName);
  return funcAddr;
}

inline void *GetOpApiLibHandler(const char *libName) {
  auto handler = dlopen(libName, RTLD_LAZY);
  return handler;
}

inline void *GetOpApiFuncAddr(const char *apiName)
{
    if (!g_custom_lib_path.empty()) {
        for (auto &it : g_custom_lib_path) {
            auto cust_opapi_lib = real_path(it + "/" + GetCustOpApiLibName());
            if (cust_opapi_lib.empty()) {
                continue;
            }
            auto custOpApiHandler = GetOpApiLibHandler(cust_opapi_lib.c_str());
            if (custOpApiHandler != nullptr) {
                auto funcAddr =
                    GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
                if (funcAddr != nullptr) {
                    return funcAddr;
                }
            }
        }
    }

    if (!g_default_custom_lib_path.empty()) {
        for (auto &it : g_default_custom_lib_path) {
            auto default_cust_opapi_lib = real_path(it + "/" + GetCustOpApiLibName());
            if (default_cust_opapi_lib.empty()) {
                continue;
            }
            auto custOpApiHandler = GetOpApiLibHandler(default_cust_opapi_lib.c_str());
            if (custOpApiHandler != nullptr) {
                auto funcAddr =
                    GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
                if (funcAddr != nullptr) {
                    return funcAddr;
                }
            }
        }
    }

    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
}

inline c10::Scalar ConvertTensorToScalar(const at::Tensor &tensor) {
  c10::Scalar expScalar;
  const at::Tensor *aclInput = &tensor;
  if (aclInput->scalar_type() == at::ScalarType::Double) {
    double value = *(double *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Long) {
    int64_t value = *(int64_t *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Float) {
    float value = *(float *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Int) {
    int value = *(int *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Half) {
    c10::Half value = *(c10::Half *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Bool) {
    int8_t value = *(int8_t *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::ComplexDouble) {
    c10::complex<double> value = *(c10::complex<double> *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::ComplexFloat) {
    c10::complex<float> value = *(c10::complex<float> *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::BFloat16) {
    c10::BFloat16 value = *(c10::BFloat16 *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  }
  return expScalar;
}

inline at::Tensor CopyTensorHostToDevice(const at::Tensor &cpu_tensor) {
  at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
  int deviceIndex = 0;
  return cpuPinMemTensor.to(c10::Device(DEVICE_TYPE, deviceIndex),
                            cpuPinMemTensor.scalar_type(), true, true);
}

inline at::Tensor CopyScalarToDevice(const c10::Scalar &cpu_scalar,
                                     at::ScalarType scalar_data_type) {
  return CopyTensorHostToDevice(
      scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

// ---------------------------------------------------------------------------
// 4-bit (MXFP4 / INT4) descriptors.
//
// acl counts a 4-bit tensor in **logical nibbles**; torch counts
// ``Float4_e2m1fn_x2`` in **pairs** and reports ``itemsize() == 1``. A
// [T, H, 128] index-K row occupies 64 bytes, and without the correction below
// acl is told the last dimension is 64:
//
//     Parameter q of QuantLightningIndexerV2 has incorrect shape [256,32,64].
//     Reason: The head dim of q only supports 128.
//
// 🔴 There are **three** independent places the count leaks, and fixing one is
// not fixing the operator:
//   (1) ``sizes()``/``strides()``  -> the view dims handed to aclCreateTensor
//   (2) ``storageDims``            -> a *separate* argument, also derived from
//                                     ``itemsize`` (nbytes / itemsize)
//   (3) ``storage_offset()``       -> acl expresses it in nibbles too
// (1) and (2) are different parameters of ``aclCreateTensor``; getting only
// one right produces an out-of-range read rather than a clean refusal.
//
// ⚠ The non-base-format branch takes ``storage_sizes_`` from the NPU desc and
// never divides by itemsize, so this correction does **not** apply to it and
// must not be applied blindly -- see the refusal in ConvertType.
constexpr int64_t kAclNibblesPerByte = 2;

inline bool IsAclFourBitDtype(aclDataType t) {
  // 🔴 ACL_FLOAT8_E8M0 (37) is deliberately absent: the MXFP4 scale plane is
  // one byte per element, not sub-byte. The two planes are asymmetric and
  // exactly one of them is packed; doubling the scale would describe twice as
  // many exponents as exist.
  //
  // 🔴 ``ACL_INT4`` and ``ACL_FLOAT4_E1M2`` are **also** absent, deliberately.
  // They are in the dtype table, but measured today: zero callers hand this
  // binding an Int4 tensor (``dtype=torch.int4`` / ``.view(torch.int4)`` in
  // ``vllm_ascend/**.py``: **0**; positive control ``ScalarType::Float4_e2m1fn_x2``
  // in csrc: 3). Widening the branch to a path nobody exercises would change
  // a route that works today and is checked by nobody -- and it would be
  // "half a fix" for whoever later starts using it, since none of the
  // acceptance criteria below cover it. ConvertType refuses them instead, so
  // the first caller finds out at once rather than through wrong values.
  return t == ACL_FLOAT4_E2M1;
}

// Describe a 4-bit tensor in acl's units: **logical nibble elements**.
//
// 🔴 This is the **element semantics**, not the rule CANN's wheel uses.
//    The wheel doubles exactly ``stride[rank-2]``. At rank 2 that is the row
//    pitch and is right; at our rank 4 that axis has extent 1 and never
//    participates in addressing, while the two strides that do are left in
//    **byte** units. ABC measured the consequence:
//
//      torch    [3,4,1,64]  strides=[272,64,64,1]   only axis 0 non-contiguous
//      wheel    ACL strides=[272, 64,128,1]   🔴 axes 0 AND 1 non-contiguous
//                 -> acl tries to make it contiguous -> ViewCopy on fp4 is
//                    unsupported -> EZ1001 (observed)
//      element  ACL strides=[544,128,128,1]   only axis 0 non-contiguous
//                 -> the doc's "仅 PA_BBND 场景下 0 轴支持非连续"档
//
// ⚠ **Element semantics is NOT itself verified.** The reason for preferring it
//    is that the wheel rule has been **falsified by measurement** -- it emits a
//    descriptor that violates the operator's own contiguity requirement.
//    Between "known bad" and "unverified", the default belongs to the
//    unverified one. Confirmation still requires the element arm to produce a
//    result and match a CPU reference element by element.
//
// The rule:
//    last size      x2
//    last stride    = 1      🔴 **not** x2 -- adjacent fp4 elements share a
//                            byte, so in element units they are 1 apart
//    other strides  x2
//    storage_offset x2       (done at the call site)
//    storageDims    x2       (done at the call site)
inline void ApplyFourBitViewDims(const at::Tensor &t,
                                 c10::SmallVector<int64_t, 5> &shape,
                                 c10::SmallVector<int64_t, 5> &stride) {
  const int64_t nDim = static_cast<int64_t>(t.sizes().size());
  if (nDim <= 0) {
    return;
  }
  const int64_t last = nDim - 1;

  // 🔴 Two premises, as raises rather than comments. We still have **no
  // documented source** for "the packed axis is the last one" -- the doc gives
  // shape semantics and says nothing about strides. Making the premises throw
  // does not supply that source; it makes their violation **loud** instead of
  // silently producing a descriptor that reads the wrong elements.
  TORCH_CHECK(t.strides()[last] == 1,
              "DSV41-ACL4BIT-VIEWDIMS: the 4-bit tensor's last stride is ",
              t.strides()[last], ", not 1. The nibble correction assumes the "
              "**last axis is the packed one** and that adjacent elements share "
              "a byte; with a non-unit last stride neither holds, and the "
              "descriptor would address elements that are not there. shape=",
              t.sizes(), " stride=", t.strides());
  if (nDim >= 2) {
    // The packed axis must also be the fastest-varying one. If some other axis
    // had stride 1 as well, "which axis is packed" stops being decidable from
    // the strides, and picking the last one would be a guess.
    for (int64_t i = 0; i < last; ++i) {
      TORCH_CHECK(t.strides()[i] != 1 || t.sizes()[i] == 1,
                  "DSV41-ACL4BIT-VIEWDIMS: axis ", i, " also has stride 1 with "
                  "extent ", t.sizes()[i], ", so the packed axis cannot be "
                  "identified from the strides alone. shape=", t.sizes(),
                  " stride=", t.strides());
    }
  }

  shape[last] *= kAclNibblesPerByte;
  stride[last] = 1;                       // 🔴 not doubled; see above
  for (int64_t i = 0; i < last; ++i) {
    stride[i] *= kAclNibblesPerByte;
  }
}

inline aclTensor *ConvertType(const at::Tensor &at_tensor) {
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }

  if (!at_tensor.defined()) {
    return nullptr;
  }
  at::ScalarType scalar_data_type = at_tensor.scalar_type();
  aclDataType acl_data_type =
      kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
  TORCH_CHECK(
      acl_data_type != ACL_DT_UNDEFINED,
      std::string(c10::toString(scalar_data_type)) + " has not been supported")
  c10::SmallVector<int64_t, 5> storageDims;
  // if acl_data_type is ACL_STRING, storageDims is empty.
  auto itemsize = at_tensor.itemsize();
  TORCH_CHECK(itemsize != 0, "When ConvertType, tensor item size cannot be zero.");

  const auto dimNum = at_tensor.sizes().size();
  aclFormat format = ACL_FORMAT_ND;
  if (!IsOpInputBaseFormat(at_tensor)) {
    format = vllm_ascend::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.npu_format_;
    if (acl_data_type != ACL_STRING) {
          storageDims = vllm_ascend::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.storage_sizes_;
      }
  } else {
    switch (dimNum) {
      case 3:
        format = ACL_FORMAT_NCL;
        break;
      case 4:
        format = ACL_FORMAT_NCHW;
        break;
      case 5:
        format = ACL_FORMAT_NCDHW;
        break;
      default:
        format = ACL_FORMAT_ND;
    }
    if (acl_data_type != ACL_STRING) {
      storageDims.push_back(at_tensor.storage().nbytes() / itemsize);
    }
  }

  if (at_tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    c10::Scalar expScalar = ConvertTensorToScalar(at_tensor);
    at::Tensor aclInput = CopyScalarToDevice(expScalar, scalar_data_type);
    return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(),
                           acl_data_type, aclInput.strides().data(),
                           aclInput.storage_offset(), format,
                           storageDims.data(), storageDims.size(),
                           const_cast<void *>(aclInput.storage().data()));
  }

  // 🔴 Refuse the 4-bit dtypes this correction does **not** implement, rather
  // than letting them through on the byte-counted path: they would be
  // described with half their extent and fail inside the kernel, or worse,
  // read the wrong rows.
  TORCH_CHECK(acl_data_type != ACL_INT4 && acl_data_type != ACL_FLOAT4_E1M2,
              "DSV41-ACL4BIT-VIEWDIMS: acl dtype ", static_cast<int>(acl_data_type),
              " is sub-byte but this binding only implements the ACL_FLOAT4_E2M1 "
              "nibble correction. Extend IsAclFourBitDtype **and** the "
              "shape/stride/offset/storageDims acceptance criteria together; "
              "adding it to one of the two is how a silent half-fix gets in.");

  if (IsAclFourBitDtype(acl_data_type)) {
    // 🔴 This literal is the build-time proof that this branch is in the .so.
    // The library is stripped: no binding function of ours appears in
    // ``nm -D --defined-only``, so a symbol-based build check would reject
    // every build -- and a check that always fails gets relaxed, taking the
    // real omissions with it. String literals survive stripping.
    // It is not a bare marker: the refusal below had no owner before, and
    // without it a non-base-format 4-bit tensor would be described with
    // half the extent and fail somewhere inside the kernel instead.
    TORCH_CHECK(IsOpInputBaseFormat(at_tensor),
                "DSV41-ACL4BIT-VIEWDIMS: a 4-bit tensor reached ConvertType in a "
                "non-base npu format; its storage_sizes_ are not derived from "
                "itemsize, so the nibble correction below does not apply and "
                "guessing would describe half the buffer. dtype=",
                at_tensor.scalar_type(), " dim=", at_tensor.sizes().size());

    c10::SmallVector<int64_t, 5> viewShape(at_tensor.sizes().begin(),
                                           at_tensor.sizes().end());
    c10::SmallVector<int64_t, 5> viewStride(at_tensor.strides().begin(),
                                            at_tensor.strides().end());
    ApplyFourBitViewDims(at_tensor, viewShape, viewStride);

    // (2) storageDims came from ``nbytes / itemsize`` in the base-format
    // branch above -- the same halving, on a *different* aclCreateTensor
    // argument. Fixing only the view dims leaves this one short.
    c10::SmallVector<int64_t, 5> storageDims4bit(storageDims);
    TORCH_CHECK(!storageDims4bit.empty(),
                "DSV41-ACL4BIT-VIEWDIMS: storageDims is empty for a 4-bit tensor");
    storageDims4bit.back() *= kAclNibblesPerByte;

    // (3) acl expresses a 4-bit storage offset in nibbles as well.
    const int64_t offset = at_tensor.storage_offset();
    TORCH_CHECK(offset <= INT64_MAX / kAclNibblesPerByte,
                "DSV41-ACL4BIT-VIEWDIMS: storage offset ", offset,
                " overflows when converted to logical nibbles");

    // -----------------------------------------------------------------------
    // 🔴 PRODUCTION evidence (NOT part of the probe -- must survive the probe
    // revert). One line per distinct descriptor, and nothing else.
    //
    // What it replaces: a CPU bench was built to answer "is the descriptor we
    // hand acl the one we think we hand it". The bench cost more than the
    // thing it was validating, and -- worse -- two of its failures were
    // **the bench's own setup**, not the subject. This line answers the same
    // question from **production itself**, which is strictly stronger: there
    // is no second implementation whose agreement has to be argued for.
    //
    // 🔴 Once per **distinct descriptor**, not once per process. A plain
    // log-once captures whichever 4-bit tensor happens to come first -- q,
    // the key cache, or a scale plane -- and then reports it as if it were
    // the one being asked about. A criterion has to act on the object it
    // judges, so each shape gets its own line, and the set is capped so the
    // log volume stays bounded no matter what arrives.
    {
        // 🔴 默认关的 env 闸 + 打满之后的原子短路。
        //
        // 上一版把字符串**无条件**构造、并且**每次调用都取锁**，上限 32 只省掉了
        // 打印。2026-09-21 那一发慢了 2.26x，这块代码被当成首要候选 —— 算术最终
        // 反对它（每次要 54.4 ms，而它是微秒级），**但那不是它该写成这样的理由**：
        // **一个仪表在没被请求时不该有任何每调用成本。**
        //
        // 现在：关 ⇒ 一次 static bool；打满 ⇒ 一次 relaxed 原子读（~1ns）；
        // 字符串只在**真的要记**的时候构造。
        //
        // 🔴 配套判据必须【两臂】：
        //   关闭时 **必须一行都不打**  —— 否则「默认关」这件事没有证据；
        //   打开时 **必须打**          —— 否则「关掉了」和「它坏了」读数相同。
        //   （只测第一臂会被一个永远不打的实现通过。）
        static const bool desc_prod_on = [] {
            const char *v = std::getenv("DSV41_ACL4BIT_DESC_PROD");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        static std::atomic<bool> desc_prod_done{false};
        if (desc_prod_on &&
            !desc_prod_done.load(std::memory_order_relaxed)) {
        static std::mutex seen_mu;
        static std::set<std::string> seen;
        std::ostringstream m;
        m << "DSV41-ACL4BIT-DESC-PROD dtype=" << static_cast<int>(acl_data_type)
          << " sizes=[";
        for (size_t i = 0; i < viewShape.size(); ++i) {
            m << (i ? "," : "") << viewShape[i];
        }
        m << "] strides=[";
        for (size_t i = 0; i < viewStride.size(); ++i) {
            m << (i ? "," : "") << viewStride[i];
        }
        m << "] offset=" << (offset * kAclNibblesPerByte) << " storageDims=[";
        for (size_t i = 0; i < storageDims4bit.size(); ++i) {
            m << (i ? "," : "") << storageDims4bit[i];
        }
        // torch_* is what we were handed; the fields above are what acl gets.
        // Both, because "the doubling landed" and "the input was already
        // doubled" are different facts and only the pair separates them.
        m << "] torch_sizes=[";
        for (size_t i = 0; i < at_tensor.sizes().size(); ++i) {
            m << (i ? "," : "") << at_tensor.sizes()[i];
        }
        m << "] torch_strides=[";
        for (size_t i = 0; i < at_tensor.strides().size(); ++i) {
            m << (i ? "," : "") << at_tensor.strides()[i];
        }
        m << "] torch_offset=" << at_tensor.storage_offset();
        const std::string line = m.str();
        bool fresh = false;
        {
            std::lock_guard<std::mutex> g(seen_mu);
            // Cap: past this many distinct descriptors, stop recording. A
            // bounded instrument that says it stopped beats an unbounded one
            // that floods the log the criterion has to be read out of.
            if (seen.size() < 32 && seen.insert(line).second) {
                fresh = true;
                if (seen.size() == 32) {
                    // 打满 ⇒ 整块从此退化成一次 relaxed 原子读。
                    desc_prod_done.store(true, std::memory_order_relaxed);
                    TORCH_WARN("DSV41-ACL4BIT-DESC-PROD-CAPPED: 32 distinct "
                               "descriptors recorded; further ones are NOT "
                               "logged. Absence below this point is not "
                               "evidence of absence.");
                }
            }
        }
        if (fresh) {
            TORCH_WARN(line);
        }
        }
    }
    // -----------------------------------------------------------------------

    return aclCreateTensor(viewShape.data(), viewShape.size(), acl_data_type,
                           viewStride.data(), offset * kAclNibblesPerByte, format,
                           storageDims4bit.data(), storageDims4bit.size(),
                           const_cast<void *>(at_tensor.storage().data()));
  }

  auto acl_tensor = aclCreateTensor(
      at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type,
      at_tensor.strides().data(), at_tensor.storage_offset(), format,
      storageDims.data(), storageDims.size(),
      const_cast<void *>(at_tensor.storage().data()));
  return acl_tensor;
}

inline aclScalar *ConvertType(const at::Scalar &at_scalar) {
  static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
  if (aclCreateScalar == nullptr) {
    return nullptr;
  }

  at::ScalarType scalar_data_type = at_scalar.type();
  aclDataType acl_data_type =
      kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
  TORCH_CHECK(
      acl_data_type != ACL_DT_UNDEFINED,
      std::string(c10::toString(scalar_data_type)) + " has not been supported")
  aclScalar *acl_scalar = nullptr;
  switch (scalar_data_type) {
    case at::ScalarType::Double: {
      double value = at_scalar.toDouble();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    case at::ScalarType::Long: {
      int64_t value = at_scalar.toLong();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    case at::ScalarType::Bool: {
      bool value = at_scalar.toBool();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    case at::ScalarType::ComplexDouble: {
      auto value = at_scalar.toComplexDouble();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    default:
      acl_scalar = nullptr;
      break;
  }
  return acl_scalar;
}

inline aclIntArray *ConvertType(const at::IntArrayRef &at_array) {
  static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
  if (aclCreateIntArray == nullptr) {
    return nullptr;
  }
  auto array = aclCreateIntArray(at_array.data(), at_array.size());
  return array;
}

template <std::size_t N>
inline aclBoolArray *ConvertType(const std::array<bool, N> &value) {
  static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
  if (aclCreateBoolArray == nullptr) {
    return nullptr;
  }

  auto array = aclCreateBoolArray(value.data(), value.size());
  return array;
}

inline aclBoolArray *ConvertType(const at::ArrayRef<bool> &value) {
  static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
  if (aclCreateBoolArray == nullptr) {
    return nullptr;
  }

  auto array = aclCreateBoolArray(value.data(), value.size());
  return array;
}

inline aclTensorList *ConvertType(const at::TensorList &at_tensor_list) {
  static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
  if (aclCreateTensorList == nullptr) {
    return nullptr;
  }

  std::vector<const aclTensor *> tensor_list(at_tensor_list.size());
  for (size_t i = 0; i < at_tensor_list.size(); i++) {
    tensor_list[i] = ConvertType(at_tensor_list[i]);
  }
  auto acl_tensor_list =
      aclCreateTensorList(tensor_list.data(), tensor_list.size());
  return acl_tensor_list;
}

inline aclTensor *ConvertType(const c10::optional<at::Tensor> &opt_tensor) {
  if (opt_tensor.has_value() && opt_tensor.value().defined()) {
    return ConvertType(opt_tensor.value());
  }
  return nullptr;
}

inline aclTensorList *ConvertType(
    const c10::optional<at::TensorList> &opt_tensor_list) {
  if (opt_tensor_list.has_value()) {
    return ConvertType(opt_tensor_list.value());
  }
  return nullptr;
}

inline aclIntArray *ConvertType(
    const c10::optional<at::IntArrayRef> &opt_array) {
  if (opt_array.has_value()) {
    return ConvertType(opt_array.value());
  }
  return nullptr;
}

inline aclScalar *ConvertType(const c10::optional<at::Scalar> &opt_scalar) {
  if (opt_scalar.has_value()) {
    return ConvertType(opt_scalar.value());
  }
  return nullptr;
}

inline aclDataType ConvertType(const at::ScalarType scalarType) {
  return kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalarType)];
}

template <typename T>
T ConvertType(T value) {
  return value;
}

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr,
                        std::index_sequence<I...>) {
  typedef int (*OpApiFunc)(
      typename std::decay<decltype(std::get<I>(params))>::type...);
  auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
  return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return ConvertToOpApiFunc(params, opApiAddr,
                            std::make_index_sequence<size>{});
}

inline void Release(aclTensor *p) {
  static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
  if (aclDestroyTensor == nullptr) {
    return;
  }
  aclDestroyTensor(p);
}

inline void Release(aclScalar *p) {
  static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
  if (aclDestroyScalar == nullptr) {
    return;
  }
  aclDestroyScalar(p);
}

inline void Release(aclIntArray *p) {
  static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
  if (aclDestroyIntArray == nullptr) {
    return;
  }

  aclDestroyIntArray(p);
}

inline void Release(aclBoolArray *p) {
  static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
  if (aclDestroyBoolArray == nullptr) {
    return;
  }

  aclDestroyBoolArray(p);
}

inline void Release(aclTensorList *p) {
  static const auto aclDestroyTensorList =
      GET_OP_API_FUNC(aclDestroyTensorList);
  if (aclDestroyTensorList == nullptr) {
    return;
  }

  aclDestroyTensorList(p);
}

template <typename T>
void Release(T value) {
  (void)value;
}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(Tuple &t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  CallRelease(t, std::make_index_sequence<size>{});
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts &... args) {
  return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return call(f, t, std::make_index_sequence<size>{});
}

template <std::size_t N>
void AddParamToBuf(const std::array<bool, N> &value) {
  MEMCPY_TO_BUF(value.data(), value.size() * sizeof(bool));
}

template <typename T>
void AddParamToBuf(const T &value) {
  MEMCPY_TO_BUF(&value, sizeof(T));
}

void AddParamToBuf(const at::Tensor &);
void AddParamToBuf(const at::Scalar &);
void AddParamToBuf(const at::IntArrayRef &);
void AddParamToBuf(const at::ArrayRef<bool> &);
void AddParamToBuf(const at::TensorList &);
void AddParamToBuf(const c10::optional<at::Tensor> &);
void AddParamToBuf(const c10::optional<at::IntArrayRef> &);
void AddParamToBuf(const c10::optional<at::Scalar> &);
void AddParamToBuf(const at::ScalarType);
void AddParamToBuf(const string &);
void AddParamToBuf();

template <typename T, typename... Args>
void AddParamToBuf(const T &arg, Args &... args) {
  AddParamToBuf(arg);
  AddParamToBuf(args...);
}

uint64_t CalcHashId();
typedef int (*InitHugeMemThreadLocal)(void *, bool);
typedef void (*UnInitHugeMemThreadLocal)(void *, bool);
typedef void (*ReleaseHugeMem)(void *, bool);

#define EXEC_NPU_CMD(aclnn_api, ...)                                          \
  do {                                                                        \
    static const auto getWorkspaceSizeFuncAddr =                              \
        GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                      \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);           \
    static const auto initMemAddr =                                           \
        GetOpApiFuncAddr("InitHugeMemThreadLocal");                           \
    static const auto unInitMemAddr =                                         \
        GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                         \
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");    \
    TORCH_CHECK(                                                              \
        getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,      \
        #aclnn_api, " or ", #aclnn_api "GetWorkspaceSize", " not in ",        \
        GetOpApiLibName(), ", or ", GetOpApiLibName(), "not found.");         \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);           \
    uint64_t workspace_size = 0;                                              \
    uint64_t *workspace_size_addr = &workspace_size;                          \
    aclOpExecutor *executor = nullptr;                                        \
    aclOpExecutor **executor_addr = &executor;                                \
    InitHugeMemThreadLocal initMemFunc =                                      \
        reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                \
    UnInitHugeMemThreadLocal unInitMemFunc =                                  \
        reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
    if (initMemFunc) {                                                        \
      initMemFunc(nullptr, false);                                            \
    }                                                                         \
    auto converted_params =                                                   \
        ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);        \
    static auto getWorkspaceSizeFunc =                                        \
        ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);       \
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);     \
    TORCH_CHECK(workspace_status == 0,                                        \
                "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg()); \
    void *workspace_addr = nullptr;                                           \
    if (workspace_size != 0) {                                                \
      at::TensorOptions options =                                             \
          at::TensorOptions(torch_npu::utils::get_npu_device_type());         \
      auto workspace_tensor =                                                 \
          at::empty({workspace_size}, options.dtype(kByte));                  \
      workspace_addr = const_cast<void *>(workspace_tensor.storage().data()); \
    }                                                                         \
    auto acl_call = [converted_params, workspace_addr, workspace_size,        \
                     acl_stream, executor]() -> int {                         \
      typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *,             \
                               const aclrtStream);                            \
      OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);       \
      auto api_ret =                                                          \
          opApiFunc(workspace_addr, workspace_size, executor, acl_stream);    \
      TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:",        \
                  aclGetRecentErrMsg());                                      \
      ReleaseConvertTypes(converted_params);                                  \
      ReleaseHugeMem releaseMemFunc =                                         \
          reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                   \
      if (releaseMemFunc) {                                                   \
        releaseMemFunc(nullptr, false);                                       \
      }                                                                       \
      return api_ret;                                                         \
    };                                                                        \
    at_npu::native::OpCommand cmd;                                            \
    cmd.Name(#aclnn_api);                                                     \
    cmd.SetCustomHandler(acl_call);                                           \
    cmd.Run();                                                                \
    if (unInitMemFunc) {                                                      \
      unInitMemFunc(nullptr, false);                                          \
    }                                                                         \
  } while (false)

#endif
