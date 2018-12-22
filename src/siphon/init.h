#pragma once

#include "siphon/utils.h"

#include <gflags/gflags.h>

namespace siphon
{
    DECLARE_string(load);
    DECLARE_string(save);
    DECLARE_string(load_onnx);
    DECLARE_string(save_onnx);

    SIPHON_API
    int Init(const bool force = false);
}
