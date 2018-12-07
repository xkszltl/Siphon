#pragma once

#include <gflags/gflags.h>

namespace siphon
{
    DECLARE_string(load);
    DECLARE_string(save);
    DECLARE_string(save_onnx);

    int Init(const bool force = false);
}
