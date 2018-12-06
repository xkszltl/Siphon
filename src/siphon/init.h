#pragma once

#include <gflags/gflags.h>

namespace siphon
{
    DECLARE_string(init);
    DECLARE_string(pred);
    DECLARE_string(save_onnx);

    int Init(const bool force = false);
}
