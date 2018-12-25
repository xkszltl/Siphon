#include "siphon/init.h"

#include <caffe2/core/init.h>

#include <mutex>

using namespace std;
using namespace caffe2;

namespace siphon
{
    DEFINE_string(load,      "", "Directory to load Caffe2/ONNX network.");
    DEFINE_string(save,      "", "Directory to save in Caffe2 format.");
    DEFINE_string(save_onnx, "", "Directory to save in ONNX format.");

    SIPHON_API
    int Init(const bool force)
    {
        static bool initialized = false;
        static mutex mtx;
        static int res = 0;

        lock_guard<mutex> lck(mtx);
        if (initialized && !force)
        {
            return res;
        }

        res = GlobalInit();
        initialized = true;
        return res;
    }
}
