#include "siphon/init.h"

#include <caffe2/core/init.h>

#include <mutex>

using namespace std;
using namespace caffe2;

namespace siphon
{
    DEFINE_string(load_c2,      "", "Directory to load Caffe2 network.");
    DEFINE_string(save_onnx,    "", "Directory to save in ONNX format.");

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
