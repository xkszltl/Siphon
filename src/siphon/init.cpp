#include <caffe2/core/init.h>

using namespace caffe2;

namespace siphon
{
    int Init()
    {
        return GlobalInit();
    }
}
