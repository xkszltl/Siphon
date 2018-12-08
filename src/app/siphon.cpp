#include "siphon/core.h"
#include "siphon/init.h"

#include <gflags/gflags.h>

#include <iostream>

using namespace std;
using namespace gflags;
using namespace siphon;

int main(int argc, char *argv[])
{
    ParseCommandLineFlags(&argc, &argv, true);
    Siphon sp;
    if (FLAGS_load != "")
    {
        sp.Load(FLAGS_load);
    }
    if (FLAGS_save != "")
    {
        sp.Save(FLAGS_save);
    }
    if (FLAGS_save_onnx != "")
    {
        sp.SaveONNX(FLAGS_save_onnx);
    }
    return 0;
}
