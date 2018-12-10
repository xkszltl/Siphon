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
        sp.load(FLAGS_load);
    }
    if (FLAGS_save != "")
    {
        sp.save(FLAGS_save);
    }
    if (FLAGS_save_onnx != "")
    {
        sp.save_onnx(FLAGS_save_onnx);
    }
    return 0;
}
