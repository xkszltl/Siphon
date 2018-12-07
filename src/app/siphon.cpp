#include "siphon/core.h"
#include "siphon/init.h"

#include <iostream>

using namespace std;
using namespace siphon;

int main(int argc, char *argv[])
{
    Siphon sp;
    if (FLAGS_load != "")
    {
        sp.Load(FLAGS_load);
    }
    if (FLAGS_save != "")
    {
        sp.Save(FLAGS_save);
    }
    if (FLAGS_save != "")
    {
        sp.SaveONNX(FLAGS_save_onnx);
    }
    return 0;
}
