#include "siphon/core.h"
#include "siphon/init.h"
#include "siphon/pyenv.h"

#include <gflags/gflags.h>

#include <iostream>
#include <sstream>
#include <string>

using namespace std;
using namespace gflags;
using namespace siphon;

void summary()
{
    ostringstream buf;
    if (FLAGS_load.size())
        buf << " load:      " << FLAGS_load << endl;
    if (FLAGS_save.size())
        buf << " save:      " << FLAGS_save << endl;
    if (FLAGS_save_onnx.size())
        buf << " save_onnx: " << FLAGS_save_onnx << endl;

    if (buf.str().size())
    {
        cout << string(40, '=') + "\n" + buf.str() + string(40, '=') + "\n" << endl;
    }
    else
    {
        LOG(WARNING) << "No argument provided.";
    }
}

int main(int argc, char *argv[])
{
    ParseCommandLineFlags(&argc, &argv, true);

    summary();

    /*
     * Extend the life span of embedded python interpreter.
     * Numpy cannot be loaded twice.
     */
    PyEnv pyenv;

    Siphon sp;
    if (FLAGS_load.size())
    {
        sp.load(FLAGS_load);
    }
    if (FLAGS_save.size())
    {
        sp.save(FLAGS_save);
    }
    if (FLAGS_save_onnx.size())
    {
        sp.save_onnx(FLAGS_save_onnx);
    }

    return 0;
}
