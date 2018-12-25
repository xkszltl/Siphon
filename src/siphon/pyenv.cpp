#include "siphon/pyenv.h"

#include <caffe2/core/logging.h>

#include <pybind11/embed.h>

#include <exception>
#include <mutex>
#include <string>

using namespace std;

namespace siphon
{
    namespace py = pybind11;

    SIPHON_API
    PyEnv::PyEnv()
    {
        lock_guard<recursive_mutex> lck(mtx);
        if (!inst)
        {
            inst = this;
            LOG(INFO) << "Initialize embedded Python interpreter.";
            py::initialize_interpreter();
        }
        ++inst->counter;
    }

    SIPHON_API
    PyEnv::~PyEnv()
    {
        lock_guard<recursive_mutex> lck(mtx);
        CAFFE_ENFORCE(inst, "PyEnv instance does not exist.");
        if (!--inst->counter)
        {
            LOG(INFO) << "Finalize embedded Python interpreter.";
            py::finalize_interpreter();
            inst = nullptr;
        }
    }

    SIPHON_API
    py::object PyEnv::import(const string& module)
    {
        lock_guard<recursive_mutex> lck(mtx);
        CAFFE_ENFORCE(inst, "PyEnv is not created");

        try
        {
            return py::module::import(module.c_str());
        }
        catch (const exception& e)
        {
            LOG(FATAL)
                << "Failed to import module \"" << module << "\":" << endl
                << string(40, '-') << endl
                << e.what() << endl
                << string(40, '0');
            throw;
        }
    }

    SIPHON_API
    void PyEnv::exec(function<void()> f)
    {
        lock_guard<recursive_mutex> lck(mtx);
        CAFFE_ENFORCE(inst, "PyEnv is not created");
        f();
    }

    recursive_mutex PyEnv::mtx;
    PyEnv* PyEnv::inst = nullptr;
}
