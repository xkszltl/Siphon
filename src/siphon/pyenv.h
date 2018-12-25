#pragma once

#include "siphon/utils.h"

#include <pybind11/embed.h>

#include <functional>
#include <mutex>

namespace siphon
{
    class PyEnv
    {
    public:
        template <typename T>
        using function = std::function<T>;

        using recursive_mutex = std::recursive_mutex;

        using string = std::string;

        SIPHON_API
        PyEnv();

        SIPHON_API
        ~PyEnv();

        SIPHON_API
        static pybind11::object import(const string& module);

        SIPHON_API
        void exec(function<void()> f);

        long long counter = 0;

        static PyEnv* inst;

    private:
        static recursive_mutex mtx;
    };
}
