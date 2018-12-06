#pragma once

#include <caffe2/core/net.h>
#include <caffe2/core/workspace.h>

#if __has_include(<filesystem>)
    #include <filesystem>
#else
    #include <experimental/filesystem>
#endif

namespace siphon
{
    class Siphon
    {
    public:
        using NetDef = caffe2::NetDef;

        using Workspace = caffe2::Workspace;

    #if __has_include(<filesystem>)
        using path = std::filesystem::path;
    #else
        using path = std::experimental::filesystem::path;
    #endif

        Siphon();

        NetDef LoadC2(path fn);
        void Load(path dir);

        Workspace ws;
    };
}
