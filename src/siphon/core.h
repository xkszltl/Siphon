#pragma once

#include "siphon/utils.h"

#include <caffe2/core/net.h>
#include <caffe2/core/workspace.h>

#if __has_include(<filesystem>)
    #include <filesystem>
#else
    #include <experimental/filesystem>
#endif

#include <map>
#include <string>

namespace siphon
{
    class Siphon
    {
    public:
        using NetDef = caffe2::NetDef;

        using Workspace = caffe2::Workspace;

        template <typename K, typename V>
        using map = std::map<K, V>;

    #if __has_include(<filesystem>)
        using path = std::filesystem::path;
    #else
        using path = std::experimental::filesystem::path;
    #endif

        SIPHON_API
        Siphon();

        SIPHON_API
        void Load(path dir);

        SIPHON_API
        void Save(path dir);

        SIPHON_API
        void SaveONNX(path fn);

        Workspace ws;
        map<string, NetDef> nets;

    private:
        SIPHON_HIDDEN
        NetDef LoadC2(path fn);

        SIPHON_HIDDEN
        void SaveC2(const NetDef& net, path fn);
    };
}
