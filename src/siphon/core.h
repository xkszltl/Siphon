#pragma once

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

        Siphon();

        void Load(path dir);
        void Save(path dir);
        void SaveONNX(path fn);

        Workspace ws;
        map<string, NetDef> nets;

    private:
        NetDef LoadC2(path fn);
        void SaveC2(const NetDef& net, path fn);
    };
}
