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
#include <memory>
#include <string>
#include <vector>

namespace siphon
{
    class Siphon
    {
    public:
        using TensorProto_DataType = caffe2::TensorProto_DataType;

        using NetDef = caffe2::NetDef;

        using Workspace = caffe2::Workspace;

        template <typename K, typename V>
        using map = std::map<K, V>;

    #if __has_include(<filesystem>)
        using path = std::filesystem::path;
    #else
        using path = std::experimental::filesystem::path;
    #endif

        template <typename T>
        using unique_ptr = std::unique_ptr<T>;

        template <typename T>
        using vector = std::vector<T>;

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

        struct ValueInfo
        {
            string input;
            TensorProto_DataType type;
            vector<int> dims;
        };
        
        unique_ptr<ValueInfo> value_info;

    private:
        SIPHON_HIDDEN
        NetDef LoadC2(path fn);

        SIPHON_HIDDEN
        void SaveC2(const NetDef& net, path fn);
    };
}
