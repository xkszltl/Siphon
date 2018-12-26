#pragma once

#include "siphon/pyenv.h"
#include "siphon/utils.h"

#include <c10/Device.h>

#include <caffe2/core/net.h>
#include <caffe2/core/workspace.h>

#include <onnx/onnx_pb.h>

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
        using NetDef = caffe2::NetDef;

        using Workspace = caffe2::Workspace;

        template <typename K, typename V>
        using map = std::map<K, V>;

    #if __has_include(<filesystem>)
        using path = std::filesystem::path;
    #else
        using path = std::experimental::filesystem::path;
    #endif

        using string = std::string;

        template <typename T>
        using unique_ptr = std::unique_ptr<T>;

        template <typename T>
        using vector = std::vector<T>;

        SIPHON_API
        Siphon();

        SIPHON_API
        void load(path dir);

        SIPHON_API
        void save(path dir);

        SIPHON_API
        void save_onnx(path dir);

        SIPHON_API
        string show_value_info(const string& prefix = "");

        Workspace ws;
        map<string, NetDef> nets;

        c10::DeviceType dev_type = c10::DeviceType::CPU;

        struct ValueInfo
        {
            string input;
            onnx::TensorProto_DataType type;
            vector<int> dims;
        };
        
        unique_ptr<ValueInfo> value_info;

    private:
        SIPHON_HIDDEN
        static NetDef load_c2(path fn);

        SIPHON_HIDDEN
        static void save_c2(const NetDef& net, path fn);

        SIPHON_HIDDEN
        void load_onnx(path dir);

        SIPHON_HIDDEN
        void save_value_info(const path& fn);

        PyEnv pyenv;
    };
}
