#pragma once

#include "siphon/pyenv.h"
#include "siphon/utils.h"

#include <c10/core/Device.h>

#include <caffe2/core/net.h>
#include <caffe2/core/workspace.h>

#include <onnx/onnx_pb.h>

#include <filesystem>
#include <map>
#include <memory>
#include <regex>
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

        using path = std::filesystem::path;

        using regex = std::regex;

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
            onnx::TensorProto_DataType type;
            vector<int> dims;
        };
        
        map<string, ValueInfo> value_info;

    private:
        SIPHON_HIDDEN
        NetDef& eval_fill(NetDef& net) const;

        SIPHON_HIDDEN
        static NetDef load_c2(path fn);

        SIPHON_HIDDEN
        static void save_c2(const NetDef& net, path fn);

        SIPHON_HIDDEN
        void optimize_c2();

        SIPHON_HIDDEN
        void load_onnx(path dir);

        SIPHON_HIDDEN
        void load_value_info(path fn);

        SIPHON_HIDDEN
        void save_value_info(const path& fn);

        PyEnv pyenv;

        static const regex gr_multi;
        static const regex gr_single;
        static const regex gr_dim;
    };
}
