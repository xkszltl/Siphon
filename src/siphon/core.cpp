#include "siphon/core.h"
#include "siphon/init.h"

#include <caffe2/core/logging.h>
#include <caffe2/utils/proto_utils.h>

#include <onnx/onnx_pb.h>

#include <pybind11/pybind11.h>

#include <locale>
#include <memory>
#include <set>
#include <string>

using namespace std;

#if __has_include(<filesystem>)
    using namespace std::filesystem;
#else
    using namespace std::experimental::filesystem;
#endif

using namespace caffe2;
using namespace pybind11::literals;

namespace siphon
{
    namespace py = pybind11;

    SIPHON_API
    Siphon::Siphon()
    {
        Init();
    }

    SIPHON_API
    void Siphon::Load(path dir)
    {
        LOG(INFO) << "Load model from " << dir << ".";

        dir = canonical(dir);
        CAFFE_ENFORCE(exists(dir), "Model directory \"" + dir.string() + "\" doesn't exist.");
        CAFFE_ENFORCE(is_directory(dir), "\"" + dir.string() + "\" is not a directory.");
        for (const auto& fn : directory_iterator(dir))
        {
            auto ext = fn.path().extension().string();
            {
                for (auto& c : ext)
                    c = tolower(c, locale());
            }
            if (ext == ".pb" || ext == ".prototxt")
            {
                auto&& net = LoadC2(canonical(fn.path()));
                {
                    auto name = fn.path().stem().string();
                    for (auto& c : name)
                        c = tolower(c, locale());
                    if (set<string>{("init"), ("init_net")}.count(name))
                        net.set_name("init");
                    else if (set<string>{"pred", "predict", "pred_net", "predict_net"}.count(name))
                        net.set_name("pred");
                }
                ws.CreateNet(net);
                nets[net.name()] = move(net);
            }
        }
    }

    SIPHON_API
    void Siphon::Save(path dir)
    {
        LOG(INFO) << "Save model to " << dir << ".";

        dir = canonical(dir);
        CAFFE_ENFORCE(create_directories(dir), "Cannot create output directory \"" + dir.string() + "\".");

        CAFFE_ENFORCE(nets.count("init"), "Init net doesn't exist.");
        SaveC2(nets["init"], dir / "init.pb");

        CAFFE_ENFORCE(nets.count("pred"), "Predict net doesn't exist.");
        SaveC2(nets["pred"], dir / "pred.prototxt");
    }

    SIPHON_API
    void Siphon::SaveONNX(path fn)
    {
        LOG(INFO) << "Save model to " << fn << " in ONNX.";

        CAFFE_ENFORCE(nets.count("init"), "Init net doesn't exist.");
        CAFFE_ENFORCE(nets.count("pred"), "Predict net doesn't exist.");
        auto value_info = py::dict(
                "data"_a = py::make_tuple(
                    py::module::import("onnx").attr("TensorProto.FLOAT"),
                    py::make_tuple(1, 1, 1, 1)));
        auto caffe2_net_to_onnx_model = py::module::import("caffe2.python.onnx.frontend").attr("caffe2_net_to_onnx_model");
        py::bytes onnx_model_str = caffe2_net_to_onnx_model(nets["pred"], nets["init"], value_info).attr("SerializeToString")();
        onnx_c2::ModelProto onnx_model;
        ParseProtoFromLargeString(static_cast<string>(onnx_model_str), &onnx_model);

        fn = canonical(fn);

        {
            auto ext = fn.extension().string();
            for (auto& c : ext)
                c = tolower(c, locale());
            CHECK_EQ(ext, ".onnx") << "Unexpected extension " << ext << " for ONNX.";
        }

        WriteProtoToBinaryFile(onnx_model, fn.string());
    }

    SIPHON_HIDDEN
    NetDef Siphon::LoadC2(path fn)
    {
        fn = canonical(fn);
        CAFFE_ENFORCE(exists(fn), "Caffe2 model file \"" + fn.string() + "\" doesn't exist.");
        CAFFE_ENFORCE(!is_directory(fn), "Get directory \"" + fn.string() + "\" while expecting Caffe2 model file.");

        NetDef net;
        CAFFE_ENFORCE(ReadProtoFromFile(fn.string(), &net), "Failed to read Caffe2 model \"" + fn.string() + "\".");
        return net;
    }

    SIPHON_HIDDEN
    void Siphon::SaveC2(const NetDef& net, path fn)
    {
        auto ext = fn.extension().string();
        {
            for (auto& c : ext)
                c = tolower(c, locale());
        }
        fn = canonical(fn);
        if (ext == ".pb")
        {
            WriteProtoToBinaryFile(net, fn.string());
        }
        else if (ext == ".prototxt")
        {
            WriteProtoToTextFile(net, fn.string());
        }
        else
        {
            CAFFE_ENFORCE(false, "Unknown extension \"" + ext + "\" when writing to \"" + fn.string() + "\"");
        }
    }
}
