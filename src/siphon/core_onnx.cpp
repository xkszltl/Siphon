#include "siphon/core.h"

#include <caffe2/core/logging.h>
#include <caffe2/utils/proto_utils.h>

#include <onnx/onnx_pb.h>

#include <pybind11/embed.h>

#include <locale>
#include <string>
#include <tuple>

using namespace std;

#if __has_include(<filesystem>)
    using namespace std::filesystem;
#else
    using namespace std::experimental::filesystem;
#endif

using namespace caffe2;
using namespace pybind11::literals;

using ::ONNX_NAMESPACE::ModelProto;

namespace siphon
{
    namespace py = pybind11;

    SIPHON_API
    void Siphon::save_onnx(path dir)
    {
        LOG(INFO) << "Save model to " << dir << " in ONNX.";

        CAFFE_ENFORCE(create_directories(dir), "Cannot create output directory \"" + dir.string() + "\".");
        dir = canonical(dir);

        CAFFE_ENFORCE(nets.count("init"), "Init net doesn't exist.");
        CAFFE_ENFORCE(nets.count("pred"), "Predict net doesn't exist.");
        CAFFE_ENFORCE(value_info.size(), "Missing value info.");

        LOG(INFO) << "Convert fill ops into GivenTensor*Fill ops for init net.";
        eval_fill(nets["init"]);

        LOG(INFO) << "Convert fill ops into GivenTensor*Fill ops for predict net.";
        eval_fill(nets["pred"]);

        save_value_info(dir / "value_info.json");

        string init_str;
        string pred_str;
        nets["init"].SerializeToString(&init_str);
        nets["pred"].SerializeToString(&pred_str);

        LOG(INFO) << "Found suitable network for ONNX. Send to python for processing.";

        string onnx_model_str;
        ModelProto onnx_model;

        LOG(INFO) << "Save to ONNX using value info:\n" << show_value_info("\t");

        // Python inter-ops.
        pyenv.exec([&]()
            {
                py::dict value_info_py;
                for (const auto& info : value_info)
                {
                    CHECK_GT(info.second.dims.size(), static_cast<size_t>(0)) << "Missing dimension info.";
                    py::tuple dims_py(info.second.dims.size());
                    for (size_t i = 0; i < info.second.dims.size(); ++i)
                    {
                        dims_py[i] = info.second.dims[i];
                    }
                    value_info_py[info.first.c_str()] = make_tuple(static_cast<int>(info.second.type), dims_py);
                }

                auto onnx_module = pyenv.import("onnx");
                auto proto_module = pyenv.import("caffe2.proto.caffe2_pb2");
                auto frontend_module = pyenv.import("caffe2.python.onnx.frontend");

                LOG(INFO) << "Deserialize init network in python.";
                auto init = proto_module.attr("NetDef")();
                init.attr("ParseFromString")(py::bytes(init_str));

                LOG(INFO) << "Deserialize predict network in python.";
                auto pred = proto_module.attr("NetDef")();
                pred.attr("ParseFromString")(py::bytes(pred_str));

                LOG(INFO) << "Create ONNX model in python.";
                auto caffe2_net_to_onnx_model = frontend_module.attr("caffe2_net_to_onnx_model");
                auto onnx_model_py = caffe2_net_to_onnx_model(pred, init, value_info_py);

                LOG(INFO) << "Serialize ONNX model and send back to C++.";
                {
                    py::bytes onnx_model_str_py = onnx_model_py.attr("SerializeToString")();
                    onnx_model_str = static_cast<string>(onnx_model_str_py);
                }

                LOG(INFO) << "Deserialize ONNX model in C++..";
                ParseProtoFromLargeString(static_cast<string>(onnx_model_str), &onnx_model);
                onnx_model.SerializeToString(&onnx_model_str);

                LOG(INFO) << "Send ONNX model back to python for check.";
                auto model_proto_py = onnx_module.attr("ModelProto")();
                model_proto_py.attr("ParseFromString")(py::bytes(onnx_model_str));

                LOG(INFO) << "Check ONNX model in python.";
                auto check_model = onnx_module.attr("checker").attr("check_model");
                check_model(py::bytes(model_proto_py));
            });

        LOG(INFO) << "Passed sanity check for ONNX model.";

        WriteProtoToBinaryFile(onnx_model, (dir / "model.onnx").string());

        LOG(INFO) << "Model saved in ONNX format successfully.";
    }

    SIPHON_HIDDEN
    void Siphon::load_onnx(path fn)
    {
        fn = canonical(fn);

        LOG(INFO) << "Load ONNX model " << fn << ".";

        // Parse ONNX model in C++ first for better debugging experience.

        ModelProto onnx_model;
        CAFFE_ENFORCE(ReadProtoFromFile(fn.string(), &onnx_model), "Failed to read ONNX model \"" + fn.string() + "\".");

        string onnx_model_str;
        onnx_model.SerializeToString(&onnx_model_str);

        LOG(INFO) << "Parse ONNX model in C++ successfully. Send to python for processing.";

        string init_str;
        string pred_str;

        // Python inter-ops.
        pyenv.exec([&]()
            {
                auto onnx_module = pyenv.import("onnx");
                auto proto_module = pyenv.import("caffe2.proto.caffe2_pb2");
                auto backend_module = pyenv.import("caffe2.python.onnx.backend");

                LOG(INFO) << "Deserialize ONNX model in python.";
                auto model_proto_py = onnx_module.attr("ModelProto")();
                model_proto_py.attr("ParseFromString")(py::bytes(onnx_model_str));

                LOG(INFO) << "Convert ONNX model to Caffe2 format in python.";

                auto backend = backend_module.attr("Caffe2Backend")();
                auto onnx_graph_to_caffe2_net = backend.attr("onnx_graph_to_caffe2_net");
                auto c2_nets_py = py::tuple(onnx_graph_to_caffe2_net(model_proto_py, DeviceTypeName(dev_type), 8));

                LOG(INFO) << "Serialize Caffe2 model in python and send back to C++.";

                init_str = static_cast<string>(py::bytes(c2_nets_py[0].attr("SerializeToString")()));
                pred_str = static_cast<string>(py::bytes(c2_nets_py[1].attr("SerializeToString")()));
            });

        LOG(INFO) << "Deserialize and initialize Caffe2 init net in C++.";
        {
            NetDef net;
            ParseProtoFromLargeString(init_str, &net);
            net.set_name("init");
            nets[net.name()] = move(net);
        }

        LOG(INFO) << "Deserialize and initialize Caffe2 predict net in C++.";
        {
            NetDef net;
            ParseProtoFromLargeString(pred_str, &net);
            net.set_name("pred");
            nets[net.name()] = move(net);
        }

        LOG(INFO) << "ONNX model loaded successfully.";
    }
}
