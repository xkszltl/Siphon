#include "siphon/core.h"
#include "siphon/init.h"

#include <caffe2/core/logging.h>
#include <caffe2/utils/proto_utils.h>

#include <onnx/onnx_pb.h>

#include <pybind11/embed.h>

#include <exception>
#include <fstream>
#include <locale>
#include <memory>
#include <regex>
#include <set>
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

    static py::object py_import(const string& module)
    {
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
    Siphon::Siphon()
    {
        Init();
    }

    SIPHON_API
    void Siphon::load(path dir)
    {
        LOG(INFO) << "Load model from " << dir << ".";

        dir = canonical(dir);
        CAFFE_ENFORCE(exists(dir), "Model directory \"" + dir.string() + "\" doesn't exist.");
        CAFFE_ENFORCE(is_directory(dir), "\"" + dir.string() + "\" is not a directory.");

        vector<path> pred_paths;

        for (const auto& fn : directory_iterator(dir))
        {
            LOG(INFO) << "Examing " << fn << ".";

            auto name = fn.path().stem().string();
            auto ext = fn.path().extension().string();
            for (auto& c : name)
                c = tolower(c, locale());
            for (auto& c : ext)
                c = tolower(c, locale());

            const auto& canonical_path = canonical(fn.path());

            if (set<string>{ ".pb", ".pbtxt", ".prototxt" }.count(ext))
            {
                if (set<string>{ "init", "init_net" }.count(name))
                {
                    LOG(INFO) << "Found init net " << canonical_path << ".";
                    auto&& net = load_c2(canonical_path);
                    net.set_name("init");
                    ws.RunNetOnce(nets[net.name()] = move(net));
                }
                else if (set<string>{ "pred", "predict", "pred_net", "predict_net" }.count(name))
                {
                    LOG(INFO) << "Found predict net " << canonical_path << ".";
                    pred_paths.emplace_back(canonical_path);
                }
                else
                {
                    LOG(WARNING) << "Unknown Caffe2 model " << canonical_path << ". Ignore.";
                }
            }
            else if (set<string>{ "value_info", "valueinfo" }.count(name) && ext == ".json")
            {
                LOG(INFO) << "Found value info file " << canonical_path << ".";

                string json;
                {
                    ifstream fin(canonical_path);
                    CAFFE_ENFORCE(fin.is_open(), "Cannot open \"" + canonical_path.string() + "\".");
                    for (string buf; fin >> buf; json += buf + "\n");
                }

                regex syntax(R"(^\s*\{\s*"([^"]+)\"\s*:\s*\[\s*([0-9]+)\s*,\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*\]\}\s*$)");
                smatch match;
                CAFFE_ENFORCE(regex_match(json, match, syntax), "Syntax error in \"" + canonical_path.string() + "\":\n" + string(80, '-') + "\n" + json + "\n" + string(80, '-'));
                CHECK_EQ(match.size(), 7) << "Wrong number of matched components.";
                value_info.reset(new ValueInfo);
                value_info->input = match[1];
                value_info->type = static_cast<TensorProto_DataType>(stoi(match[2]));
                for (size_t i = 3; i < match.size(); ++i)
                {
                    value_info->dims.emplace_back(stoi(match[i]));
                }

                string buf;
                for (size_t i = 0; i < value_info->dims.size(); buf += to_string(value_info->dims[i++]) + ", ");
                buf.erase(buf.size() - 2);
                LOG(INFO)
                    << "Create input blob based on value info:\n\tname: "
                    << value_info->input
                    << "\n\ttype: "
                    << value_info->type
                    << "\n\tdims: ["
                    << buf
                    << "]";

                BlobSetTensor(ws.CreateBlob(value_info->input), Tensor(value_info->dims, CUDA));
            }
        }

        for (const auto& i : pred_paths)
        {
            LOG(INFO) << "Load predict net " << i << ".";

            auto&& net = load_c2(i);
            net.set_name("pred");
            ws.CreateNet(nets[net.name()] = move(net));
        }

        LOG(INFO) << "Model loaded successfully.";
    }

    SIPHON_API
    void Siphon::save(path dir)
    {
        LOG(INFO) << "Save model to " << dir << ".";

        CAFFE_ENFORCE(create_directories(dir), "Cannot create output directory \"" + dir.string() + "\".");
        dir = canonical(dir);

        if (value_info)
        {
            const auto& fn = dir / "value_info.json";
            ofstream fout(fn);
            CAFFE_ENFORCE(fout.is_open(), "Failed to open \"" + fn.string() + "\".");
            fout << "{\"" << value_info->input << "\": [" << static_cast<int>(value_info->type) << ", [";
            for (size_t i = 0; i < value_info->dims.size(); ++i)
            {
                fout << (i ? ", " : "") << value_info->dims[i];
            }
            fout << "]]}" << endl;
            CAFFE_ENFORCE(fout, "Failied to write to \"" + fn.string() + "\".");
        }
        else
        {
            LOG(WARNING) << "No value_info available. Saved Caffe2 model may not be converted to ONNX.";
        }

        CAFFE_ENFORCE(nets.count("init"), "Init net doesn't exist.");
        save_c2(nets["init"], dir / "init.pb");

        CAFFE_ENFORCE(nets.count("pred"), "Predict net doesn't exist.");
        save_c2(nets["pred"], dir / "pred.prototxt");

        LOG(INFO) << "Model saved in Caffe2 format successfully.";
    }

    SIPHON_API
    void Siphon::load_onnx(path fn)
    {
        // fn = canonical(fn);

        {
            auto ext = fn.extension().string();
            for (auto& c : ext)
                c = tolower(c, locale());
            CHECK_EQ(ext, ".onnx") << "Unexpected extension " << ext << " for ONNX.";
        }

        LOG(INFO) << "Load ONNX model " << fn << ".";

        // Parse ONNX model in C++ first for better debugging experience.

        ModelProto onnx_model;
        CAFFE_ENFORCE(ReadProtoFromFile(fn.string(), &onnx_model), "Failed to read ONNX model \"" + fn.string() + "\".");

        for (const auto& input : onnx_model.graph().input())
        {
            value_info.reset(new ValueInfo);
            value_info->input = input.name();
            value_info->type = static_cast<TensorProto_DataType>(input.type().tensor_type().elem_type());
            for (const auto& dim : input.type().tensor_type().shape().dim())
            {
                value_info->dims.emplace_back(static_cast<int>(dim.dim_value()));
            }

            BlobSetTensor(ws.CreateBlob(value_info->input), Tensor(value_info->dims, CUDA));
        }

        string onnx_model_str;
        onnx_model.SerializeToString(&onnx_model_str);

        LOG(INFO) << "Parse ONNX model in C++ successfully. Send to python for processing.";

        string init_str;
        string pred_str;

        // Python inter-ops.
        {
            py::scoped_interpreter guard;
            {
                auto onnx_module = py_import("onnx");
                auto proto_module = py_import("caffe2.proto.caffe2_pb2");
                auto backend_module = py_import("caffe2.python.onnx.backend");

                LOG(INFO) << "Deserialize ONNX model in python.";
                auto model_proto_py = onnx_module.attr("ModelProto")();
                model_proto_py.attr("ParseFromString")(py::bytes(onnx_model_str));

                LOG(INFO) << "Convert ONNX model to Caffe2 format in python.";

                auto backend = backend_module.attr("Caffe2Backend")();
                auto onnx_graph_to_caffe2_net = backend.attr("onnx_graph_to_caffe2_net");
                auto c2_nets_py = py::tuple(onnx_graph_to_caffe2_net(model_proto_py, "CUDA", 8));

                LOG(INFO) << "Serialize Caffe2 model in python and send back to C++.";

                init_str = static_cast<string>(py::bytes(c2_nets_py[0].attr("SerializeToString")()));
                pred_str = static_cast<string>(py::bytes(c2_nets_py[1].attr("SerializeToString")()));
            }
        }

        LOG(INFO) << "Deserialize and initialize Caffe2 init net in C++.";
        {
            NetDef net;
            ParseProtoFromLargeString(init_str, &net);
            net.set_name("init");
            ws.RunNetOnce(nets[net.name()] = move(net));
        }

        LOG(INFO) << "Deserialize and initialize Caffe2 predict net in C++.";
        {
            NetDef net;
            ParseProtoFromLargeString(pred_str, &net);
            net.set_name("pred");
            ws.CreateNet(nets[net.name()] = move(net));
        }

        LOG(INFO) << "ONNX model loaded successfully.";
    }

    SIPHON_API
    void Siphon::save_onnx(path fn)
    {
        LOG(INFO) << "Save model to " << fn << " in ONNX.";

        CAFFE_ENFORCE(nets.count("init"), "Init net doesn't exist.");
        CAFFE_ENFORCE(nets.count("pred"), "Predict net doesn't exist.");
        CAFFE_ENFORCE(value_info, "Missing value info.");

        string init_str;
        string pred_str;
        nets["init"].SerializeToString(&init_str);
        nets["pred"].SerializeToString(&pred_str);

        LOG(INFO) << "Found suitable network for ONNX. Send to python for processing.";

        string onnx_model_str;
        ModelProto onnx_model;

        // Python inter-ops.
        {
            py::scoped_interpreter guard;
            {
                py::dict value_info_py;
                {
                    auto dims = make_tuple(
                            value_info->dims[0],
                            value_info->dims[1],
                            value_info->dims[2],
                            value_info->dims[3]);
                    value_info_py[value_info->input.c_str()] = make_tuple(static_cast<int>(value_info->type), dims);
                }

                auto onnx_module = py_import("onnx");
                auto proto_module = py_import("caffe2.proto.caffe2_pb2");
                auto frontend_module = py_import("caffe2.python.onnx.frontend");

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
            }
        }

        LOG(INFO) << "Passed sanity check for ONNX model.";

        // fn = canonical(fn);

        {
            auto ext = fn.extension().string();
            for (auto& c : ext)
                c = tolower(c, locale());
            CHECK_EQ(ext, ".onnx") << "Unexpected extension " << ext << " for ONNX.";
        }

        WriteProtoToBinaryFile(onnx_model, fn.string());

        LOG(INFO) << "Model saved in ONNX format successfully.";
    }

    SIPHON_HIDDEN
    NetDef Siphon::load_c2(path fn)
    {
        fn = canonical(fn);

        LOG(INFO) << "Loading Caffe2 model from " << fn << ".";

        CAFFE_ENFORCE(exists(fn), "Caffe2 model file \"" + fn.string() + "\" doesn't exist.");
        CAFFE_ENFORCE(!is_directory(fn), "Get directory \"" + fn.string() + "\" while expecting Caffe2 model file.");

        NetDef net;
        CAFFE_ENFORCE(ReadProtoFromFile(fn.string(), &net), "Failed to read Caffe2 model \"" + fn.string() + "\".");
        return net;
    }

    SIPHON_HIDDEN
    void Siphon::save_c2(const NetDef& net, path fn)
    {
        auto ext = fn.extension().string();
        {
            for (auto& c : ext)
                c = tolower(c, locale());
        }
        // fn = canonical(fn);
        if (ext == ".pb")
        {
            WriteProtoToBinaryFile(net, fn.string());
        }
        else if (set<string>{ ".pbtxt", ".prototxt" }.count(ext))
        {
            WriteProtoToTextFile(net, fn.string());
        }
        else
        {
            CAFFE_ENFORCE(false, "Unknown extension \"" + ext + "\" when writing to \"" + fn.string() + "\"");
        }
    }
}
