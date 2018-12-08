#include "siphon/core.h"
#include "siphon/init.h"

#include <caffe2/core/logging.h>
#include <caffe2/utils/proto_utils.h>

#include <onnx/onnx_pb.h>

#include <pybind11/pybind11.h>

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
                    auto&& net = LoadC2(canonical_path);
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

                BlobSetTensor(ws.CreateBlob(value_info->input), Tensor(value_info->dims, CPU));
            }
        }

        for (const auto& i : pred_paths)
        {
            LOG(INFO) << "Load predict net " << i << ".";

            auto&& net = LoadC2(i);
            net.set_name("pred");
            ws.CreateNet(nets[net.name()] = move(net));
        }

        LOG(INFO) << "Model loaded successfully.";
    }

    SIPHON_API
    void Siphon::Save(path dir)
    {
        LOG(INFO) << "Save model to " << dir << ".";

        CAFFE_ENFORCE(create_directories(dir), "Cannot create output directory \"" + dir.string() + "\".");
        dir = canonical(dir);

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

        CAFFE_ENFORCE(nets.count("init"), "Init net doesn't exist.");
        SaveC2(nets["init"], dir / "init.pb");

        CAFFE_ENFORCE(nets.count("pred"), "Predict net doesn't exist.");
        SaveC2(nets["pred"], dir / "pred.prototxt");

        LOG(INFO) << "Model saved in Caffe2 format successfully.";
    }

    SIPHON_API
    void Siphon::SaveONNX(path fn)
    {
        LOG(INFO) << "Save model to " << fn << " in ONNX.";

        CAFFE_ENFORCE(nets.count("init"), "Init net doesn't exist.");
        CAFFE_ENFORCE(nets.count("pred"), "Predict net doesn't exist.");

        py::dict value_info_py;
        {
            auto dims = make_tuple(
                    value_info->dims[0],
                    value_info->dims[1],
                    value_info->dims[2],
                    value_info->dims[3]);
            value_info_py[value_info->input.c_str()] = make_tuple(static_cast<int>(value_info->type), dims);
        }

        string init_str;
        string pred_str;
        nets["init"].SerializeToString(&init_str);
        nets["pred"].SerializeToString(&pred_str);
        auto proto_module = py::module::import("caffe2.proto.caffe2_pb2");
        auto init = proto_module.attr("ParseFromString")(init_str);
        auto pred = proto_module.attr("ParseFromString")(pred_str);

        auto caffe2_net_to_onnx_model = py::module::import("caffe2.python.onnx.frontend").attr("caffe2_net_to_onnx_model");
        py::bytes onnx_model_str = caffe2_net_to_onnx_model(pred, init, value_info_py).attr("SerializeToString")();

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

        LOG(INFO) << "Model saved in ONNX format successfully.";
    }

    SIPHON_HIDDEN
    NetDef Siphon::LoadC2(path fn)
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
    void Siphon::SaveC2(const NetDef& net, path fn)
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
