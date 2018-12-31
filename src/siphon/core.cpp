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
                    if (nets.count(net.name()))
                    {
                        LOG(WARNING) << "Overwriting init net. Check if multiple models exist within the same directory.";
                    }
                    nets[net.name()] = move(net);
                }
                else if (set<string>{ "pred", "predict", "pred_net", "predict_net" }.count(name))
                {
                    LOG(INFO) << "Found predict net " << canonical_path << ".";
                    auto&& net = load_c2(canonical_path);
                    net.set_name("pred");
                    if (nets.count(net.name()))
                    {
                        LOG(WARNING) << "Overwriting predict net. Check if multiple models exist within the same directory.";
                    }
                    nets[net.name()] = move(net);
                }
                else
                {
                    LOG(WARNING) << "Unknown Caffe2 model " << canonical_path << ". Ignore.";
                }
            }
            else if (set<string>{ "model", "net", "network" }.count(name) && ext == ".onnx")
            {
                LOG(INFO) << "Found ONNX model " << canonical_path << ".";
                load_onnx(canonical_path);
            }
            else if (set<string>{ "value_info", "valueinfo" }.count(name) && ext == ".json")
            {
                LOG(INFO) << "Found value info file " << canonical_path << ".";
                load_value_info(canonical_path);

                LOG(INFO) << "Create input blob based on value info:\n" << show_value_info("\t");

                BlobSetTensor(ws.CreateBlob(value_info->input), Tensor(value_info->dims, dev_type));
                ws.GetBlob(value_info->input)->GetMutable<Tensor>()->mutable_data<float>();
            }
        }

        if (nets.count("init"))
        {
            LOG(INFO) << "Run init net.";
            ws.RunNetOnce(nets["init"]);
        }
        else
        {
            LOG(WARNING) << "Init net not found.";
        }

        CHECK_GT(nets.count("pred"), 0) << "Predict net not found.";
        LOG(INFO) << "Create predict net.";
        ws.CreateNet(nets["pred"]);
        // ws.RunNet("pred");

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
            save_value_info(dir / "value_info.json");
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
    string Siphon::show_value_info(const string& prefix)
    {
        string buf;
        for (size_t i = 0; i < value_info->dims.size(); buf += to_string(value_info->dims[i++]) + ", ");
        buf.erase(buf.size() - 2);

        return prefix + "name: " + value_info->input
            + "\n" + prefix + "type: " + onnx::TensorProto_DataType_Name(value_info->type)
            + "\n"+ prefix + "dims: [" + buf + "]";
    }

    SIPHON_HIDDEN
    void Siphon::load_value_info(path fn)
    {
        fn = canonical(fn);
        CAFFE_ENFORCE(exists(fn), "Value-info file \"" + fn.string() + "\" doesn't exist.");
        CAFFE_ENFORCE(!is_directory(fn), "Get directory \"" + fn.string() + "\" while expecting value-info file.");

        string json;
        {
            ifstream fin(fn);
            CAFFE_ENFORCE(fin.is_open(), "Cannot open \"" + fn.string() + "\".");
            for (string buf; fin >> buf; json += buf + "\n");
        }

        smatch mt_single;
        CAFFE_ENFORCE(regex_match(json, mt_single, gr_single), "Syntax error in \"" + fn.string() + "\":\n" + string(80, '-') + "\n" + json + "\n" + string(80, '-'));
        CHECK_EQ(mt_single.size(), 4) << "Wrong number of matched components.";
        value_info.reset(new ValueInfo);
        value_info->input = mt_single[1];
        value_info->type = static_cast<onnx::TensorProto_DataType>(stoi(mt_single[2]));
        {
            sregex_iterator iter(mt_single[3].first, mt_single[3].second, gr_dim, regex_constants::match_continuous);
            auto dims_str = static_cast<string>(mt_single[3]);
            auto pending_size = dims_str.size();
            for (const sregex_iterator end; iter != end; ++iter)
            {
                value_info->dims.emplace_back(stoi((*iter)[1]));
                pending_size -= iter->str().size();
            }
            CHECK_EQ(pending_size, 0) << "Syntax error (cannot parse the entire input) in " << fn << ":\n" + string(80, '-') + "\n" + dims_str + "\n" + string(80, '-');
        }
    }

    SIPHON_HIDDEN
    void Siphon::save_value_info(const path& fn)
    {
        if (value_info)
        {
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
    }
}
