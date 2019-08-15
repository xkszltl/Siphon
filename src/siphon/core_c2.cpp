#include "siphon/core.h"

#include <caffe2/core/logging.h>
#include <caffe2/core/types.h>
#include <caffe2/opt/optimizer.h>
#include <caffe2/utils/proto_utils.h>

#include <pybind11/embed.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <locale>
#include <set>
#include <string>

using namespace std;
using namespace std::filesystem;

using namespace caffe2;
using namespace pybind11::literals;

namespace siphon
{
    namespace py = pybind11;

    SIPHON_HIDDEN
    NetDef& Siphon::eval_fill(NetDef& net) const
    {
        for (int64_t op_idx = 0; op_idx < net.op_size(); ++op_idx)
        {
            auto& op = *net.mutable_op(op_idx);
            if (!op.has_type())
            {
                continue;
            }
            if (!op.output_size())
            {
                continue;
            }

            if (op.type() == "ConstantFill")
            {
                LOG(INFO) << "Evaluate op:\n" + op.DebugString();

                Workspace tmp_ws;
                tmp_ws.RunOperatorOnce(op);

                for (int64_t output_idx = 0; output_idx < op.output_size(); ++output_idx)
                {
                    const auto& output = op.output(output_idx);

                    auto& tensor = *BlobGetMutableTensor(tmp_ws.GetBlob(output), dev_type);
                    const auto& dtype = TypeMetaToDataType(tensor.dtype());

                    OperatorDef dst_op;
                    dst_op.add_output(output);
                    if (op.has_name())
                    {
                        dst_op.set_name(op.name());
                    }
                    {
                        auto& arg = *dst_op.add_arg();
                        arg.set_name("dtype");
                        arg.set_i(static_cast<long long>(dtype));
                    }
                    {
                        auto& arg = *dst_op.add_arg();
                        arg.set_name("values");
                        const auto numel = static_cast<size_t>(tensor.numel());
                        switch (dtype)
                        {
                        case caffe2::TensorProto_DataType_FLOAT:
                            {
                                dst_op.set_type("GivenTensorFill");
                                LOG(INFO) << "Generate " << dst_op.type() << " op.";
                                const auto data = tensor.data<float>();
                                for (size_t i = 0; i < numel; arg.add_floats(data[i++]));
                            }
                            break;
                        case caffe2::TensorProto_DataType_INT32:
                            {
                                dst_op.set_type("GivenTensorIntFill");
                                LOG(INFO) << "Generate " << dst_op.type() << " op.";
                                const auto data = tensor.data<int>();
                                for (size_t i = 0; i < numel; arg.add_ints(data[i++]));
                            }
                            break;
                        case caffe2::TensorProto_DataType_INT64:
                            {
                                dst_op.set_type("GivenTensorInt64Fill");
                                LOG(INFO) << "Generate " << dst_op.type() << " op.";
                                const auto data = tensor.data<int64_t>();
                                for (size_t i = 0; i < numel; arg.add_ints(data[i++]));
                            }
                            break;
                        case caffe2::TensorProto_DataType_DOUBLE:
                            {
                                dst_op.set_type("GivenTensorDoubleFill");
                                LOG(INFO) << "Generate " << dst_op.type() << " op.";
                                const auto data = tensor.data<double>();
                                for (size_t i = 0; i < numel; arg.add_floats(data[i++]));
                            }
                            break;
                        default:
                            LOG(FATAL) << "Unsupported type " << caffe2::TensorProto_DataType_Name(dtype) << ".";
                        }
                    }
                    {
                        auto& device_option = *dst_op.mutable_device_option();
                        device_option.set_device_type(static_cast<int>(dev_type));
                    }

                    if (output_idx)
                    {
                        LOG(INFO) << "Append new op to network.";
                        *(net.add_op()) = move(dst_op);
                    }
                    else
                    {
                        LOG(INFO) << "Inject new op in-place.";
                        op = move(dst_op);
                    }
                }
            }
        }
        return net;
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

    SIPHON_HIDDEN
    void Siphon::optimize_c2()
    {
        string init_lvl = "init";
        string pred_lvl = "pred";

        CAFFE_ENFORCE(nets.count("init"), "Init net doesn't exist.");
        CAFFE_ENFORCE(nets.count("pred"), "Predict net doesn't exist.");

        {
            const auto& before = nets["pred"].DebugString();
            nets["init_O1"] = opt::optimize(nets["init"]);
            const auto& after = nets["pred"].DebugString();
            if (before != after)
            {
                LOG(INFO) << "Init network optimized with graph optimization.";
                init_lvl = "init_O1";
            }
        }

        {
            const auto& before = nets["pred"].DebugString();
            nets["pred_O1"] = opt::optimize(nets["pred"]);
            const auto& after = nets["pred"].DebugString();
            if (before != after)
            {
                LOG(INFO) << "Predict network optimized with graph optimization.";
                init_lvl = "pred_O1";
            }
        }

        string init_str;
        string pred_str;
        nets[init_lvl].SerializeToString(&init_str);
        nets[pred_lvl].SerializeToString(&pred_str);

        set<string> static_blobs;
        for (const auto& blob_name : nets[pred_lvl].external_input())
            static_blobs.emplace(blob_name);
        for (const auto& blob_name : nets[pred_lvl].external_output())
            static_blobs.emplace(blob_name);
        auto input_blobs = static_blobs;
        for (const auto& blob_name : nets[init_lvl].external_output())
        {
            static_blobs.emplace(blob_name);
            input_blobs.erase(blob_name);
        }

        string pred_opt_str;

        pyenv.exec([&]()
            {
                py::list static_blobs_py;
                for (const auto& blob_name : static_blobs)
                    static_blobs_py.append(blob_name);

                py::list input_blobs_py;
                for (const auto& blob_name : input_blobs)
                    input_blobs_py.append(blob_name);

                auto proto_module = pyenv.import("caffe2.proto.caffe2_pb2");
                auto memonger_module = pyenv.import("caffe2.python.memonger");

                LOG(INFO) << "Deserialize init network in python.";
                auto init_py = proto_module.attr("NetDef")();
                init_py.attr("ParseFromString")(py::bytes(init_str));

                LOG(INFO) << "Deserialize predict network in python.";
                auto pred_py = proto_module.attr("NetDef")();
                pred_py.attr("ParseFromString")(py::bytes(pred_str));

                LOG(INFO) << "Optimizing predict network in python.";
                auto optimize_interference = memonger_module.attr("optimize_interference");
                auto optimize_inference_fast = memonger_module.attr("optimize_inference_fast");
                auto optimize_inference_for_dag = memonger_module.attr("optimize_inference_for_dag");
                auto pred_opt_tuple_py = optimize_interference(pred_py, static_blobs_py);
                auto pred_opt_py = pred_opt_tuple_py.attr("net");
                // auto pred_opt_py = optimize_inference_fast(pred_py, static_blobs_py);
                // auto pred_opt_py = optimize_inference_for_dag(pred_py, input_blobs_py);

                LOG(INFO) << "Serialize predict network and send back to C++.";
                {
                    py::bytes pred_opt_str_py = pred_opt_py.attr("SerializeToString")();
                    pred_opt_str = static_cast<string>(pred_opt_str_py);
                }
            });

        if (pred_opt_str == pred_str)
            LOG(INFO) << "No memonger optimzation available for predict network.";
        else
        {
            LOG(INFO) << "Deserialize predict network in C++.";
            NetDef pred_opt;
            CAFFE_ENFORCE(ParseProtoFromLargeString(pred_opt_str, &pred_opt), "Failed to deserialize optimized predict network.");
            nets["pred_O2"] = pred_opt;
            LOG(INFO) << "Predict network optimized with memonger.";
            pred_lvl = "pred_O2";
        }
    }
}
