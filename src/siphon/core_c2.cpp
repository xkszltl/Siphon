#include "siphon/core.h"

#include <caffe2/core/logging.h>
#include <caffe2/core/types.h>
#include <caffe2/utils/proto_utils.h>

#include <cstdint>
#include <fstream>
#include <locale>
#include <set>
#include <string>

using namespace std;

#if __has_include(<filesystem>)
    using namespace std::filesystem;
#else
    using namespace std::experimental::filesystem;
#endif

using namespace caffe2;

namespace siphon
{
    SIPHON_HIDDEN
    NetDef& Siphon::eval_fill(NetDef& net) const
    {
        for (size_t op_idx = 0; op_idx < net.op_size(); ++op_idx)
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

                for (size_t output_idx = 0; output_idx < op.output_size(); ++output_idx)
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
}
