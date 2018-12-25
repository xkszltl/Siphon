#include "siphon/core.h"

#include <caffe2/core/logging.h>
#include <caffe2/utils/proto_utils.h>

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
