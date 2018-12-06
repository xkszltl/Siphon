#include "siphon/core.h"
#include "siphon/init.h"

#include <caffe2/core/logging.h>
#include <caffe2/utils/proto_utils.h>

#include <locale>
#include <set>
#include <string>

using namespace std;

#if __has_include(<filesystem>)
    using namespace filesystem;
#else
    using namespace experimental::filesystem;
#endif

using namespace caffe2;

namespace siphon
{
    Siphon::Siphon()
    {
        Init();
    }

    NetDef Siphon::LoadC2(path fn)
    {
        fn = canonical(fn);
        CAFFE_ENFORCE(exists(fn), "Caffe2 model file \"" + fn.string() + "\" doesn't exist.");
        CAFFE_ENFORCE(!is_directory(fn), "Get directory \"" + fn.string() + "\" while expecting Caffe2 model file.");

        NetDef net;
        CAFFE_ENFORCE(ReadProtoFromFile(fn.string(), &net), "Failed to read Caffe2 model \"" + fn.string() + "\".");
        return net;
    }

    void Siphon::Load(path dir)
    {
        dir = canonical(dir);
        CAFFE_ENFORCE(exists(dir), "Model directory \"" + dir.string() + "\" doesn't exist.");
        CAFFE_ENFORCE(is_directory(dir), "\"" + dir.string() + "\" is not a directory.");
        for (const auto& fn : directory_iterator(dir))
        {
            const auto& ext = fn.path().extension().string();
            if (ext == ".pb")
            {
                auto&& net = LoadC2(canonical(fn.path()));
                auto name = fn.path().stem().string();
                for (auto& c : name)
                {
                    c = tolower(c, locale());
                }
                if (set<string>{("init"), ("init_net")}.count(name))
                {
                    net.set_name("init");
                }
                else if (set<string>{"pred", "predict", "pred_net", "predict_net"}.count(name))
                {
                    net.set_name("pred");
                }
                ws.CreateNet(net);
            }
        }
    }
}
