#include <siphon/core.h>

#include <regex>

using namespace std;

namespace siphon
{
    const regex Siphon::gr_multi("^", regex::optimize);

    const regex Siphon::gr_single("^"
            "\\s*\\{"
                "\\s*\"([^\"]+)\""
                "\\s*:"
                "\\s*\\["
                    "\\s*([0-9]+)"
                    "\\s*,"
                    "\\s*\\[("
                        "\\s*\\d+"
                        "(?:\\s*,\\s*\\d+)*"
                    "\\s*)\\]"
                "\\s*\\]"
            "\\s*\\}"
        "\\s*$", regex::optimize);

    const regex Siphon::gr_dim("\\s*(\\d+)\\s*(?:,|$)", regex::optimize);
}
