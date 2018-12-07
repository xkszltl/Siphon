#pragma once

#ifdef _MSC_VER
    #define SIPHON_IMPORT __declspec(dllimport)
    #define SIPHON_EXPORT __declspec(dllexport)
    #define SIPHON_HIDDEN
#elif __GNUC__
    #define SIPHON_IMPORT [[gnu::visibility("default")]]
    #define SIPHON_EXPORT SIPHON_IMPORT
    #define SIPHON_HIDDEN [[gnu::visibility("hidden")]]
#else
    #define SIPHON_IMPORT __attribute__((__visibility__("default")))
    #define SIPHON_EXPORT SIPHON_IMPORT
    #define SIPHON_HIDDEN __attribute__((__visibility__("hidden")))
#endif

#ifdef siphon_cpu_EXPORTS
    #define SIPHON_API SIPHON_EXPORT
#else
    #define SIPHON_API SIPHON_IMPORT
#endif
