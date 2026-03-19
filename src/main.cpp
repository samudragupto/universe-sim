#include "application.h"
#include <cstdio>

// --- FIX FOR LAPTOP DUAL-GPU (OPTIMUS) ---
// This forces the Windows graphics drivers to bind the OpenGL context 
// to the dedicated NVIDIA GPU instead of the integrated AMD/Intel iGPU.
// Without this, CUDA and OpenGL cannot share memory on laptops.
#ifdef _WIN32
#include <windows.h>
extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif
// -----------------------------------------

int main(int argc, char** argv) {
    Application app;
    
    if (!app.init()) {
        fprintf(stderr, "Initialization failed\n");
        return 1;
    }
    
    app.run();
    app.shutdown();
    
    return 0;
}