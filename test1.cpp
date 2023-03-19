#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webgpu.h>
#include <webgpu/webgpu.h>
#include <stdio.h>
#include <math.h>

struct Size
{
    uint32_t width = 0;
    uint32_t height = 0;
};

inline bool operator==(const Size &a, const Size &b)
{
    return a.width == b.width && a.height == b.height;
}

inline bool operator!=(const Size &a, const Size &b)
{
    return !(a == b);
}

struct
{
    Size win_size;
    Size fb_size;
    float dpr = 0.0f;

    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUSwapChain swapchain = nullptr;
    WGPUTextureView backbuffer = nullptr;

    Size attachments_size;
    WGPUTexture ds = nullptr;
    WGPUTextureView ds_view = nullptr;
} d;

static void update_size()
{
    double w, h;
    emscripten_get_element_css_size("#canvas", &w, &h);
    double dpr = emscripten_get_device_pixel_ratio();

    d.win_size.width = uint32_t(roundf(w));
    d.win_size.height = uint32_t(roundf(h));
    d.fb_size.width = uint32_t(roundf(w * dpr));
    d.fb_size.height = uint32_t(roundf(h * dpr));
    d.dpr = float(dpr);

    emscripten_set_canvas_element_size("#canvas", d.fb_size.width, d.fb_size.height);

    printf("size: win %dx%d fb %dx%d dpr %f\n", d.win_size.width, d.win_size.height, d.fb_size.width, d.fb_size.height, d.dpr);
}

static EM_BOOL size_changed(int event_type, const EmscriptenUiEvent *ui_event, void *user_data)
{
    update_size();
    return true;
}

static void ensure_attachments()
{
    if (d.ds && d.ds_view && d.attachments_size == d.fb_size)
        return;

    if (d.ds_view) {
        wgpuTextureViewRelease(d.ds_view);
        d.ds_view = nullptr;
    }

    if (d.ds) {
        wgpuTextureDestroy(d.ds);
        d.ds = nullptr;
    }

    d.attachments_size = d.fb_size;

    WGPUTextureDescriptor desc = {};
    desc.usage = WGPUTextureUsage_RenderAttachment;
    desc.dimension = WGPUTextureDimension_2D;
    desc.size.width = d.attachments_size.width;
    desc.size.height = d.attachments_size.height;
    desc.size.depthOrArrayLayers = 1;
    desc.format = WGPUTextureFormat_Depth24PlusStencil8;
    desc.mipLevelCount = 1;
    desc.sampleCount = 1;
    d.ds = wgpuDeviceCreateTexture(d.device, &desc);
    d.ds_view = wgpuTextureCreateView(d.ds, nullptr);

    printf("Created depth-stencil %dx%d (%p, %p)\n", d.attachments_size.width, d.attachments_size.height, d.ds, d.ds_view);
}

static void render()
{
    if (!d.swapchain)
        return;

    if (d.backbuffer) {
        wgpuTextureViewRelease(d.backbuffer);
        d.backbuffer = nullptr;
    }

    d.backbuffer = wgpuSwapChainGetCurrentTextureView(d.swapchain);

    ensure_attachments();

    WGPURenderPassColorAttachment attachment = {};
    attachment.view = d.backbuffer;
    attachment.loadOp = WGPULoadOp_Clear;
    attachment.storeOp = WGPUStoreOp_Store;
    attachment.clearValue = { 0, 1, 0, 1};

    WGPURenderPassDescriptor renderpass = {};
    renderpass.colorAttachmentCount = 1;
    renderpass.colorAttachments = &attachment;

    WGPUCommandBuffer cb;
    {
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(d.device, nullptr);
        {
            WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &renderpass);
            wgpuRenderPassEncoderEnd(pass);
        }
        cb = wgpuCommandEncoderFinish(encoder, nullptr);
    }
    wgpuQueueSubmit(d.queue, 1, &cb);
}

static void frame()
{
    render();
}

using InitWGpuCallback = void (*)(WGPUDevice);

static void init_wgpu(InitWGpuCallback callback)
{
    static const WGPUInstance instance = nullptr;
    wgpuInstanceRequestAdapter(instance, nullptr, [](WGPURequestAdapterStatus status, WGPUAdapter adapter, const char* message, void* userdata) {
        if (message)
            printf("wgpuInstanceRequestAdapter: %s\n", message);
        if (status == WGPURequestAdapterStatus_Unavailable) {
            puts("WebGPU unavailable");
            exit(0);
        }
        wgpuAdapterRequestDevice(adapter, nullptr, [](WGPURequestDeviceStatus status, WGPUDevice dev, const char* message, void* userdata) {
            if (message)
                printf("wgpuAdapterRequestDevice: %s\n", message);
            reinterpret_cast<InitWGpuCallback>(userdata)(dev);
        }, userdata);
    }, reinterpret_cast<void *>(callback));
}

static void init()
{
    wgpuDeviceSetUncapturedErrorCallback(d.device, [](WGPUErrorType errorType, const char* message, void*) {
        printf("%d: %s\n", errorType, message);
    }, nullptr);

    d.queue = wgpuDeviceGetQueue(d.device);

    WGPUSurfaceDescriptorFromCanvasHTMLSelector canvasDesc = {};
    canvasDesc.chain.sType = WGPUSType_SurfaceDescriptorFromCanvasHTMLSelector;
    canvasDesc.selector = "#canvas";

    WGPUSurfaceDescriptor surfDesc = {};
    surfDesc.nextInChain = &canvasDesc.chain;
    WGPUSurface surface = wgpuInstanceCreateSurface(nullptr, &surfDesc);

    WGPUSwapChainDescriptor scDesc = {};
    scDesc.usage = WGPUTextureUsage_RenderAttachment;
    scDesc.format = WGPUTextureFormat_BGRA8Unorm;
    scDesc.width = d.fb_size.width;
    scDesc.height = d.fb_size.height;
    scDesc.presentMode = WGPUPresentMode_Fifo;
    d.swapchain = wgpuDeviceCreateSwapChain(d.device, surface, &scDesc);

    printf("Created swapchain %dx%d (%p)\n", d.fb_size.width, d.fb_size.height, d.swapchain);
}

int main()
{
    update_size();
    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, 0, false, size_changed);

    init_wgpu([](WGPUDevice dev) {
        d.device = dev;
        init();
        emscripten_set_main_loop(frame, 0, false);
    });

    return 0;
}
