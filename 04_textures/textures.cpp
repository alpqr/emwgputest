#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webgpu.h>
#include <webgpu/webgpu.h>
#include <stdio.h>
#include <math.h>
#include <memory>
#include <functional>
#include <vector>
#include <string>

#define HANDMADE_MATH_USE_DEGREES
#include "../3rdparty/HandmadeMath/HandmadeMath.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../3rdparty/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../3rdparty/stb/stb_image_write.h"

#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#include "../3rdparty/tinyexr/tinyexr.h"

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

struct SceneData;

struct Scene
{
    void init();
    void cleanup();
    void render();

    std::unique_ptr<SceneData> sd;
};

static const uint32_t MAX_UBUF_SIZE = 65536;

using LoadWebTextureCallback = std::function<void(WGPUTexture)>;

struct
{
    Size win_size;
    Size fb_size;
    float dpr = 0.0f;

    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUSurface surface = nullptr;
    WGPUSwapChain swapchain = nullptr;
    WGPUTextureView backbuffer = nullptr;

    Size attachments_size;
    WGPUTexture ds = nullptr;
    WGPUTextureView ds_view = nullptr;

    WGPUCommandEncoder res_encoder = nullptr;
    WGPUCommandEncoder render_encoder = nullptr;
    std::vector<WGPUBuffer> free_ubuf_staging_buffers;
    std::vector<WGPUBuffer> active_ubuf_staging_buffers;

    std::vector<std::pair<std::string, LoadWebTextureCallback>> pending_web_texture_loads;

    bool quit = false;

    Scene scene;
} d;

WGPUShaderModule create_shader_module(const char *wgsl_source)
{
    WGPUShaderModuleWGSLDescriptor wgsl_desc = {
        .chain = {
            .sType = WGPUSType_ShaderModuleWGSLDescriptor
        },
        .source = wgsl_source
    };
    WGPUShaderModuleDescriptor desc = {
        .nextInChain = &wgsl_desc.chain
    };
    return wgpuDeviceCreateShaderModule(d.device, &desc);
}

static WGPUBuffer create_buffer(WGPUBufferUsageFlags usage, uint64_t size, bool mapped = false)
{
    WGPUBufferDescriptor desc = {
        .usage = usage,
        .size = size,
        .mappedAtCreation = mapped
    };
    return wgpuDeviceCreateBuffer(d.device, &desc);
}

static WGPUBuffer create_buffer_with_data(WGPUBufferUsageFlags usage, uint64_t size, const void *data, uint32_t data_size = 0)
{
    WGPUBuffer buffer = create_buffer(usage, size, true);
    char *p = static_cast<char *>(wgpuBufferGetMappedRange(buffer, 0, size));
    memcpy(p, data, data_size ? data_size : size);
    wgpuBufferUnmap(buffer);
    return buffer;
}

static WGPUBuffer create_uniform_buffer(uint64_t size)
{
    return create_buffer(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, size);
}

static WGPUBuffer create_staging_buffer(uint64_t size)
{
    return create_buffer(WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc, size, true);
}

struct UBufStagingArea
{
    char *p;
    WGPUBuffer buf;
};

static UBufStagingArea next_ubuf_staging_area_for_current_frame()
{
    WGPUBuffer buf;
    if (d.free_ubuf_staging_buffers.empty()) {
        buf = create_staging_buffer(MAX_UBUF_SIZE);
    } else {
        buf = d.free_ubuf_staging_buffers.back();
        d.free_ubuf_staging_buffers.pop_back();
    }
    d.active_ubuf_staging_buffers.push_back(buf);
    return {
        static_cast<char *>(wgpuBufferGetMappedRange(buf, 0, MAX_UBUF_SIZE)),
        buf
    };
}

static void enqueue_ubuf_staging_copy(const UBufStagingArea &u, WGPUBuffer dst, uint32_t size, uint32_t src_offset = 0, uint32_t dst_offset = 0)
{
    wgpuCommandEncoderCopyBufferToBuffer(d.res_encoder, u.buf, src_offset, dst, dst_offset, size);
}

extern "C" {
EMSCRIPTEN_KEEPALIVE void _web_texture_loaded(int textureId, const char *uri)
{
    WGPUTexture texture = reinterpret_cast<WGPUTexture>(textureId);
    for (auto it = d.pending_web_texture_loads.begin(); it != d.pending_web_texture_loads.end(); ++it) {
        if (it->first == uri) {
            it->second(texture);
            d.pending_web_texture_loads.erase(it);
            break;
        }
    }
}
}

EM_JS(void, _begin_load_web_texture, (int deviceId, const char *uri), {
    const device = WebGPU.mgrDevice.get(deviceId);
    fetch(UTF8ToString(uri)).then((response) => {
        response.blob().then((blob) => {
            createImageBitmap(blob).then((imgBitmap) => {
                const textureDescriptor = {
                    size: { width: imgBitmap.width, height: imgBitmap.height },
                    format: 'rgba8unorm',
                    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
                };
                const texture = device.createTexture(textureDescriptor);
                const textureId = WebGPU.mgrTexture.create(texture);
                device.queue.copyExternalImageToTexture({ source: imgBitmap }, { texture: texture }, textureDescriptor.size);
                __web_texture_loaded(textureId, uri);
            });
        });
    });
});

static void load_web_texture(const char *uri, LoadWebTextureCallback callback)
{
    d.pending_web_texture_loads.push_back({ uri, callback });
    _begin_load_web_texture(reinterpret_cast<int>(d.device), uri);
}

static WGPUTexture load_texture(const char *filename)
{
    int w, h, n;
    unsigned char *data = stbi_load(filename, &w, &h, &n, 4);
    if (!data) {
        printf("load_texture: %s\n", stbi_failure_reason());
        return nullptr;
    }

    WGPUTextureFormat view_format = WGPUTextureFormat_RGBA8Unorm;
    WGPUTextureDescriptor desc = {
        .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
        .dimension = WGPUTextureDimension_2D,
        .size = {
            .width = uint32_t(w),
            .height = uint32_t(h),
            .depthOrArrayLayers = 1
        },
        .format = WGPUTextureFormat_RGBA8Unorm,
        .mipLevelCount = 1,
        .sampleCount = 1,
        .viewFormatCount = 1,
        .viewFormats = &view_format
    };
    WGPUTexture texture = wgpuDeviceCreateTexture(d.device, &desc);

    WGPUImageCopyTexture dst_desc = {
        .texture = texture
    };
    WGPUTextureDataLayout data_layout = {
        .offset = 0,
        .bytesPerRow = uint32_t(w * 4),
        .rowsPerImage = uint32_t(h)
    };
    WGPUExtent3D write_size = {
        .width = uint32_t(w),
        .height = uint32_t(h),
        .depthOrArrayLayers = 1
    };
    wgpuQueueWriteTexture(d.queue, &dst_desc, data, w * h * 4, &data_layout, &write_size);

    stbi_image_free(data);
    return texture;
}

static WGPUTexture load_exr_simple_f32(const char *filename)
{
    float *data;
    int w, h;
    const char *err = nullptr;
    const int ret = LoadEXR(&data, &w, &h, filename, &err);
    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            printf("load_exr: %s\n", err);
            FreeEXRErrorMessage(err);
        }
        return nullptr;
    }

    WGPUTextureFormat view_format = WGPUTextureFormat_RGBA32Float;
    WGPUTextureDescriptor desc = {
        .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
        .dimension = WGPUTextureDimension_2D,
        .size = {
            .width = uint32_t(w),
            .height = uint32_t(h),
            .depthOrArrayLayers = 1
        },
        .format = WGPUTextureFormat_RGBA32Float,
        .mipLevelCount = 1,
        .sampleCount = 1,
        .viewFormatCount = 1,
        .viewFormats = &view_format
    };
    WGPUTexture texture = wgpuDeviceCreateTexture(d.device, &desc);

    WGPUImageCopyTexture dst_desc = {
        .texture = texture
    };
    WGPUTextureDataLayout data_layout = {
        .offset = 0,
        .bytesPerRow = uint32_t(w * 16),
        .rowsPerImage = uint32_t(h)
    };
    WGPUExtent3D write_size = {
        .width = uint32_t(w),
        .height = uint32_t(h),
        .depthOrArrayLayers = 1
    };
    wgpuQueueWriteTexture(d.queue, &dst_desc, data, w * h * 16, &data_layout, &write_size);

    free(data);
    return texture;
}

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

    WGPUTextureDescriptor desc = {
        .usage = WGPUTextureUsage_RenderAttachment,
        .dimension = WGPUTextureDimension_2D,
        .size = {
            .width = d.attachments_size.width,
            .height = d.attachments_size.height,
            .depthOrArrayLayers = 1
        },
        .format = WGPUTextureFormat_Depth24PlusStencil8,
        .mipLevelCount = 1,
        .sampleCount = 1
    };
    d.ds = wgpuDeviceCreateTexture(d.device, &desc);
    d.ds_view = wgpuTextureCreateView(d.ds, nullptr);

    printf("Created depth-stencil %dx%d (%p, %p)\n", d.attachments_size.width, d.attachments_size.height, d.ds, d.ds_view);
}

static void begin_frame()
{
    if (d.backbuffer) {
        wgpuTextureViewRelease(d.backbuffer);
        d.backbuffer = nullptr;
    }

    d.backbuffer = wgpuSwapChainGetCurrentTextureView(d.swapchain);

    ensure_attachments();

    d.res_encoder = wgpuDeviceCreateCommandEncoder(d.device, nullptr);
    d.render_encoder = wgpuDeviceCreateCommandEncoder(d.device, nullptr);
}

static void end_frame()
{
    for (WGPUBuffer buf : d.active_ubuf_staging_buffers)
        wgpuBufferUnmap(buf);

    WGPUCommandBuffer res_cb = wgpuCommandEncoderFinish(d.res_encoder, nullptr);
    wgpuCommandEncoderRelease(d.res_encoder);
    d.res_encoder = nullptr;

    WGPUCommandBuffer render_cb = wgpuCommandEncoderFinish(d.render_encoder, nullptr);
    wgpuCommandEncoderRelease(d.render_encoder);
    d.render_encoder = nullptr;

    WGPUCommandBuffer cbs[] = { res_cb, render_cb };
    wgpuQueueSubmit(d.queue, 2, cbs);

    wgpuCommandBufferRelease(render_cb);
    wgpuCommandBufferRelease(res_cb);

    for (WGPUBuffer buf : d.active_ubuf_staging_buffers) {
        wgpuBufferMapAsync(buf, WGPUMapMode_Write, 0, MAX_UBUF_SIZE, [](WGPUBufferMapAsyncStatus status, void *userdata) {
            d.free_ubuf_staging_buffers.push_back(static_cast<WGPUBuffer>(userdata));
        }, buf);
    }
    d.active_ubuf_staging_buffers.clear();
}

static WGPURenderPassEncoder begin_render_pass(WGPUColor clear_color, float depth_clear_value = 1.0f, uint32_t stencil_clear_value = 0)
{
    WGPURenderPassColorAttachment attachment = {
        .view = d.backbuffer,
        .loadOp = WGPULoadOp_Clear,
        .storeOp = WGPUStoreOp_Store,
        .clearValue = clear_color
    };

    WGPURenderPassDepthStencilAttachment depthStencilAttachment = {
        .view = d.ds_view,
        .depthLoadOp = WGPULoadOp_Clear,
        .depthStoreOp = WGPUStoreOp_Discard,
        .depthClearValue = depth_clear_value,
        .stencilLoadOp = WGPULoadOp_Clear,
        .stencilStoreOp = WGPUStoreOp_Discard,
        .stencilClearValue = stencil_clear_value
    };

    WGPURenderPassDescriptor renderpass = {
        .colorAttachmentCount = 1,
        .colorAttachments = &attachment,
        .depthStencilAttachment = &depthStencilAttachment
    };

    return wgpuCommandEncoderBeginRenderPass(d.render_encoder, &renderpass);
}

static void end_render_pass(WGPURenderPassEncoder pass)
{
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
}

static void init()
{
    wgpuDeviceSetUncapturedErrorCallback(d.device, [](WGPUErrorType errorType, const char* message, void*) {
        printf("%d: %s\n", errorType, message);
    }, nullptr);

    d.queue = wgpuDeviceGetQueue(d.device);

    WGPUSurfaceDescriptorFromCanvasHTMLSelector canvasDesc = {
        .chain = {
            .sType = WGPUSType_SurfaceDescriptorFromCanvasHTMLSelector
        },
        .selector = "#canvas"
    };

    WGPUSurfaceDescriptor surfDesc = {
        .nextInChain = &canvasDesc.chain
    };
    d.surface = wgpuInstanceCreateSurface(nullptr, &surfDesc);

    WGPUSwapChainDescriptor scDesc = {
        .usage = WGPUTextureUsage_RenderAttachment,
        .format = WGPUTextureFormat_BGRA8Unorm,
        .width = d.fb_size.width,
        .height = d.fb_size.height,
        .presentMode = WGPUPresentMode_Fifo
    };
    d.swapchain = wgpuDeviceCreateSwapChain(d.device, d.surface, &scDesc);

    printf("Created swapchain %dx%d (%p)\n", d.fb_size.width, d.fb_size.height, d.swapchain);

    d.scene.init();
}

static void cleanup()
{
    d.scene.cleanup();

    if (d.ds_view) {
        wgpuTextureViewRelease(d.ds_view);
        d.ds_view = nullptr;
    }
    if (d.ds) {
        wgpuTextureDestroy(d.ds);
        d.ds = nullptr;
    }
    if (d.backbuffer) {
        wgpuTextureViewRelease(d.backbuffer);
        d.backbuffer = nullptr;
    }
    if (d.swapchain) {
        wgpuSwapChainRelease(d.swapchain);
        d.swapchain = nullptr;
    }
    if (d.surface) {
        wgpuSurfaceRelease(d.surface);
        d.surface = nullptr;
    }
    if (d.queue) {
        wgpuQueueRelease(d.queue);
        d.queue = nullptr;
    }
    if (d.device) {
        wgpuDeviceRelease(d.device);
        d.device = nullptr;
    }
}

static void frame()
{
    if (d.swapchain) {
        begin_frame();
        d.scene.render();
        end_frame();
    }

    if (d.quit) {
        cleanup();
        emscripten_cancel_main_loop();
        exit(0);
    }
}

using InitWGpuCallback = void (*)(WGPUDevice);

static void init_wgpu(InitWGpuCallback callback)
{
    wgpuInstanceRequestAdapter(nullptr, nullptr, [](WGPURequestAdapterStatus status, WGPUAdapter adapter, const char* message, void* userdata) {
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

// ----------

static const char *shaders1 = R"end(
struct Uniforms {
    mvp : mat4x4<f32>,
}
@binding(0) @group(0) var<uniform> u : Uniforms;

struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) uv : vec2<f32>
}

@vertex fn v_main(@location(0) position : vec4<f32>, @location(1) uv : vec2<f32>) -> VertexOutput {
    var output : VertexOutput;
    output.Position = u.mvp * position;
    output.uv = uv;
    return output;
}

@group(0) @binding(1) var tex : texture_2d<f32>;
@group(0) @binding(2) var samp : sampler;

@fragment fn f_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, uv);
}
)end";

static float vertexData[] = {
    -0.5f,   0.5f,     0.0f, 0.0f,
    -0.5f,  -0.5f,     0.0f, 1.0f,
     0.5f,  -0.5f,     1.0f, 1.0f,
    -0.5f,   0.5f,     0.0f, 0.0f,
     0.5f,  -0.5f,     1.0f, 1.0f,
     0.5f,   0.5f,     1.0f, 0.0f
};

struct SceneData
{
    ~SceneData();

    void start_load_assets();
    bool are_assets_ready() const;
    void init_with_assets();

    bool ready = false;
    Size last_fb_size;
    WGPUShaderModule shader_module1 = nullptr;
    static const uint32_t UBUF_SIZE1 = 64;
    WGPUBuffer vbuf = nullptr;
    uint32_t vbuf_size = 0;
    WGPUBuffer ubuf = nullptr;
    WGPUBindGroupLayout bgl = nullptr;
    WGPUPipelineLayout pl = nullptr;
    WGPURenderPipeline ps = nullptr;

    WGPUBindGroup bg_rgba = nullptr;
    WGPUBindGroup bg_float = nullptr;
    WGPUTexture texturergba = nullptr;
    WGPUTexture texturefloat = nullptr;
    WGPUSampler sampler;
    WGPUTextureView texturefloatView;
    WGPUTextureView texturergbaView;

    float rotation = 0.0f;
    HMM_Mat4 projection_matrix;
    HMM_Mat4 view_matrix;
};

void SceneData::start_load_assets()
{
    texturergba = load_texture("test.png");
    texturefloat = load_exr_simple_f32("test.exr");
    printf("texturergba = %p texturefloat=%p\n", texturergba, texturefloat);
}

bool SceneData::are_assets_ready() const
{
    return true;
}

template <class Int>
inline Int aligned(Int v, Int byteAlign)
{
    return (v + byteAlign - 1) & ~(byteAlign - 1);
}

void SceneData::init_with_assets()
{
    shader_module1 = create_shader_module(shaders1);
    vbuf_size = sizeof(vertexData);
    vbuf = create_buffer_with_data(WGPUBufferUsage_Vertex, vbuf_size, vertexData);
    ubuf = create_uniform_buffer(aligned(UBUF_SIZE1, 256u) * 2);

    WGPUSamplerDescriptor samplerDesc = {
        .addressModeU = WGPUAddressMode_ClampToEdge,
        .addressModeV = WGPUAddressMode_ClampToEdge
        // Nearest, to play nice if RGBA32F is non-filterable
    };
    sampler = wgpuDeviceCreateSampler(d.device, &samplerDesc);

    WGPUTextureViewDescriptor viewDesc_rgba = {
        .format = WGPUTextureFormat_RGBA8Unorm,
        .dimension = WGPUTextureViewDimension_2D,
        .mipLevelCount = 1,
        .arrayLayerCount = 1
    };
    texturergbaView = wgpuTextureCreateView(texturergba, &viewDesc_rgba);
    WGPUTextureViewDescriptor viewDesc_float = {
        .format = WGPUTextureFormat_RGBA32Float,
        .dimension = WGPUTextureViewDimension_2D,
        .mipLevelCount = 1,
        .arrayLayerCount = 1
    };
    texturefloatView = wgpuTextureCreateView(texturefloat, &viewDesc_float);

    WGPUBindGroupLayoutEntry bgl_entries[] = {
        {
            .binding = 0,
            .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
            .buffer = {
                .type = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = true,
                .minBindingSize = UBUF_SIZE1
            }
        },
        {
            .binding = 1,
            .visibility = WGPUShaderStage_Fragment,
            .texture = {
                .sampleType = WGPUTextureSampleType_UnfilterableFloat,
                .viewDimension = WGPUTextureViewDimension_2D
            }
        },
        {
            .binding = 2,
            .visibility = WGPUShaderStage_Fragment,
            .sampler {
                .type = WGPUSamplerBindingType_NonFiltering // for RGBA32F
            }
        }
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
        .entryCount = 3,
        .entries = bgl_entries
    };
    bgl = wgpuDeviceCreateBindGroupLayout(d.device, &bgl_desc);

    WGPUPipelineLayoutDescriptor pl_desc = {
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &bgl
    };
    pl = wgpuDeviceCreatePipelineLayout(d.device, &pl_desc);

    WGPUDepthStencilState ds_state = {
        .format = WGPUTextureFormat_Depth24PlusStencil8,
        .depthWriteEnabled = true,
        .depthCompare = WGPUCompareFunction_Less
    };

    WGPUColorTargetState color0 = {
        .format = WGPUTextureFormat_BGRA8Unorm,
        .writeMask = WGPUColorWriteMask_All
    };

    WGPUFragmentState fs = {
        .module = shader_module1,
        .entryPoint = "f_main",
        .targetCount = 1,
        .targets = &color0
    };

    WGPUVertexAttribute vertex_attrs[] = {
        {
            .format = WGPUVertexFormat_Float32x2,
            .offset = 0,
            .shaderLocation = 0
        },
        {
            .format = WGPUVertexFormat_Float32x2,
            .offset = 2 * sizeof(float),
            .shaderLocation = 1
        }
    };
    WGPUVertexBufferLayout vbuf_layout = {
        .arrayStride = 4 * sizeof(float),
        .attributeCount = 2,
        .attributes = vertex_attrs
    };

    WGPURenderPipelineDescriptor ps_desc = {
        .layout = pl,
        .vertex = {
            .module = shader_module1,
            .entryPoint = "v_main",
            .bufferCount = 1,
            .buffers = &vbuf_layout
        },
        .primitive = {
            .topology = WGPUPrimitiveTopology_TriangleList
        },
        .depthStencil = &ds_state,
        .multisample = {
            .count = 1,
            .mask = 0xFFFFFFFF
        },
        .fragment = &fs
    };
    ps = wgpuDeviceCreateRenderPipeline(d.device, &ps_desc);

    WGPUBindGroupEntry bg_entries[] = {
        {
            .binding = 0,
            .buffer = ubuf,
            .offset = 0,
            .size = UBUF_SIZE1
        },
        {
            .binding = 1,
            .textureView = texturergbaView
        },
        {
            .binding = 2,
            .sampler = sampler
        }
    };
    WGPUBindGroupDescriptor bg_desc = {
        .layout = bgl,
        .entryCount = 3,
        .entries = bg_entries
    };
    bg_rgba = wgpuDeviceCreateBindGroup(d.device, &bg_desc);
    bg_entries[1].textureView = texturefloatView;
    bg_float = wgpuDeviceCreateBindGroup(d.device, &bg_desc);

    view_matrix = HMM_Translate(HMM_V3(0.0f, 0.0f, -4.0f));
}

SceneData::~SceneData()
{
    wgpuTextureDestroy(texturergba);
    wgpuTextureDestroy(texturefloat);
    wgpuBindGroupRelease(bg_rgba);
    wgpuBindGroupRelease(bg_float);
    wgpuRenderPipelineRelease(ps);
    wgpuPipelineLayoutRelease(pl);
    wgpuBindGroupLayoutRelease(bgl);
    wgpuTextureViewRelease(texturergbaView);
    wgpuTextureViewRelease(texturefloatView);
    wgpuSamplerRelease(sampler);
    wgpuBufferDestroy(ubuf);
    wgpuBufferDestroy(vbuf);
    wgpuShaderModuleRelease(shader_module1);
}

void Scene::init()
{
    sd.reset(new SceneData);
    sd->start_load_assets();
}

void Scene::cleanup()
{
    sd.reset();
}

void Scene::render()
{
    if (!sd->are_assets_ready()) {
        WGPUColor loading_clear_color = { 1.0f, 1.0f, 1.0f, 1.0f };
        WGPURenderPassEncoder pass = begin_render_pass(loading_clear_color);
        end_render_pass(pass);
        return;
    }

    if (!sd->ready) {
        sd->init_with_assets();
        sd->ready = true;
    }

    if (sd->last_fb_size != d.fb_size) {
        sd->last_fb_size = d.fb_size;
        sd->projection_matrix = HMM_Perspective_RH_ZO(45.0f, float(d.fb_size.width) / d.fb_size.height, 0.01f, 1000.0f);
    }

    HMM_Mat4 view_projection_matrix = HMM_Mul(sd->projection_matrix, sd->view_matrix);
    HMM_Mat4 rotation = HMM_Rotate_RH(sd->rotation, HMM_V3(0.0f, 1.0f, 0.0f));

    HMM_Mat4 translation = HMM_Translate(HMM_V3(-2.0f, 0.0f, 0.0f));
    HMM_Mat4 model_matrix1 = HMM_Mul(translation, rotation);
    translation = HMM_Translate(HMM_V3(2.0f, 0.0f, 0.0f));
    HMM_Mat4 model_matrix2 = HMM_Mul(translation, rotation);

    UBufStagingArea u = next_ubuf_staging_area_for_current_frame();
    HMM_Mat4 mvp = HMM_Mul(view_projection_matrix, model_matrix1);
    memcpy(u.p, &mvp.Elements[0][0], 64);
    mvp = HMM_Mul(view_projection_matrix, model_matrix2);
    memcpy(u.p + 64, &mvp.Elements[0][0], 64);

    const uint32_t second_buffer_start_offset = aligned(SceneData::UBUF_SIZE1, 256u);
    enqueue_ubuf_staging_copy(u, sd->ubuf, SceneData::UBUF_SIZE1);
    enqueue_ubuf_staging_copy(u, sd->ubuf, SceneData::UBUF_SIZE1, 64, second_buffer_start_offset);

    sd->rotation += 1.0f;

    WGPUColor clear_color = { 0.0f, 1.0f, 0.0f, 1.0f };
    WGPURenderPassEncoder pass = begin_render_pass(clear_color);

    wgpuRenderPassEncoderSetPipeline(pass, sd->ps);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, sd->vbuf, 0, sd->vbuf_size);
    uint32_t dynamic_offset = 0;
    wgpuRenderPassEncoderSetBindGroup(pass, 0, sd->bg_rgba, 1, &dynamic_offset);
    wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
    dynamic_offset = second_buffer_start_offset;
    wgpuRenderPassEncoderSetBindGroup(pass, 0, sd->bg_float, 1, &dynamic_offset);
    wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);

    end_render_pass(pass);
}
