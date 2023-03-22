#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webgpu.h>
#include <webgpu/webgpu.h>
#include <stdio.h>
#include <math.h>
#include <memory>
#include <vector>

#define HANDMADE_MATH_USE_DEGREES
#include "../3rdparty/HandmadeMath/HandmadeMath.h"

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
    @location(0) color : vec3<f32>
}

@vertex fn v_main(@location(0) position : vec4<f32>, @location(1) color : vec3<f32>) -> VertexOutput {
    var output : VertexOutput;
    output.Position = u.mvp * position;
    output.color = color;
    return output;
}

@fragment fn f_main(@location(0) color : vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 1.0);
}
)end";

static float vertexData[] = {
     0.0f,   0.5f,  0.0f,    1.0f, 0.0f, 0.0f,
    -0.5f,  -0.5f,  0.0f,    0.0f, 1.0f, 0.0f,
     0.5f,  -0.5f,  0.0f,    0.0f, 0.0f, 1.0f
};

struct SceneData
{
    SceneData();
    ~SceneData();

    Size last_fb_size;
    WGPUShaderModule shader_module1;
    static const uint32_t UBUF_SIZE1 = 64;
    WGPUBuffer vbuf;
    uint32_t vbuf_size;
    WGPUBuffer ubuf;
    WGPUBindGroupLayout bgl;
    WGPUPipelineLayout pl;
    WGPURenderPipeline ps;
    WGPUBindGroup bg;

    float rotation = 0.0f;
    HMM_Mat4 projection_matrix;
    HMM_Mat4 view_matrix;
};

SceneData::SceneData()
{
    shader_module1 = create_shader_module(shaders1);
    vbuf_size = sizeof(vertexData);
    vbuf = create_buffer_with_data(WGPUBufferUsage_Vertex, vbuf_size, vertexData);
    ubuf = create_uniform_buffer(UBUF_SIZE1);

    WGPUBindGroupLayoutEntry bgl_entries[] = {
        {
            .binding = 0,
            .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
            .buffer = {
                .type = WGPUBufferBindingType_Uniform,
                .minBindingSize = UBUF_SIZE1
            }
        }
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
        .entryCount = 1,
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
            .format = WGPUVertexFormat_Float32x3,
            .offset = 0,
            .shaderLocation = 0
        },
        {
            .format = WGPUVertexFormat_Float32x3,
            .offset = 3 * sizeof(float),
            .shaderLocation = 1
        }
    };
    WGPUVertexBufferLayout vbuf_layout = {
        .arrayStride = 6 * sizeof(float),
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

    WGPUBindGroupEntry bg_entry = {
        .buffer = ubuf,
        .offset = 0,
        .size = UBUF_SIZE1
    };
    WGPUBindGroupDescriptor bg_desc = {
        .layout = bgl,
        .entryCount = 1,
        .entries = &bg_entry
    };
    bg = wgpuDeviceCreateBindGroup(d.device, &bg_desc);

    view_matrix = HMM_Translate(HMM_V3(0.0f, 0.0f, -4.0f));
}

SceneData::~SceneData()
{
    wgpuBindGroupRelease(bg);
    wgpuRenderPipelineRelease(ps);
    wgpuPipelineLayoutRelease(pl);
    wgpuBindGroupLayoutRelease(bgl);
    wgpuBufferDestroy(ubuf);
    wgpuBufferDestroy(vbuf);
    wgpuShaderModuleRelease(shader_module1);
}

void Scene::init()
{
    sd.reset(new SceneData);
}

void Scene::cleanup()
{
    sd.reset();
}

void Scene::render()
{
    if (sd->last_fb_size != d.fb_size) {
        sd->last_fb_size = d.fb_size;
        sd->projection_matrix = HMM_Perspective_RH_ZO(45.0f, float(d.fb_size.width) / d.fb_size.height, 0.01f, 1000.0f);
    }

    HMM_Mat4 model_matrix = HMM_Rotate_RH(sd->rotation, HMM_V3(0.0f, 1.0f, 0.0f));
    HMM_Mat4 view_projection_matrix = HMM_Mul(sd->projection_matrix, sd->view_matrix);
    HMM_Mat4 mvp = HMM_Mul(view_projection_matrix, model_matrix);

    UBufStagingArea u = next_ubuf_staging_area_for_current_frame();
    memcpy(u.p, &mvp.Elements[0][0], 16 * sizeof(float));
    enqueue_ubuf_staging_copy(u, sd->ubuf, SceneData::UBUF_SIZE1);

    sd->rotation += 1.0f;

    WGPUColor clear_color = { 0.0f, 1.0f, 0.0f, 1.0f };
    WGPURenderPassEncoder pass = begin_render_pass(clear_color);

    wgpuRenderPassEncoderSetPipeline(pass, sd->ps);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, sd->bg, 0, nullptr);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, sd->vbuf, 0, sd->vbuf_size);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);

    end_render_pass(pass);
}
