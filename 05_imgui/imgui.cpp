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

#include "imgui.h"

// the imgui default
static_assert(sizeof(ImDrawVert) == 20);
// switched to uint in imconfig.h
static_assert(sizeof(ImDrawIdx) == 4);

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
    void gui();
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
    std::vector<uint32_t> gui_ibuf_offsets;
    WGPUShaderModule gui_shader_module = nullptr;
    WGPUBuffer gui_vbuf = nullptr;
    uint32_t gui_vbuf_size;
    WGPUBuffer gui_ibuf = nullptr;
    WGPUBuffer gui_ubuf = nullptr;
    WGPUTexture gui_font_texture = nullptr;
    WGPUTextureView gui_font_texture_view = nullptr;
    WGPUSampler gui_sampler = nullptr;
    WGPUBindGroupLayout gui_bgl = nullptr;
    WGPUPipelineLayout gui_pl = nullptr;
    WGPURenderPipeline gui_ps = nullptr;
    WGPUBindGroup gui_bg = nullptr;
    Size last_gui_win_size;

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

static WGPUTexture rebuild_gui_font_atlas()
{
    ImGuiIO &io(ImGui::GetIO());
    unsigned char *pixels;
    int w, h;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &w, &h);

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
    wgpuQueueWriteTexture(d.queue, &dst_desc, pixels, w * h * 4, &data_layout, &write_size);

    return texture;
}

static void next_gui_frame()
{
    ImGuiIO &io(ImGui::GetIO());

    io.DisplaySize.x = d.win_size.width;
    io.DisplaySize.y = d.win_size.height;
    io.DisplayFramebufferScale = ImVec2(d.dpr, d.dpr);

    ImGui::NewFrame();
    d.scene.gui();
    ImGui::Render();

    ImDrawData *draw = ImGui::GetDrawData();
    d.gui_ibuf_offsets.clear();
    d.gui_ibuf_offsets.reserve(draw->CmdListsCount);
    std::vector<ImDrawVert> vbuf_data;
    std::vector<ImDrawIdx> ibuf_data;
    uint32_t vbuf_byte_size = 0;
    uint32_t ibuf_byte_size = 0;
    for (int n = 0; n < draw->CmdListsCount; ++n) {
        const ImDrawList *cmd_list = draw->CmdLists[n];
        uint32_t vbuf_offset = vbuf_byte_size;
        vbuf_byte_size += cmd_list->VtxBuffer.Size * sizeof(ImDrawVert);
        std::copy(cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Data + cmd_list->VtxBuffer.Size, std::back_inserter(vbuf_data));
        uint32_t ibuf_offset = ibuf_byte_size;
        ibuf_byte_size += cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx);
        std::copy(cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Data + cmd_list->IdxBuffer.Size, std::back_inserter(ibuf_data));
        d.gui_ibuf_offsets.push_back(ibuf_offset);
    }
    d.gui_vbuf_size = vbuf_byte_size;

    if (d.gui_vbuf && wgpuBufferGetSize(d.gui_vbuf) < vbuf_byte_size) {
        wgpuBufferDestroy(d.gui_vbuf);
        d.gui_vbuf = nullptr;
    }

    if (!d.gui_vbuf)
        d.gui_vbuf = create_buffer_with_data(WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst, vbuf_byte_size, vbuf_data.data());
    else
        wgpuQueueWriteBuffer(d.queue, d.gui_vbuf, 0, vbuf_data.data(), vbuf_byte_size);

    if (d.gui_ibuf && wgpuBufferGetSize(d.gui_ibuf) < ibuf_byte_size) {
        wgpuBufferDestroy(d.gui_ibuf);
        d.gui_ibuf = nullptr;
    }

    if (!d.gui_ibuf)
        d.gui_ibuf = create_buffer_with_data(WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst, ibuf_byte_size, ibuf_data.data());
    else
        wgpuQueueWriteBuffer(d.queue, d.gui_ibuf, 0, ibuf_data.data(), ibuf_byte_size);

    if (d.last_gui_win_size != d.win_size) {
        d.last_gui_win_size = d.win_size;
        HMM_Mat4 mvp = HMM_Orthographic_RH_ZO(0, d.win_size.width, d.win_size.height, 0, 1, -1);
        wgpuQueueWriteBuffer(d.queue, d.gui_ubuf, 0, &mvp.Elements[0][0], 64);
    }
}

static void render_gui(WGPURenderPassEncoder pass)
{
    ImDrawData *draw = ImGui::GetDrawData();
    draw->ScaleClipRects(ImVec2(d.dpr, d.dpr));

    for (int n = 0; n < draw->CmdListsCount; ++n) {
        const ImDrawList *cmd_list = draw->CmdLists[n];
        const ImDrawIdx *index_buf_offset = nullptr;
        for (int i = 0; i < cmd_list->CmdBuffer.Size; ++i) {
            const ImDrawCmd *cmd = &cmd_list->CmdBuffer[i];
            if (!cmd->UserCallback) {
                const uint32_t index_offset = d.gui_ibuf_offsets[n] + uintptr_t(index_buf_offset);
                // ### clamp clip rect
                wgpuRenderPassEncoderSetScissorRect(pass, cmd->ClipRect.x, cmd->ClipRect.y, cmd->ClipRect.z - cmd->ClipRect.x, cmd->ClipRect.w - cmd->ClipRect.y);
                wgpuRenderPassEncoderSetPipeline(pass, d.gui_ps);
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, d.gui_vbuf, 0, d.gui_vbuf_size);
                wgpuRenderPassEncoderSetIndexBuffer(pass, d.gui_ibuf, WGPUIndexFormat_Uint32, index_offset, cmd->ElemCount * 4);
                wgpuRenderPassEncoderSetBindGroup(pass, 0, d.gui_bg, 0, nullptr);
                wgpuRenderPassEncoderDrawIndexed(pass, cmd->ElemCount, 1, 0, 0, 0);
            } else {
                cmd->UserCallback(cmd_list, cmd);
            }
            index_buf_offset += cmd->ElemCount;
        }
    }
}

static void init_gui_renderer()
{
    static const char *shaders = R"end(
    struct Uniforms {
        mvp : mat4x4<f32>
    }
    @group(0) @binding(0) var<uniform> u : Uniforms;

    struct VertexOutput {
        @builtin(position) Position : vec4<f32>,
        @location(0) uv : vec2<f32>,
        @location(1) color : vec4<f32>
    }

    @vertex fn v_main(@location(0) position : vec4<f32>, @location(1) uv : vec2<f32>, @location(2) color : vec4<f32>) -> VertexOutput {
        var output : VertexOutput;
        output.Position = u.mvp * vec4<f32>(position.xy, 0.0, 1.0);
        output.uv = uv;
        output.color = color;
        return output;
    }

    @group(0) @binding(1) var tex : texture_2d<f32>;
    @group(0) @binding(2) var samp : sampler;

    @fragment fn f_main(@location(0) uv : vec2<f32>, @location(1) color : vec4<f32>) -> @location(0) vec4<f32> {
        var c = color * textureSample(tex, samp, uv);
        // ???!!!
        // c.rgb *= c.a;
        c.r *= c.a;
        c.g *= c.a;
        c.b *= c.a;
        return c;
    }
    )end";

    d.gui_shader_module = create_shader_module(shaders);

    d.gui_font_texture = rebuild_gui_font_atlas();

    WGPUTextureViewDescriptor view_desc = {
        .format = WGPUTextureFormat_RGBA8Unorm,
        .dimension = WGPUTextureViewDimension_2D,
        .mipLevelCount = 1,
        .arrayLayerCount = 1
    };
    d.gui_font_texture_view = wgpuTextureCreateView(d.gui_font_texture, &view_desc);

    WGPUSamplerDescriptor samplerDesc = {
        .addressModeU = WGPUAddressMode_Repeat,
        .addressModeV = WGPUAddressMode_Repeat,
        .magFilter = WGPUFilterMode_Linear,
        .minFilter = WGPUFilterMode_Linear
    };
    d.gui_sampler = wgpuDeviceCreateSampler(d.device, &samplerDesc);

    WGPUBindGroupLayoutEntry bgl_entries[] = {
        {
            .binding = 0,
            .visibility = WGPUShaderStage_Vertex,
            .buffer = {
                .type = WGPUBufferBindingType_Uniform,
                .minBindingSize = 64
            }
        },
        {
            .binding = 1,
            .visibility = WGPUShaderStage_Fragment,
            .texture = {
                .sampleType = WGPUTextureSampleType_Float,
                .viewDimension = WGPUTextureViewDimension_2D
            }
        },
        {
            .binding = 2,
            .visibility = WGPUShaderStage_Fragment,
            .sampler {
                .type = WGPUSamplerBindingType_Filtering
            }
        }
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
        .entryCount = 3,
        .entries = bgl_entries
    };
    d.gui_bgl = wgpuDeviceCreateBindGroupLayout(d.device, &bgl_desc);

    WGPUPipelineLayoutDescriptor pl_desc = {
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &d.gui_bgl
    };
    d.gui_pl = wgpuDeviceCreatePipelineLayout(d.device, &pl_desc);

    WGPUDepthStencilState ds_state = {
        .format = WGPUTextureFormat_Depth24PlusStencil8,
        .depthWriteEnabled = false,
        .depthCompare = WGPUCompareFunction_Less
    };

    WGPUBlendState blend = {
        .color = {
            .srcFactor = WGPUBlendFactor_One,
            .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha
        },
        .alpha = {
            .srcFactor = WGPUBlendFactor_One,
            .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha
        }
    };
    WGPUColorTargetState color0 = {
        .format = WGPUTextureFormat_BGRA8Unorm,
        .blend = &blend,
        .writeMask = WGPUColorWriteMask_All
    };

    WGPUFragmentState fs = {
        .module = d.gui_shader_module,
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
        },
        {
            .format = WGPUVertexFormat_Unorm8x4,
            .offset = 4 * sizeof(float),
            .shaderLocation = 2
        }
    };
    WGPUVertexBufferLayout vbuf_layout = {
        .arrayStride = 4 * sizeof(float) + sizeof(uint32_t),
        .attributeCount = 3,
        .attributes = vertex_attrs
    };

    WGPURenderPipelineDescriptor ps_desc = {
        .layout = d.gui_pl,
        .vertex = {
            .module = d.gui_shader_module,
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
    d.gui_ps = wgpuDeviceCreateRenderPipeline(d.device, &ps_desc);

    d.gui_ubuf = create_uniform_buffer(64);

    WGPUBindGroupEntry bg_entries[] = {
        {
            .binding = 0,
            .buffer = d.gui_ubuf,
            .size = 64,
        },
        {
            .binding = 1,
            .textureView = d.gui_font_texture_view
        },
        {
            .binding = 2,
            .sampler = d.gui_sampler
        }
    };
    WGPUBindGroupDescriptor bg_desc = {
        .layout = d.gui_bgl,
        .entryCount = 3,
        .entries = bg_entries
    };
    d.gui_bg = wgpuDeviceCreateBindGroup(d.device, &bg_desc);
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

static EM_BOOL mouse_callback(int emsc_type, const EmscriptenMouseEvent *emsc_event, void *user_data)
{
    ImGuiIO &io(ImGui::GetIO());
    const float x = float(emsc_event->targetX);
    const float y = float(emsc_event->targetY);

    switch (emsc_type) {
    case EMSCRIPTEN_EVENT_MOUSEMOVE:
        io.AddMousePosEvent(x, y);
        return true;
    case EMSCRIPTEN_EVENT_MOUSEENTER:
        break;
    case EMSCRIPTEN_EVENT_MOUSELEAVE:
        break;
    default:
        break;
    }

    if (emsc_event->button >= 0 && emsc_event->button <= 2) {
        io.AddKeyEvent(ImGuiKey_ModCtrl, emsc_event->ctrlKey);
        io.AddKeyEvent(ImGuiKey_ModShift, emsc_event->shiftKey);
        io.AddKeyEvent(ImGuiKey_ModAlt, emsc_event->altKey);
        io.AddKeyEvent(ImGuiKey_ModSuper, emsc_event->metaKey);
        int imgui_button = 0;
        switch (emsc_event->button) {
        case 1:
            imgui_button = 2;
            break;
        case 2:
            imgui_button = 1;
            break;
        default:
            break;
        }
        switch (emsc_type) {
        case EMSCRIPTEN_EVENT_MOUSEDOWN:
            io.AddMouseButtonEvent(imgui_button, true);
            break;
        case EMSCRIPTEN_EVENT_MOUSEUP:
            io.AddMouseButtonEvent(imgui_button, false);
            break;
        default:
            break;
        }
    }

    return true;
}

static EM_BOOL wheel_callback(int emsc_type, const EmscriptenWheelEvent *emsc_event, void *user_data)
{
    ImGuiIO &io(ImGui::GetIO());
    const float x = float(emsc_event->deltaX / 120.0f);
    const float y = float(emsc_event->deltaY / 120.0f);
    io.AddMouseWheelEvent(x, y);
    return true;
}

static EM_BOOL key_callback(int emsc_type, const EmscriptenKeyboardEvent *emsc_event, void *user_data)
{
    // ###

    bool result = false; // don't consume
    switch (emsc_type) {
    case EMSCRIPTEN_EVENT_KEYDOWN:
        break;
    case EMSCRIPTEN_EVENT_KEYUP:
        break;
    case EMSCRIPTEN_EVENT_KEYPRESS:
        break;
    default:

        break;
    }
    return result;
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

    init_gui_renderer();

    d.scene.init();
}

static void cleanup()
{
    d.scene.cleanup();

    if (d.gui_font_texture) {
        wgpuTextureDestroy(d.gui_font_texture);
        d.gui_font_texture = nullptr;
    }
    if (d.gui_font_texture_view) {
        wgpuTextureViewRelease(d.gui_font_texture_view);
        d.gui_font_texture_view = nullptr;
    }
    if (d.gui_sampler) {
        wgpuSamplerRelease(d.gui_sampler);
        d.gui_sampler = nullptr;
    }
    if (d.gui_vbuf) {
        wgpuBufferDestroy(d.gui_vbuf);
        d.gui_vbuf = nullptr;
    }
    if (d.gui_ibuf) {
        wgpuBufferDestroy(d.gui_ibuf);
        d.gui_ibuf = nullptr;
    }
    if (d.gui_ubuf) {
        wgpuBufferDestroy(d.gui_ubuf);
        d.gui_ubuf = nullptr;
    }
    if (d.gui_shader_module) {
        wgpuShaderModuleRelease(d.gui_shader_module);
        d.gui_shader_module = nullptr;
    }
    if (d.gui_bgl) {
        wgpuBindGroupLayoutRelease(d.gui_bgl);
        d.gui_bgl = nullptr;
    }
    if (d.gui_pl) {
        wgpuPipelineLayoutRelease(d.gui_pl);
        d.gui_pl = nullptr;
    }
    if (d.gui_ps) {
        wgpuRenderPipelineRelease(d.gui_ps);
        d.gui_ps = nullptr;
    }
    if (d.gui_bg) {
        wgpuBindGroupRelease(d.gui_bg);
        d.gui_bg = nullptr;
    }

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

    ImGui::DestroyContext();
}

static void frame()
{
    if (d.swapchain) {
        begin_frame();
        next_gui_frame();
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
    ImGui::CreateContext();

    update_size();

    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, 0, false, size_changed);

    emscripten_set_mousedown_callback("canvas", nullptr, true, mouse_callback);
    emscripten_set_mouseup_callback("canvas", nullptr, true, mouse_callback);
    emscripten_set_mousemove_callback("canvas", nullptr, true, mouse_callback);
    emscripten_set_mouseenter_callback("canvas", nullptr, true, mouse_callback);
    emscripten_set_mouseleave_callback("canvas", nullptr, true, mouse_callback);

    emscripten_set_wheel_callback("canvas", nullptr, true, wheel_callback);

    emscripten_set_keydown_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true, key_callback);
    emscripten_set_keyup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true, key_callback);
    emscripten_set_keypress_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true, key_callback);

    init_wgpu([](WGPUDevice dev) {
        d.device = dev;
        init();
        emscripten_set_main_loop(frame, 0, false);
    });

    return 0;
}

// ----------

struct SceneData
{
    ~SceneData();

    void start_load_assets();
    bool assets_ready() const;
    void init();

    bool initialized = false;

    bool show_demo_window = true;
};

void SceneData::start_load_assets()
{
}

bool SceneData::assets_ready() const
{
    return true;
}

template <class Int>
inline Int aligned(Int v, Int byteAlign)
{
    return (v + byteAlign - 1) & ~(byteAlign - 1);
}

void SceneData::init()
{
}

SceneData::~SceneData()
{
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

void Scene::gui()
{
    ImGuiIO &io(ImGui::GetIO());
    io.IniFilename = nullptr; // no imgui.ini

    ImGui::ShowDemoWindow(&sd->show_demo_window);
}

void Scene::render()
{
    if (!sd->assets_ready()) {
        WGPUColor loading_clear_color = { 1.0f, 1.0f, 1.0f, 1.0f };
        WGPURenderPassEncoder pass = begin_render_pass(loading_clear_color);
        end_render_pass(pass);
        return;
    }

    if (!sd->initialized) {
        sd->init();
        sd->initialized = true;
    }

    WGPUColor clear_color = { 0.0f, 1.0f, 0.0f, 1.0f };
    WGPURenderPassEncoder pass = begin_render_pass(clear_color);

    render_gui(pass);

    end_render_pass(pass);
}
