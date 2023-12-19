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

using LocalFileLoadCallback = std::function<void(const char *filename, const char *mime_type, char *data, size_t size)>;
using LocalFileLoadFsApiCallback = std::function<void(const char *filename, char *data, size_t size)>;

struct
{
    Size win_size;
    Size fb_size;
    float dpr = 0.0f;

    WGPUInstance instance = nullptr;
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

    struct GuiBufOffset {
        uint32_t v_offset;
        uint32_t v_size;
        uint32_t i_offset;
        uint32_t i_size;
    };
    std::vector<GuiBufOffset> gui_buf_offsets;
    WGPUShaderModule gui_shader_module = nullptr;
    WGPUBuffer gui_vbuf = nullptr;
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
    LocalFileLoadCallback local_file_load_callback = nullptr;
    LocalFileLoadFsApiCallback local_file_load_fs_api_callback = nullptr;

    Scene scene;
} d;

WGPUShaderModule create_shader_module(const char *wgsl_source)
{
    WGPUShaderModuleWGSLDescriptor wgsl_desc = {
        .chain = {
            .sType = WGPUSType_ShaderModuleWGSLDescriptor
        },
        .code = wgsl_source
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

static void releaseAndNull(WGPUTexture &obj)
{
    if (obj) {
        wgpuTextureDestroy(obj);
        obj = nullptr;
    }
}

static void releaseAndNull(WGPUTextureView &obj)
{
    if (obj) {
        wgpuTextureViewRelease(obj);
        obj = nullptr;
    }
}

static void releaseAndNull(WGPUSampler &obj)
{
    if (obj) {
        wgpuSamplerRelease(obj);
        obj = nullptr;
    }
}

static void releaseAndNull(WGPUBuffer &obj)
{
    if (obj) {
        wgpuBufferDestroy(obj);
        obj = nullptr;
    }
}

static void releaseAndNull(WGPUShaderModule &obj)
{
    if (obj) {
        wgpuShaderModuleRelease(obj);
        obj = nullptr;
    }
}

static void releaseAndNull(WGPUBindGroupLayout &obj)
{
    if (obj) {
        wgpuBindGroupLayoutRelease(obj);
        obj = nullptr;
    }
}

static void releaseAndNull(WGPUPipelineLayout &obj)
{
    if (obj) {
        wgpuPipelineLayoutRelease(obj);
        obj = nullptr;
    }
}

static void releaseAndNull(WGPURenderPipeline &obj)
{
    if (obj) {
        wgpuRenderPipelineRelease(obj);
        obj = nullptr;
    }
}

static void releaseAndNull(WGPUBindGroup &obj)
{
    if (obj) {
        wgpuBindGroupRelease(obj);
        obj = nullptr;
    }
}

static void releaseAndNull(WGPUCommandEncoder &obj)
{
    if (obj) {
        wgpuCommandEncoderRelease(obj);
        obj = nullptr;
    }
}

static void releaseAndNull(WGPUSwapChain &obj)
{
    if (obj) {
        wgpuSwapChainRelease(obj);
        obj = nullptr;
    }
}

static void releaseAndNull(WGPUSurface &obj)
{
    if (obj) {
        wgpuSurfaceRelease(obj);
        obj = nullptr;
    }
}

static void releaseAndNull(WGPUQueue &obj)
{
    if (obj) {
        wgpuQueueRelease(obj);
        obj = nullptr;
    }
}

static void releaseAndNull(WGPUDevice &obj)
{
    if (obj) {
        wgpuDeviceRelease(obj);
        obj = nullptr;
    }
}

static void releaseAndNull(WGPUInstance &obj)
{
    if (obj) {
        wgpuInstanceRelease(obj);
        obj = nullptr;
    }
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
    d.gui_buf_offsets.clear();
    d.gui_buf_offsets.reserve(draw->CmdListsCount);
    std::vector<ImDrawVert> vbuf_data;
    std::vector<ImDrawIdx> ibuf_data;
    uint32_t vbuf_total_byte_size = 0;
    uint32_t ibuf_total_byte_size = 0;
    for (int n = 0; n < draw->CmdListsCount; ++n) {
        const ImDrawList *cmd_list = draw->CmdLists[n];
        uint32_t vbuf_offset = vbuf_total_byte_size;
        vbuf_total_byte_size += cmd_list->VtxBuffer.Size * sizeof(ImDrawVert);
        std::copy(cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Data + cmd_list->VtxBuffer.Size, std::back_inserter(vbuf_data));
        uint32_t ibuf_offset = ibuf_total_byte_size;
        ibuf_total_byte_size += cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx);
        std::copy(cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Data + cmd_list->IdxBuffer.Size, std::back_inserter(ibuf_data));
        d.gui_buf_offsets.push_back({
            vbuf_offset,
            cmd_list->VtxBuffer.Size * sizeof(ImDrawVert),
            ibuf_offset,
            cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx)
        });
    }

    if (d.gui_vbuf && wgpuBufferGetSize(d.gui_vbuf) < vbuf_total_byte_size) {
        wgpuBufferDestroy(d.gui_vbuf);
        d.gui_vbuf = nullptr;
    }

    if (!d.gui_vbuf)
        d.gui_vbuf = create_buffer_with_data(WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst, vbuf_total_byte_size, vbuf_data.data());
    else
        wgpuQueueWriteBuffer(d.queue, d.gui_vbuf, 0, vbuf_data.data(), vbuf_total_byte_size);

    if (d.gui_ibuf && wgpuBufferGetSize(d.gui_ibuf) < ibuf_total_byte_size) {
        wgpuBufferDestroy(d.gui_ibuf);
        d.gui_ibuf = nullptr;
    }

    if (!d.gui_ibuf)
        d.gui_ibuf = create_buffer_with_data(WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst, ibuf_total_byte_size, ibuf_data.data());
    else
        wgpuQueueWriteBuffer(d.queue, d.gui_ibuf, 0, ibuf_data.data(), ibuf_total_byte_size);

    if (d.last_gui_win_size != d.win_size) {
        d.last_gui_win_size = d.win_size;
        HMM_Mat4 mvp = HMM_Orthographic_RH_ZO(0, d.win_size.width, d.win_size.height, 0, 1, -1);
        wgpuQueueWriteBuffer(d.queue, d.gui_ubuf, 0, &mvp.Elements[0][0], 64);
    }
}

template<typename T>
bool clamp_scissor(const Size &renderTargetPixelSize, T *x, T *y, T *w, T *h)
{
    if (*w < 0 || *h < 0)
        return false;

    const T outputWidth = renderTargetPixelSize.width;
    const T outputHeight = renderTargetPixelSize.height;
    const T widthOffset = *x < 0 ? -*x : 0;
    const T heightOffset = *y < 0 ? -*y : 0;
    *w = *x < outputWidth ? std::max<T>(0, *w - widthOffset) : 0;
    *h = *y < outputHeight ? std::max<T>(0, *h - heightOffset) : 0;

    if (outputWidth > 0)
        *x = std::clamp<T>(*x, 0, outputWidth - 1);
    if (outputHeight > 0)
        *y = std::clamp<T>(*y, 0, outputHeight - 1);
    if (*x + *w > outputWidth)
        *w = std::max<T>(0, outputWidth - *x);
    if (*y + *h > outputHeight)
        *h = std::max<T>(0, outputHeight - *y);

    return true;
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
                const uint32_t index_offset = d.gui_buf_offsets[n].i_offset + uintptr_t(index_buf_offset);
                float sx = cmd->ClipRect.x;
                float sy = cmd->ClipRect.y;
                float sw = cmd->ClipRect.z - cmd->ClipRect.x;
                float sh = cmd->ClipRect.w - cmd->ClipRect.y;
                if (clamp_scissor(d.fb_size, &sx, &sy, &sw, &sh))
                    wgpuRenderPassEncoderSetScissorRect(pass, sx, sy, sw, sh);
                wgpuRenderPassEncoderSetPipeline(pass, d.gui_ps);
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, d.gui_vbuf, d.gui_buf_offsets[n].v_offset, d.gui_buf_offsets[n].v_size);
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
    if (d.quit)
        return false;

    update_size();
    return true;
}

static EM_BOOL mouse_callback(int emsc_type, const EmscriptenMouseEvent *emsc_event, void *user_data)
{
    if (d.quit)
        return false;

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
    if (d.quit)
        return false;

    ImGuiIO &io(ImGui::GetIO());
    const float x = float(emsc_event->deltaX / 120.0f);
    const float y = float(emsc_event->deltaY / -120.0f);
    io.AddMouseWheelEvent(x, y);
    return true;
}

static ImGuiKey mapKey(int k, bool *consume)
{
    *consume = false;
    switch (k) {
    case 8:
        *consume = true;
        return ImGuiKey_Backspace;
    case 9:
        *consume = true;
        return ImGuiKey_Tab;
    case 13:
        *consume = true;
        return ImGuiKey_Enter;
    case 16:
        *consume = true;
        return ImGuiKey_LeftShift;
    case 17:
        *consume = true;
        return ImGuiKey_LeftCtrl;
    case 18:
        *consume = true;
        return ImGuiKey_LeftAlt;
    case 20:
        *consume = true;
        return ImGuiKey_CapsLock;
    case 27:
        *consume = true;
        return ImGuiKey_Escape;
    case 32:
        return ImGuiKey_Space;
    case 33:
        *consume = true;
        return ImGuiKey_PageUp;
    case 34:
        *consume = true;
        return ImGuiKey_PageDown;
    case 35:
        *consume = true;
        return ImGuiKey_End;
    case 36:
        *consume = true;
        return ImGuiKey_Home;
    case 37:
        *consume = true;
        return ImGuiKey_LeftArrow;
    case 38:
        *consume = true;
        return ImGuiKey_UpArrow;
    case 39:
        *consume = true;
        return ImGuiKey_RightArrow;
    case 40:
        *consume = true;
        return ImGuiKey_DownArrow;
    case 45:
        *consume = true;
        return ImGuiKey_Insert;
    case 46:
        *consume = true;
        return ImGuiKey_Delete;
    case 48:
        return ImGuiKey_0;
    case 49:
        return ImGuiKey_1;
    case 50:
        return ImGuiKey_2;
    case 51:
        return ImGuiKey_3;
    case 52:
        return ImGuiKey_4;
    case 53:
        return ImGuiKey_5;
    case 54:
        return ImGuiKey_6;
    case 55:
        return ImGuiKey_7;
    case 56:
        return ImGuiKey_8;
    case 57:
        return ImGuiKey_9;
    case 59:
        return ImGuiKey_Semicolon;
    case 64:
        return ImGuiKey_Equal;
    case 65:
        return ImGuiKey_A;
    case 66:
        return ImGuiKey_B;
    case 67:
        return ImGuiKey_C;
    case 68:
        return ImGuiKey_D;
    case 69:
        return ImGuiKey_E;
    case 70:
        return ImGuiKey_F;
    case 71:
        return ImGuiKey_G;
    case 72:
        return ImGuiKey_H;
    case 73:
        return ImGuiKey_I;
    case 74:
        return ImGuiKey_J;
    case 75:
        return ImGuiKey_K;
    case 76:
        return ImGuiKey_L;
    case 77:
        return ImGuiKey_M;
    case 78:
        return ImGuiKey_N;
    case 79:
        return ImGuiKey_O;
    case 80:
        return ImGuiKey_P;
    case 81:
        return ImGuiKey_Q;
    case 82:
        return ImGuiKey_R;
    case 83:
        return ImGuiKey_S;
    case 84:
        return ImGuiKey_T;
    case 85:
        return ImGuiKey_U;
    case 86:
        return ImGuiKey_V;
    case 87:
        return ImGuiKey_W;
    case 88:
        return ImGuiKey_X;
    case 89:
        return ImGuiKey_Y;
    case 90:
        return ImGuiKey_Z;
    case 91:
        *consume = true;
        return ImGuiKey_LeftSuper;
    case 93:
        *consume = true;
        return ImGuiKey_Menu;
    case 96:
        return ImGuiKey_Keypad0;
    case 97:
        return ImGuiKey_Keypad1;
    case 98:
        return ImGuiKey_Keypad2;
    case 99:
        return ImGuiKey_Keypad3;
    case 100:
        return ImGuiKey_Keypad4;
    case 101:
        return ImGuiKey_Keypad5;
    case 102:
        return ImGuiKey_Keypad6;
    case 103:
        return ImGuiKey_Keypad7;
    case 104:
        return ImGuiKey_Keypad8;
    case 105:
        return ImGuiKey_Keypad9;
    case 106:
        return ImGuiKey_KeypadMultiply;
    case 107:
        return ImGuiKey_KeypadAdd;
    case 109:
        return ImGuiKey_KeypadSubtract;
    case 110:
        return ImGuiKey_KeypadDecimal;
    case 111:
        return ImGuiKey_KeypadDivide;
    case 112:
        *consume = true;
        return ImGuiKey_F1;
    case 113:
        *consume = true;
        return ImGuiKey_F2;
    case 114:
        *consume = true;
        return ImGuiKey_F3;
    case 115:
        *consume = true;
        return ImGuiKey_F4;
    case 116:
        *consume = true;
        return ImGuiKey_F5;
    case 117:
        *consume = true;
        return ImGuiKey_F6;
    case 118:
        *consume = true;
        return ImGuiKey_F7;
    case 119:
        *consume = true;
        return ImGuiKey_F8;
    case 120:
        *consume = true;
        return ImGuiKey_F9;
    case 121:
        *consume = true;
        return ImGuiKey_F10;
    case 122:
        return ImGuiKey_F11;
    case 123:
        return ImGuiKey_F12;
    case 144:
        *consume = true;
        return ImGuiKey_NumLock;
    case 145:
        *consume = true;
        return ImGuiKey_ScrollLock;
    case 173:
        return ImGuiKey_Minus;
    case 186:
        return ImGuiKey_Semicolon;
    case 187:
        return ImGuiKey_Equal;
    case 188:
        return ImGuiKey_Comma;
    case 189:
        return ImGuiKey_Minus;
    case 190:
        return ImGuiKey_Period;
    case 191:
        return ImGuiKey_Slash;
    case 192:
        return ImGuiKey_GraveAccent;
    case 219:
        return ImGuiKey_LeftBracket;
    case 220:
        return ImGuiKey_Backslash;
    case 221:
        return ImGuiKey_RightBracket;
    case 222:
        return ImGuiKey_Apostrophe;
    case 224:
        *consume = true;
        return ImGuiKey_LeftSuper;
    default:
        break;
    }
    return ImGuiKey_None;
}

static EM_BOOL key_callback(int emsc_type, const EmscriptenKeyboardEvent *emsc_event, void *user_data)
{
    if (d.quit)
        return false;

    ImGuiIO &io(ImGui::GetIO());
    io.AddKeyEvent(ImGuiKey_ModCtrl, emsc_event->ctrlKey);
    io.AddKeyEvent(ImGuiKey_ModShift, emsc_event->shiftKey);
    io.AddKeyEvent(ImGuiKey_ModAlt, emsc_event->altKey);
    io.AddKeyEvent(ImGuiKey_ModSuper, emsc_event->metaKey);

    bool consume = false;
    switch (emsc_type) {
    case EMSCRIPTEN_EVENT_KEYDOWN:
        io.AddKeyEvent(mapKey(emsc_event->keyCode, &consume), true);
        break;
    case EMSCRIPTEN_EVENT_KEYUP:
        io.AddKeyEvent(mapKey(emsc_event->keyCode, &consume), false);
        break;
    case EMSCRIPTEN_EVENT_KEYPRESS:
        if (strlen(emsc_event->key))
            io.AddInputCharactersUTF8(emsc_event->key);
        break;
    default:
        break;
    }
    return consume;
}

static void ensure_attachments()
{
    if (d.ds && d.ds_view && d.attachments_size == d.fb_size)
        return;

    releaseAndNull(d.ds_view);
    releaseAndNull(d.ds);

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
    releaseAndNull(d.backbuffer);
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
    releaseAndNull(d.res_encoder);

    WGPUCommandBuffer render_cb = wgpuCommandEncoderFinish(d.render_encoder, nullptr);
    releaseAndNull(d.render_encoder);

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
    d.surface = wgpuInstanceCreateSurface(d.instance, &surfDesc);

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

    releaseAndNull(d.gui_font_texture);
    releaseAndNull(d.gui_font_texture_view);
    releaseAndNull(d.gui_sampler);
    releaseAndNull(d.gui_vbuf);
    releaseAndNull(d.gui_ibuf);
    releaseAndNull(d.gui_ubuf);
    releaseAndNull(d.gui_shader_module);
    releaseAndNull(d.gui_bgl);
    releaseAndNull(d.gui_pl);
    releaseAndNull(d.gui_ps);
    releaseAndNull(d.gui_bg);

    releaseAndNull(d.ds_view);
    releaseAndNull(d.ds);
    releaseAndNull(d.backbuffer);
    releaseAndNull(d.swapchain);
    releaseAndNull(d.surface);
    releaseAndNull(d.queue);
    releaseAndNull(d.device);
    releaseAndNull(d.instance);

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
    }
}

using InitWGpuCallback = void (*)(WGPUInstance, WGPUDevice);

static void init_wgpu(InitWGpuCallback callback)
{
    WGPUInstanceDescriptor instanceDesc = {};
    static WGPUInstance instance;
    instance = wgpuCreateInstance(&instanceDesc);

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
            reinterpret_cast<InitWGpuCallback>(userdata)(instance, dev);
        }, userdata);
    }, reinterpret_cast<void *>(callback));
}

extern "C" {
EMSCRIPTEN_KEEPALIVE int _file_loaded(const char *filename, const char *mime_type, char *data, size_t size)
{
    if (d.local_file_load_callback)
        d.local_file_load_callback(filename, mime_type, data, size);
    return 1;
}
}

EM_JS(void, initialize_local_file_uploader, (), {
    globalThis["load_file"] = function(e) {
        const file_reader = new FileReader();
        file_reader.onload = (event) => {
            const data = new Uint8Array(event.target.result);
            const buf = Module._malloc(data.length);
            Module.HEAPU8.set(data, buf);
            Module.ccall('_file_loaded', 'number', ['string', 'string', 'number', 'number'],
                [event.target.filename, event.target.mime_type, buf, data.length]);
            Module._free(buf);
        };
        file_reader.filename = e.target.files[0].name;
        file_reader.mime_type = e.target.files[0].type;
        file_reader.readAsArrayBuffer(e.target.files[0]);
    };
    var file_selector = document.createElement('input');
    file_selector.setAttribute('type', 'file');
    file_selector.setAttribute('onchange', 'globalThis["load_file"](event)');
    globalThis["load_file_selector"] = file_selector;
});

EM_JS(void, begin_load_local_file, (const char *accept_types), {
    var file_selector = globalThis["load_file_selector"];
    file_selector.setAttribute('accept', UTF8ToString(accept_types));
    file_selector.click();
});

static void load_local_file(const char *accept_types, LocalFileLoadCallback callback)
{
    d.local_file_load_callback = callback;
    begin_load_local_file(accept_types);
}

EM_JS(void, initialize_local_file_downloader, (), {
    var a = document.createElement('a');
    globalThis["save_file_element"] = a;
});

EM_JS(void, begin_save_local_file, (const char *filename, const char *mime_type, const void *data, size_t size), {
    var data = new Uint8Array(Module.HEAPU8.buffer, data, size);
    var blob = new Blob([data]);
    var a = globalThis["save_file_element"];
    a.download = UTF8ToString(filename);
    a.href = URL.createObjectURL(blob, { type: UTF8ToString(mime_type) });
    a.click();
});

static void save_local_file(const char *filename, const char *mime_type, const void *data, size_t size)
{
    begin_save_local_file(filename, mime_type, data, size);
}

// Modern alternative using the Filesystem API

EM_JS(bool, has_fs_api, (), {
    if (window.showOpenFilePicker === undefined)
        return false;
    if (window.showSaveFilePicker === undefined)
        return false;
    return true;
});

extern "C" {
EMSCRIPTEN_KEEPALIVE int _file_loaded_fs_api(const char *filename, char *data, size_t size)
{
    if (d.local_file_load_fs_api_callback)
        d.local_file_load_fs_api_callback(filename, data, size);
    return 1;
}
}

EM_JS(void, begin_load_local_file_fs_api, (), {
    window.showOpenFilePicker().then((fileHandles) => {
        if (fileHandles.length > 0) {
            const data = fileHandles[0].getFile().then((file) => {
                file.arrayBuffer().then((result) => {
                    const data = new Uint8Array(result);
                    const buf = Module._malloc(data.length);
                    Module.HEAPU8.set(data, buf);
                    Module.ccall('_file_loaded_fs_api', 'number', ['string', 'number', 'number'],
                        [file.name, buf, data.length]);
                    Module._free(buf);
                });
            }).catch(err => { console.log(err); });
        }
    }).catch(err => {});
});

static void load_local_file_fs_api(LocalFileLoadFsApiCallback callback)
{
    d.local_file_load_fs_api_callback = callback;
    begin_load_local_file_fs_api();
}

EM_JS(void, begin_save_local_file_fs_api, (const char *filename, const void *data, size_t size), {
    window.showSaveFilePicker({
        "suggestedName": UTF8ToString(filename)
    }).then((fileHandle) => {
        fileHandle.createWritable().then((writableHandle) => {
            var arr = new Uint8Array(Module.HEAPU8.buffer, data, size);
            writableHandle.write(arr).then(() => {
                writableHandle.close();
            }).catch(err => { console.log(err); });
        }).catch(err => { console.log(err); });
    }).catch(err => {});
});

static void save_local_file_fs_api(const char *filename, const void *data, size_t size)
{
    begin_save_local_file_fs_api(filename, data, size);
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

    initialize_local_file_uploader();
    initialize_local_file_downloader();

    init_wgpu([](WGPUInstance instance, WGPUDevice dev) {
        d.instance = instance;
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
    std::unique_ptr<char[]> file_contents;
    size_t file_contents_alloc_size = 0;
    std::string filename;
    std::string mime_type;
};

void SceneData::start_load_assets()
{
}

bool SceneData::assets_ready() const
{
    return true;
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

    const int W = d.win_size.width - 200;
    const int H = d.win_size.height - 200;
    ImGui::SetNextWindowPos(ImVec2(d.win_size.width / 2 - W / 2, d.win_size.height / 2 - H / 2), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(W, H), ImGuiCond_FirstUseEver);
    ImGui::Begin("Notepad 2023");

    if (ImGui::Button("Quit")) {
        puts("quit");
        d.quit = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("New")) {
        sd->file_contents_alloc_size = 4096;
        sd->file_contents.reset(new char[sd->file_contents_alloc_size]);
        sd->file_contents[0] = '\0';
        sd->filename = "document.txt";
        sd->mime_type = "text/plain";
    }
    ImGui::SameLine();
    if (ImGui::Button("Open local text file")) {
        if (has_fs_api()) {
            load_local_file_fs_api([this](const char *filename, char *data, size_t size) {
                printf("load callback: %s %p %lu\n", filename, data, size);
                sd->file_contents_alloc_size = size * 2;
                sd->file_contents.reset(new char[sd->file_contents_alloc_size]);
                memcpy(sd->file_contents.get(), data, size);
                sd->file_contents[size] = '\0';
                sd->filename = filename;
            });
        } else {
            load_local_file("text/*", [this](const char *filename, const char *mime_type, char *data, size_t size) {
                printf("load callback: %s %s %p %lu\n", filename, mime_type, data, size);
                sd->file_contents_alloc_size = size * 2;
                sd->file_contents.reset(new char[sd->file_contents_alloc_size]);
                memcpy(sd->file_contents.get(), data, size);
                sd->file_contents[size] = '\0';
                sd->filename = filename;
                sd->mime_type = mime_type;
            });
        }
    }
    if (sd->file_contents) {
        ImGui::SameLine();
        if (ImGui::Button("Save As")) {
            if (has_fs_api())
                save_local_file_fs_api(sd->filename.c_str(), sd->file_contents.get(), strlen(sd->file_contents.get()));
            else
                save_local_file(sd->filename.c_str(), sd->mime_type.c_str(), sd->file_contents.get(), strlen(sd->file_contents.get()));
        }
        ImGui::TextUnformatted(sd->filename.c_str());
        ImGui::SameLine();
        ImGui::TextUnformatted(sd->mime_type.c_str());
        ImGui::InputTextMultiline("##textedit", sd->file_contents.get(), sd->file_contents_alloc_size, ImVec2(-FLT_MIN, -FLT_MIN));
    }

    ImGui::End();
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
