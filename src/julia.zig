const builtin = @import("builtin");
const std = @import("std");
const c = @import("c.zig");
const math = std.math;
const comptimePrint = std.fmt.comptimePrint;

const window_name = "quaternion julia sets";
const window_width = 1920;
const window_height = 1080;
var oglppo: c.GLuint = 0;

// zig fmt: off
const cs_image_a = struct {
const group_size_x = 8;
const group_size_y = 8;
const frame_loc = 0;
const time_loc = 1;
const resolution_loc = 2;
const image_unit = 0;
const src =
\\  #version 460 core
\\
++ "layout(\n"
++ "    local_size_x = " ++ comptimePrint("{d}", .{group_size_x}) ++ ",\n"
++ "    local_size_y = " ++ comptimePrint("{d}", .{group_size_y}) ++ ") in;\n" ++
\\
++ "layout(location = " ++ comptimePrint("{d}", .{frame_loc}) ++ ") uniform int u_frame;\n"
++ "layout(location = " ++ comptimePrint("{d}", .{time_loc}) ++ ") uniform float u_time;\n"
++ "layout(location = " ++ comptimePrint("{d}", .{resolution_loc}) ++ ") uniform vec2 u_resolution;\n" ++
\\
++ "layout(rgba32f, binding = " ++ comptimePrint("{d}", .{image_unit}) ++ ") uniform image2D u_image;\n" ++
\\
\\  //#define Z2
\\  const float k_foc_len = 3.0;
\\  const float k_bounding_sphere_rad = 1.2;
\\  const float k_precis = 0.00025;
\\  const int k_num_iter = 200;
\\  const vec4 k_c = vec4(-2.0, 6.0, 15.0, -6.0) / 22.0;
\\
\\  int seed = 1;
\\  void srand(int s) {
\\      seed = s;
\\  }
\\  int rand() {
\\      seed = seed * 0x343fd + 0x269ec3;
\\      return (seed >> 16) & 32767;
\\  }
\\  float frand() {
\\      return float(rand()) / 32767.0;
\\  }
\\
\\  int hash(int n) {
\\      n = (n << 13) ^ n;
\\      return n * (n * n * 15731 + 789221) + 1376312589;
\\  }
\\
\\  vec4 qSquare(vec4 q) {
\\      return vec4(
\\          q.x * q.x - q.y * q.y - q.z * q.z - q.w * q.w,
\\          2.0 * q.x * q.yzw);
\\  }
\\  vec4 qCube(vec4 q) {
\\      vec4 q2 = q * q;
\\      return vec4(
\\          q.x * (q2.x - 3.0 * q2.y - 3.0 * q2.z - 3.0 * q2.w),
\\          q.yzw * (3.0 * q2.x - q2.y - q2.z - q2.w));
\\  }
\\  vec4 qMul(vec4 q0, vec4 q1) {
\\      return vec4(
\\          q0.x * q1.x - q0.y * q1.y - q0.z * q1.z - q0.w * q1.w,
\\          q0.y * q1.x + q0.x * q1.y + q0.z * q1.w - q0.w * q1.z,
\\          q0.z * q1.x + q0.x * q1.z + q0.w * q1.y - q0.y * q1.w,
\\          q0.w * q1.x + q0.x * q1.w + q0.y * q1.z - q0.z * q1.y);
\\  }
\\  float qLength2(vec4 q) {
\\      return dot(q, q);
\\  }
\\
\\  vec2 intersectSphere(vec3 ro, vec3 rd, float rad) {
\\      float b = dot(ro, rd);
\\      float c = dot(ro, ro) - rad * rad;
\\      float h = b * b - c;
\\      if (h < 0.0) return vec2(-1.0);
\\      h = sqrt(h);
\\      return vec2(-b - h, -b + h);
\\  }
\\
\\  vec3 calcBounceDirection(vec3 nor) {
\\      float u = frand() * 2.0 - 1.0;
\\      float a = frand() * 6.28318531;
\\      return normalize(nor + vec3(sqrt(1.0 - u * u) * vec2(cos(a), sin(a)), u));
\\  }
\\
\\  mat3 setCamera(vec3 ro, vec3 ta, float cr) {
\\      vec3 cw = normalize(ta - ro);
\\      vec3 cp = vec3(sin(cr), cos(cr), 0.0);
\\      vec3 cu = normalize(cross(cw, cp));
\\      vec3 cv = normalize(cross(cu, cw));
\\      return mat3(cu, cv, cw);
\\  }
\\
\\  vec2 map(vec3 p) {
\\      #ifdef Z2 // z^2 + c
\\
\\      vec4 z = vec4(p, 0.0);
\\      vec4 zp = vec4(1.0, 0.0, 0.0, 0.0);
\\      float m2 = 0.0;
\\      float n = 0.0;
\\      for (int i = 0; i < k_num_iter; ++i) {
\\          zp = 2.0 * qMul(z, zp);
\\          z = qSquare(z) + k_c;
\\          m2 = qLength2(z);
\\          if (m2 > 256.0) {
\\              break;
\\          }
\\          n += 1.0;
\\      }
\\      float m = sqrt(m2);
\\      return vec2(0.5 * m * log(m) / length(zp), n);
\\
\\      #else // z^3 + c
\\
\\      vec4 z = vec4(p, 0.0);
\\      float dz2 = 1.0;
\\      float m2 = 0.0;
\\      float n = 0.0;
\\      for (int i = 0; i < k_num_iter; ++i) {
\\          dz2 *= 9.0 * qLength2(qSquare(z));
\\          z = qCube(z) + k_c;
\\          m2 = qLength2(z);
\\          if (m2 > 256.0) {
\\              break;
\\          }
\\          n += 1.0;
\\      }
\\      return vec2(0.25 * log(m2) * sqrt(m2 / dz2), n);
\\
\\      #endif // #ifdef Z2
\\  }
\\
\\  vec2 castRay(vec3 ro, vec3 rd) {
\\      float tmax = 7.0;
\\      float tmin = k_precis;
\\
\\      #if 1
\\      {
\\          float tp_f = (-0.8 - ro.y) / rd.y;
\\          if (tp_f > 0.0) tmax = min(tmax, tp_f);
\\      }
\\      #endif
\\
\\      #if 1
\\      {
\\          vec2 bv = intersectSphere(ro, rd, k_bounding_sphere_rad);
\\          if (bv.y < 0.0) return vec2(-2.0, 0.0);
\\          tmin = max(tmin, bv.x);
\\          tmax = min(tmax, bv.y);
\\      }
\\      #endif
\\
\\      vec2 res = vec2(-1.0);
\\      float t = tmin;
\\      float lt = 0.0;
\\      float lh = 0.0;
\\      for (int i = 0; i < 1024; ++i) {
\\          res = map(ro + rd * t);
\\          if (res.x < k_precis) {
\\              break;
\\          }
\\          lt = t;
\\          lh = res.x;
\\          t += min(res.x, 0.2);
\\          if (t > tmax) {
\\              break;
\\          }
\\      }
\\      if (lt > 0.0001 && res.x < 0.0) t = lt - lh * (t - lt) / (res.x - lh);
\\      res.x = (t < tmax) ? t : -1.0;
\\      return res;
\\  }
\\
\\  vec3 calcNormal(vec3 pos) {
\\      const vec2 e = vec2(1.0, -1.0) * 0.5773 * k_precis;
\\      return normalize(
\\          e.xyy * map(pos + e.xyy).x +
\\          e.yyx * map(pos + e.yyx).x +
\\          e.yxy * map(pos + e.yxy).x +
\\          e.xxx * map(pos + e.xxx).x);
\\  }
\\
\\  vec3 calcSurfaceColor(vec3 pos, vec3 nor, vec2 tn) {
\\      vec3 col = 0.5 + 0.5 * cos(log2(tn.y) * 0.9 + 3.5 + vec3(0.0, 0.6, 1.0));
\\      if (pos.y > 0.0) col = mix(col, vec3(1.0), 0.2);
\\      float inside = smoothstep(14.0, 15.0, tn.y);
\\      col *= vec3(0.45, 0.42, 0.40) + vec3(0.55, 0.58, 0.60) * inside;
\\      col = mix(col * col * (3.0 - 2.0 * col), col, inside);
\\      col = mix(mix(col, vec3(dot(col, vec3(0.3333))), -0.4), col, inside);
\\      return clamp(col * 0.65, 0.0, 1.0);
\\  }
\\
\\  vec3 render(vec3 ro, vec3 rd) {
\\      vec3 color_mask = vec3(1.0);
\\
\\      for (int bounce = 0; bounce < 3; ++bounce) {
\\          vec2 tn = castRay(ro, rd);
\\          float t = tn.x;
\\          if (t < 0.0) {
\\              return bounce > 0 ? color_mask * 1.65 * step(0.0, rd.y) : vec3(clamp(0.02 + 0.021 * rd.y, 0.0, 1.0));
\\          } else {
\\              vec3 pos = ro + rd * t;
\\              vec3 nor = calcNormal(pos);
\\              color_mask *= calcSurfaceColor(pos, nor, tn);
\\              rd = calcBounceDirection(nor);
\\              ro = pos + nor * k_precis;
\\          }
\\      }
\\      return vec3(0.0);
\\  }
\\
\\  void main() {
\\      ivec2 q = ivec2(gl_GlobalInvocationID);
\\      if (q.x >= int(u_resolution.x) || q.y > int(u_resolution.y)) {
\\          return;
\\      }
\\      srand(hash(q.x + hash(q.y + hash(1117 * u_frame))));
\\
\\      vec2 fragcoord = q + vec2(0.5);
\\
\\      float an = 0.5 + u_time * 0.03;
\\      vec3 ro = 2.0 * vec3(sin(an), 0.8, cos(an));
\\      vec3 ta = vec3(0.0, -0.1, 0.0);
\\      mat3x3 cam = setCamera(ro, ta, 0.0);
\\
\\      vec2 p = (2.0 * fragcoord - u_resolution) / u_resolution.y;
\\      vec3 rd = normalize(cam * vec3(p, k_foc_len));
\\
\\      vec3 col = render(ro, rd);
\\
\\      vec3 old_col = imageLoad(u_image, q).rgb;
\\      imageStore(u_image, q, mix(vec4(old_col, 1.0), vec4(col, 1.0), 0.06));
\\  }
;};
// zig fmt: on

fn createShaderProgram(stype: c.GLenum, glsl: [*]const u8) c.GLuint {
    const prog = c.glCreateShaderProgramv(stype, 1, &@as([*c]const u8, glsl));
    var status: c.GLint = 0;
    c.glGetProgramiv(prog, c.GL_LINK_STATUS, &status);
    if (status == c.GL_FALSE) {
        var log = [_]u8{0} ** 256;
        c.glGetProgramInfoLog(prog, log.len, null, &log);
        std.debug.panic("{s}\n", .{log});
    }
    return prog;
}

pub fn main() !void {
    _ = c.glfwSetErrorCallback(handleGlfwError);
    if (c.glfwInit() == c.GLFW_FALSE) {
        std.debug.panic("glfwInit() failed.\n", .{});
    }
    defer c.glfwTerminate();

    c.glfwWindowHint(c.GLFW_CONTEXT_VERSION_MAJOR, 4);
    c.glfwWindowHint(c.GLFW_CONTEXT_VERSION_MINOR, 6);
    c.glfwWindowHint(c.GLFW_OPENGL_PROFILE, c.GLFW_OPENGL_CORE_PROFILE);
    c.glfwWindowHint(c.GLFW_OPENGL_FORWARD_COMPAT, c.GL_TRUE);
    if (builtin.mode == .ReleaseFast) {
        c.glfwWindowHint(c.GLFW_OPENGL_DEBUG_CONTEXT, c.GL_FALSE);
    } else {
        c.glfwWindowHint(c.GLFW_OPENGL_DEBUG_CONTEXT, c.GL_TRUE);
    }
    c.glfwWindowHint(c.GLFW_DEPTH_BITS, 24);
    c.glfwWindowHint(c.GLFW_STENCIL_BITS, 8);
    c.glfwWindowHint(c.GLFW_RESIZABLE, c.GLFW_FALSE);

    const window = c.glfwCreateWindow(
        window_width,
        window_height,
        window_name,
        null,
        null,
    ) orelse {
        std.debug.panic("glfwCreateWindow() failed.\n", .{});
    };
    defer c.glfwDestroyWindow(window);

    c.glfwMakeContextCurrent(window);
    c.glfwSwapInterval(0);

    c.initOpenGlEntryPoints();

    c.glCreateProgramPipelines(1, &oglppo);
    c.glBindProgramPipeline(oglppo);

    if (builtin.mode != .ReleaseFast) {
        c.glEnable(c.GL_DEBUG_OUTPUT);
        c.glDebugMessageCallback(handleGlError, null);
    }

    var image_a: c.GLuint = 0;
    c.glCreateTextures(c.GL_TEXTURE_2D, 1, &image_a);
    c.glTextureStorage2D(
        image_a,
        1,
        c.GL_RGBA32F,
        window_width,
        window_height,
    );
    c.glClearTexImage(
        image_a,
        0,
        c.GL_RGBA,
        c.GL_FLOAT,
        &[_]f32{ 0.0, 0.0, 0.0, 0.0 },
    );
    defer c.glDeleteTextures(1, &image_a);

    const cs = createShaderProgram(c.GL_COMPUTE_SHADER, cs_image_a.src);
    defer c.glDeleteProgram(cs);

    var fbo: c.GLuint = 0;
    c.glCreateFramebuffers(1, &fbo);
    defer c.glDeleteFramebuffers(1, &fbo);

    var frame_num: i32 = 0;

    while (c.glfwWindowShouldClose(window) == c.GLFW_FALSE) {
        const stats = updateFrameStats(window, window_name);

        c.glProgramUniform1i(cs, cs_image_a.frame_loc, frame_num);
        c.glProgramUniform1f(cs, cs_image_a.time_loc, @floatCast(f32, stats.time));
        c.glProgramUniform2f(
            cs,
            cs_image_a.resolution_loc,
            @intToFloat(f32, window_width),
            @intToFloat(f32, window_height),
        );
        c.glBindImageTexture(
            cs_image_a.image_unit,
            image_a,
            0, // level
            c.GL_FALSE, // layered
            0, // layer
            c.GL_READ_WRITE,
            c.GL_RGBA32F,
        );
        c.glUseProgramStages(oglppo, c.GL_COMPUTE_SHADER_BIT, cs);
        c.glDispatchCompute(
            (window_width + cs_image_a.group_size_x - 1) / cs_image_a.group_size_x,
            (window_height + cs_image_a.group_size_y - 1) / cs_image_a.group_size_y,
            1,
        );
        c.glUseProgramStages(oglppo, c.GL_ALL_SHADER_BITS, 0);
        c.glMemoryBarrier(c.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        c.glNamedFramebufferTexture(fbo, c.GL_COLOR_ATTACHMENT0, image_a, 0);
        c.glBlitNamedFramebuffer(
            fbo, // src
            0, // dst
            0, // x0
            0, // y0
            window_width,
            window_height,
            0, // x1
            0, // y1
            window_width,
            window_height,
            c.GL_COLOR_BUFFER_BIT,
            c.GL_NEAREST,
        );
        c.glNamedFramebufferTexture(fbo, c.GL_COLOR_ATTACHMENT0, 0, 0);

        frame_num += 1;
        if (c.glGetError() != c.GL_NO_ERROR) {
            std.debug.panic("OpenGL error detected.\n", .{});
        }
        c.glfwSwapBuffers(window);
        c.glfwPollEvents();
    }
}

fn updateFrameStats(
    window: *c.GLFWwindow,
    name: [*:0]const u8,
) struct { time: f64, delta_time: f32 } {
    const state = struct {
        var timer: std.time.Timer = undefined;
        var previous_time_ns: u64 = 0;
        var header_refresh_time_ns: u64 = 0;
        var frame_count: u64 = ~@as(u64, 0);
    };

    if (state.frame_count == ~@as(u64, 0)) {
        state.timer = std.time.Timer.start() catch unreachable;
        state.previous_time_ns = 0;
        state.header_refresh_time_ns = 0;
        state.frame_count = 0;
    }

    const now_ns = state.timer.read();
    const time = @intToFloat(f64, now_ns) / std.time.ns_per_s;
    const delta_time = @intToFloat(f32, now_ns - state.previous_time_ns) /
        std.time.ns_per_s;
    state.previous_time_ns = now_ns;

    if ((now_ns - state.header_refresh_time_ns) >= std.time.ns_per_s) {
        const t = @intToFloat(f64, now_ns - state.header_refresh_time_ns) /
            std.time.ns_per_s;
        const fps = @intToFloat(f64, state.frame_count) / t;
        const ms = (1.0 / fps) * 1000.0;

        var buffer = [_]u8{0} ** 128;
        const buffer_slice = buffer[0 .. buffer.len - 1];
        const header = std.fmt.bufPrint(
            buffer_slice,
            "[{d:.1} fps  {d:.3} ms] {s}",
            .{ fps, ms, name },
        ) catch buffer_slice;

        _ = c.glfwSetWindowTitle(window, header.ptr);

        state.header_refresh_time_ns = now_ns;
        state.frame_count = 0;
    }
    state.frame_count += 1;

    return .{ .time = time, .delta_time = delta_time };
}

fn handleGlfwError(err: c_int, description: [*c]const u8) callconv(.C) void {
    std.debug.panic("GLFW error: {s}\n", .{@as([*:0]const u8, description)});
}

fn handleGlError(
    source: c.GLenum,
    mtype: c.GLenum,
    id: c.GLuint,
    severity: c.GLenum,
    length: c.GLsizei,
    message: [*c]const c.GLchar,
    user_param: ?*const c_void,
) callconv(.C) void {
    if (message != null) {
        std.debug.print("{s}\n", .{message});
    }
}
