const builtin = @import("builtin");
const std = @import("std");
const c = @import("c.zig");
const math = std.math;
const comptimePrint = std.fmt.comptimePrint;

const window_name = "quaternion julia sets";
const real_time = true;
const fps = 24;
const window_width = if (real_time) 1280 else 1920;
const window_height = if (real_time) 720 else 1080;
var oglppo: c.GLuint = 0;

// zig fmt: off
const cs_image_a = struct {
var name_z2: c.GLuint = 0;
var name_z3: c.GLuint = 0;
var name_z2_cut: c.GLuint = 0;
var name_z3_cut: c.GLuint = 0;
const group_size_x = 8;
const group_size_y = 8;
const frame_loc = 0;
const time_loc = 1;
const resolution_loc = 2;
const fractal_c_loc = 3;
const beat_loc = 4;
const image_unit = 0;
const src =
 "  layout(" ++
 "      local_size_x = " ++ comptimePrint("{d}", .{group_size_x}) ++ ", " ++
 "      local_size_y = " ++ comptimePrint("{d}", .{group_size_y}) ++ ") in;\n" ++
 "  layout(location = " ++ comptimePrint("{d}", .{frame_loc}) ++ ") uniform int u_frame;\n" ++
 "  layout(location = " ++ comptimePrint("{d}", .{time_loc}) ++ ") uniform float u_time;\n" ++
 "  layout(location = " ++ comptimePrint("{d}", .{beat_loc}) ++ ") uniform float u_beat;\n" ++
 "  layout(location = " ++ comptimePrint("{d}", .{resolution_loc}) ++ ") uniform vec2 u_resolution;\n" ++
 "  layout(location = " ++ comptimePrint("{d}", .{fractal_c_loc}) ++ ") uniform vec4 u_fractal_c;\n" ++
 "  layout(rgba32f, binding = " ++ comptimePrint("{d}", .{image_unit}) ++ ") uniform image2D u_image;\n" ++
\\
\\  //#define Z3
\\  //#define CUT
\\
\\  const float k_foc_len = 3.2;
\\  const float k_bounding_sphere_rad = 2.2;
\\  const int k_num_iter = 200;
++
    blk: {
        break :blk "\n" ++
        if (real_time)
 "  const int k_num_bounces = 2;\n  const float k_precis = 0.00025;\n\n"
        else
 "  const int k_num_bounces = 4;\n  const float k_precis = 0.00025;\n\n";
    }
++
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
\\      vec4 z = vec4(p, 0.0);
\\      float m2 = 0.0;
\\      float n = 0.0;
\\      float trap_dist = 1e10;
\\
\\      #ifdef Z2 // z^2 + c
\\
\\      vec4 zp = vec4(1.0, 0.0, 0.0, 0.0);
\\      for (int i = 0; i < k_num_iter; ++i) {
\\          zp = 2.0 * qMul(z, zp);
\\          z = qSquare(z) + u_fractal_c;
\\          m2 = qLength2(z);
\\          #ifdef CUT
\\          trap_dist = min(trap_dist, length(z.z - 0.25) - 0.01);
\\          #else
\\          trap_dist = min(trap_dist, length(z.xz - vec2(0.45, 0.55)) - 0.1);
\\          #endif
\\          if (m2 > 256.0) {
\\              break;
\\          }
\\          n += 1.0;
\\      }
\\      float m = sqrt(m2);
\\      float dist = 0.5 * m * log(m) / length(zp);
\\
\\      #endif // #ifdef Z2
\\
\\      #ifdef Z3 // z^3 + c
\\
\\      float dz2 = 1.0;
\\      for (int i = 0; i < k_num_iter; ++i) {
\\          dz2 *= 9.0 * qLength2(qSquare(z));
\\          z = qCube(z) + u_fractal_c;
\\          m2 = qLength2(z);
\\          trap_dist = min(trap_dist, length(z.xz - vec2(0.45, 0.55)) - 0.1);
\\          if (m2 > 256.0) {
\\              break;
\\          }
\\          n += 1.0;
\\      }
\\      float dist = 0.25 * log(m2) * sqrt(m2 / dz2);
\\
\\      #endif // #ifdef Z3
\\
\\      dist = min(dist, trap_dist);
\\      #ifdef CUT
\\      dist = max(dist, p.y);
\\      #endif
\\
\\      return vec2(dist, n);
\\  }
\\
\\  vec2 castRay(vec3 ro, vec3 rd) {
\\      float tmax = 7.0;
\\      float tmin = k_precis;
\\
\\      #ifdef CUT
\\      {
\\          const float k_split = 0.01;
\\          float tp_s = (k_split - ro.y) / rd.y;
\\          if (tp_s > 0.0) {
\\              if (ro.y > k_split) tmin = max(tmin, tp_s);
\\              else tmax = min(tmax, tp_s);
\\          }
\\      }
\\      #endif
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
\\          t += min(res.x, 0.01) * (0.5 + 0.5 * frand());
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
\\      vec3 col = 0.5 + 0.5 * cos(log(tn.y) * vec3(1.0) * 0.9);
\\      float inside = smoothstep(11.0, 12.0, tn.y);
\\      col *= vec3(0.45, 0.42, 0.40) + vec3(u_beat, 0.0, 0.60 + u_beat) * inside;
\\      col = mix(col * col * (3.0 - 2.0 * col), col, inside);
\\      col = mix(mix(col, vec3(dot(col, vec3(0.3333))), -0.4), col, inside);
\\      return clamp(col, 0.0, 1.0);
\\  }
\\
\\  vec3 render(vec3 ro, vec3 rd) {
\\      vec3 color_mask = vec3(1.0);
\\
\\      for (int bounce = 0; bounce < k_num_bounces; ++bounce) {
\\          vec2 tn = castRay(ro, rd);
\\          float t = tn.x;
\\          if (t < 0.0) {
\\              return bounce > 0 ? color_mask * 1.65 * step(0.0, rd.y) : vec3(clamp(0.2 + 0.21 * rd.y, 0.0, 1.0));
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
\\      if (q.x >= int(u_resolution.x) || q.y >= int(u_resolution.y)) {
\\          return;
\\      }
\\      srand(hash(q.x + hash(q.y + hash(1117 * u_frame))));
\\
\\      float an = 0.5 + u_time * 0.05;
\\      vec3 ro = 2.0 * vec3(sin(an), 0.8, cos(an));
\\
\\      #ifdef CUT
\\      vec3 ta = vec3(0.0, -0.3, 0.0);
\\      #else
\\      vec3 ta = vec3(0.0, -0.1, 0.0);
\\      #endif
\\      mat3x3 cam = setCamera(ro, ta, 0.0);
\\
\\      vec2 fragcoord = q + vec2(frand(), frand());
\\
\\      vec2 p = (2.0 * fragcoord - u_resolution) / u_resolution.y;
\\      vec3 rd = normalize(cam * vec3(p, k_foc_len));
\\
\\      vec3 col = render(ro, rd);
\\
\\      vec3 old_col = imageLoad(u_image, q).rgb;
++
blk: {
    break :blk "\n" ++
    if (real_time)
 \\      imageStore(u_image, q, mix(vec4(old_col, 1.0), vec4(col, 1.0), 0.25));
    else
\\  if (u_frame == 99) {
\\      imageStore(u_image, q, vec4(0.01 * (old_col + col), 1.0));
\\  } else if (u_frame == 0) {
\\      imageStore(u_image, q, vec4(col, 1.0));
\\  } else {
\\      imageStore(u_image, q, vec4(old_col + col, 1.0));
\\  }
       ;
}
++
 "  }\n"
;};
// zig fmt: on

// zig fmt: off
const vs_full_tri = struct {
var name: c.GLuint = 0;
const src =
\\  out gl_PerVertex {
\\      vec4 gl_Position;
\\  };
\\
\\  void main() {
\\      const vec2 positions[] = { vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0) };
\\      gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
\\  }
;};
// zig fmt: on

// zig fmt: off
const fs_image = struct {
var name: c.GLuint = 0;
const resolution_loc = 0;
const image_unit = 0;
const src =
 "  layout(location = " ++ comptimePrint("{d}", .{resolution_loc}) ++ ") uniform vec2 u_resolution;\n" ++
 "  layout(binding = " ++ comptimePrint("{d}", .{image_unit}) ++ ") uniform sampler2D u_image;\n" ++
\\
\\  layout(location = 0) out vec4 o_color;
\\
\\  void main() {
\\      vec2 p = vec2(gl_FragCoord) / u_resolution;
\\      vec3 col = textureLod(u_image, p, 0.0).rgb;
\\
\\      col = pow(col, vec3(0.4545));
\\      col = pow(col, vec3(0.85, 0.97, 1.0));
\\      col = 0.5 * col + 0.5 * col * col * (3.0 - 2.0 * col);
\\
\\      o_color = vec4(col, 1.0);
\\  }
;};
// zig fmt: on

fn createShaderProgram(stype: c.GLenum, glsl: [*c]const u8, defines: [*c]const u8) c.GLuint {
    const arg = [3][*c]const u8{ "#version 460 core\n", defines, glsl };
    const prog = c.glCreateShaderProgramv(stype, 3, &arg[0]);
    var status: c.GLint = 0;
    c.glGetProgramiv(prog, c.GL_LINK_STATUS, &status);
    if (status == c.GL_FALSE) {
        var log = [_]u8{0} ** 256;
        c.glGetProgramInfoLog(prog, log.len, null, &log);
        std.debug.panic("{s}\n", .{log});
    }
    return prog;
}

fn drawFractal(time: f32, frame_num: u32, fractal_c: [4]f32, image_a: c.GLuint, shader: c.GLuint, beat: f32) void {
    c.glProgramUniform1i(shader, cs_image_a.frame_loc, @intCast(i32, frame_num));
    c.glProgramUniform1f(shader, cs_image_a.time_loc, time);
    c.glProgramUniform1f(shader, cs_image_a.beat_loc, beat);
    c.glProgramUniform2f(
        shader,
        cs_image_a.resolution_loc,
        @intToFloat(f32, window_width),
        @intToFloat(f32, window_height),
    );
    c.glProgramUniform4fv(shader, cs_image_a.fractal_c_loc, 1, &fractal_c);
    c.glBindImageTexture(
        cs_image_a.image_unit,
        image_a,
        0, // level
        c.GL_FALSE, // layered
        0, // layer
        c.GL_READ_WRITE,
        c.GL_RGBA32F,
    );
    c.glUseProgramStages(oglppo, c.GL_COMPUTE_SHADER_BIT, shader);
    c.glDispatchCompute(
        (window_width + cs_image_a.group_size_x - 1) / cs_image_a.group_size_x,
        (window_height + cs_image_a.group_size_y - 1) / cs_image_a.group_size_y,
        1,
    );
    c.glUseProgramStages(oglppo, c.GL_ALL_SHADER_BITS, 0);
    c.glMemoryBarrier(c.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    c.glProgramUniform2f(
        fs_image.name,
        fs_image.resolution_loc,
        @intToFloat(f32, window_width),
        @intToFloat(f32, window_height),
    );
    c.glBindTextureUnit(fs_image.image_unit, image_a);
    c.glUseProgramStages(oglppo, c.GL_VERTEX_SHADER_BIT, vs_full_tri.name);
    c.glUseProgramStages(oglppo, c.GL_FRAGMENT_SHADER_BIT, fs_image.name);
    c.glDrawArrays(c.GL_TRIANGLES, 0, 3);
    c.glUseProgramStages(oglppo, c.GL_ALL_SHADER_BITS, 0);
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

    const file = try std.fs.cwd().openFile("data/out_beat.clip", .{ .read = true });
    defer file.close();
    const file_reader = file.reader();

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

    cs_image_a.name_z2 = createShaderProgram(c.GL_COMPUTE_SHADER, cs_image_a.src, "#define Z2\n");
    cs_image_a.name_z3 = createShaderProgram(c.GL_COMPUTE_SHADER, cs_image_a.src, "#define Z3\n");
    cs_image_a.name_z2_cut = createShaderProgram(c.GL_COMPUTE_SHADER, cs_image_a.src, "#define Z2\n#define CUT\n");
    cs_image_a.name_z3_cut = createShaderProgram(c.GL_COMPUTE_SHADER, cs_image_a.src, "#define Z3\n#define CUT\n");
    defer c.glDeleteProgram(cs_image_a.name_z2);
    defer c.glDeleteProgram(cs_image_a.name_z3);
    defer c.glDeleteProgram(cs_image_a.name_z2_cut);
    defer c.glDeleteProgram(cs_image_a.name_z3_cut);

    vs_full_tri.name = createShaderProgram(c.GL_VERTEX_SHADER, vs_full_tri.src, "");
    defer c.glDeleteProgram(vs_full_tri.name);

    fs_image.name = createShaderProgram(c.GL_FRAGMENT_SHADER, fs_image.src, "");
    defer c.glDeleteProgram(fs_image.name);

    var vao: c.GLuint = 0;
    c.glCreateVertexArrays(1, &vao);
    defer c.glDeleteVertexArrays(1, &vao);
    c.glBindVertexArray(vao);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        std.debug.assert(leaked == false);
    }

    var image_data = try std.ArrayList(u8).initCapacity(&gpa.allocator, window_width * window_height * 3);
    try image_data.resize(window_width * window_height * 3);
    defer image_data.deinit();

    c.stbi_write_png_compression_level = 10;
    c.stbi_flip_vertically_on_write(1);

    const Stage = struct {
        c: [4]f32,
        comp: u32,
        sign: f32,
        shader: c.GLuint,
        num_frames: u32,
    };
    const stages = [_]Stage{
        .{ // 0
            .c = .{ 0.01, 0.5, -1.0, 0.0 },
            .comp = 2,
            .sign = 0.033,
            .shader = cs_image_a.name_z3_cut,
            .num_frames = 15 * fps,
        },
        .{ // 1
            .c = .{ 0.1, -1.0, 0.0, 0.5 },
            .comp = 1,
            .sign = 2.0,
            .shader = cs_image_a.name_z2,
            .num_frames = 8 * fps,
        },
        .{ // 2
            .c = .{ 0.01, -1.0, 1.0, 0.0 },
            .comp = 2,
            .sign = -5.0,
            .shader = cs_image_a.name_z2,
            .num_frames = 15 * fps,
        },
        .{ // 3
            .c = .{ 0.01, 0.0, -1.0, 1.0 },
            .comp = 3,
            .sign = -8.0,
            .shader = cs_image_a.name_z2,
            .num_frames = 10 * fps,
        },
        .{ // 4
            .c = .{ 0.25, 0.5, -1.0, 0.1 },
            .comp = 1,
            .sign = -2.5,
            .shader = cs_image_a.name_z3,
            .num_frames = 20 * fps,
        },
        .{ // 5
            .c = .{ -0.035, 6.0 / 22.0, 15.0 / 22.0, -6.0 / 22.0 },
            .comp = 0,
            .sign = 0.05,
            .shader = cs_image_a.name_z2_cut,
            .num_frames = 15 * fps,
        },
        .{ // 6
            .c = .{ 0.0, 6.0 / 22.0, 15.0 / 22.0, -6.0 / 22.0 },
            .comp = 0,
            .sign = 0.5,
            .shader = cs_image_a.name_z3,
            .num_frames = 15 * fps,
        },
        .{ // 7
            .c = .{ 0.0, 6.0 / 22.0, 15.0 / 22.0, -6.0 / 22.0 },
            .comp = 0,
            .sign = 0.5,
            .shader = cs_image_a.name_z2,
            .num_frames = 15 * fps,
        },
        .{ // 8
            .c = .{ 0.0, 6.0 / 22.0, 15.0 / 22.0, -6.0 / 22.0 },
            .comp = 0,
            .sign = 0.5,
            .shader = cs_image_a.name_z2,
            .num_frames = 15 * fps,
        },
        .{ // 9
            .c = .{ 0.0, 6.0 / 22.0, 15.0 / 22.0, -6.0 / 22.0 },
            .comp = 0,
            .sign = 0.5,
            .shader = cs_image_a.name_z2,
            .num_frames = 60 * fps,
        },
        .{ // end
            .c = .{ 20.0, 20.0, 20.0, 20.0 },
            .comp = 0,
            .sign = 0.0,
            .shader = cs_image_a.name_z2,
            .num_frames = 5 * fps,
        },
    };

    var frame_num: u32 = 0;
    var stage: u32 = 0;
    {
        var i: u32 = 0;
        while (i < stage) : (i += 1) {
            frame_num += stages[i].num_frames;
        }
    }

    var fractal_c = [4]f32{ 0.0, 0.0, 0.0, 0.0 };
    var fractal_comp: u32 = 0;
    var fractal_sign: f32 = 1.0;
    var fractal_shader: c.GLuint = 0;

    while (c.glfwWindowShouldClose(window) == c.GLFW_FALSE) {
        const stats = updateFrameStats(window, window_name);
        const time = @intToFloat(f32, frame_num) * (1.0 / @intToFloat(f32, fps));

        var stages_num_frames: u32 = 0;
        var i: u32 = 0;
        while (i < stages.len) : (i += 1) {
            if (frame_num >= stages_num_frames and stage == i) {
                stage += 1;
                fractal_c = stages[i].c;
                fractal_comp = stages[i].comp;
                fractal_sign = stages[i].sign;
                fractal_shader = stages[i].shader;
            }
            stages_num_frames += stages[i].num_frames;
        }

        fractal_c[@intCast(usize, fractal_comp)] += fractal_sign * 0.001;

        const beat = blk: {
            var file_buffer = [_]u8{0} ** 32;
            var buf = (file_reader.readUntilDelimiterOrEof(file_buffer[0..], '\x20') catch unreachable).?;
            break :blk std.fmt.parseFloat(f32, buf) catch unreachable;
        };

        var fractal_c_final = fractal_c;
        if (stage == 2) {
            fractal_c_final[1] = fractal_c[1] + 0.1 * beat;
        } else if (stage == 3) {
            fractal_c[1] += 0.025 * beat;
            fractal_c_final = fractal_c;
        } else if (stage == 5) {
            fractal_c_final[2] = fractal_c[2] + 0.05 * beat;
        } else if (stage == 6) {
            fractal_c_final[3] = fractal_c[3] + 0.01 * beat;
        } else if (stage == 7) {
            fractal_c_final[2] = fractal_c[2] + 0.007 * beat;
            fractal_c[0] -= 0.007 * beat;
            fractal_c[1] -= 0.007 * beat;
            fractal_c[3] -= 0.007 * beat;
            for (fractal_c) |*fc| {
                if (fc.* > 1.0) fc.* = -1.0 else if (fc.* < -1.0) fc.* = 1.0;
            }
            fractal_c_final[0] = fractal_c[0];
            fractal_c_final[1] = fractal_c[1];
            fractal_c_final[3] = fractal_c[3];
        } else if (stage == 9) {
            fractal_c_final = [_]f32{ 0.5, 0.1 * beat, 0.2 * beat, 0.3 * beat };
        } else if (stage == 10) {
            fractal_c[0] += beat * 0.021;
            fractal_c[1] += beat * 0.022;
            fractal_c[2] += beat * 0.023;
            fractal_c[3] += beat * 0.024;
            for (fractal_c) |*fc| {
                if (fc.* > 1.0) fc.* = -1.0 else if (fc.* < -1.0) fc.* = 1.0;
            }
            fractal_c_final = fractal_c;
        }

        if (real_time == false) {
            var image_num: u32 = 0;
            while (true) {
                drawFractal(time, image_num, fractal_c_final, image_a, fractal_shader, beat);
                c.glFinish();
                image_num += 1;

                if (image_num % 100 == 0) {
                    var buffer = [_]u8{0} ** 128;
                    const buffer_slice = buffer[0..];
                    const image_name = std.fmt.bufPrint(
                        buffer_slice,
                        "frame_{:0>5}.png",
                        .{frame_num},
                    ) catch unreachable;

                    c.glReadPixels(
                        0,
                        0,
                        window_width,
                        window_height,
                        c.GL_RGB,
                        c.GL_UNSIGNED_BYTE,
                        image_data.items.ptr,
                    );
                    _ = c.stbi_write_png(
                        image_name.ptr,
                        window_width,
                        window_height,
                        3,
                        image_data.items.ptr,
                        window_width * 3,
                    );
                    std.debug.print("Frame {:0>5} saved.\n", .{frame_num});
                    frame_num += 1;
                    image_num = 0;
                    if (frame_num == 4531) {
                        c.glfwSetWindowShouldClose(window, c.GLFW_TRUE);
                    }
                    break;
                }
            }
        } else {
            drawFractal(time, frame_num, fractal_c_final, image_a, fractal_shader, beat);
            frame_num += 1;
        }

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
        const avg_fps = @intToFloat(f64, state.frame_count) / t;
        const ms = (1.0 / avg_fps) * 1000.0;

        var buffer = [_]u8{0} ** 128;
        const buffer_slice = buffer[0 .. buffer.len - 1];
        const header = std.fmt.bufPrint(
            buffer_slice,
            "[{d:.1} fps  {d:.3} ms | time: {d:.2}] {s}",
            .{ avg_fps, ms, time, name },
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
