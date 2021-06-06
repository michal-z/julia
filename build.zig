const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    const exe = b.addExecutable("julia", "src/julia.zig");
    exe.setTarget(target);
    exe.setBuildMode(mode);

    const glfw_src = [_][]const u8{
        "context.c",
        "init.c",
        "input.c",
        "monitor.c",
        "vulkan.c",
        "window.c",
        "win32_init.c",
        "win32_joystick.c",
        "win32_monitor.c",
        "win32_time.c",
        "win32_thread.c",
        "win32_window.c",
        "wgl_context.c",
        "egl_context.c",
        "osmesa_context.c",
    };
    const glfw_flags = [_][]const u8{ "-std=c99", "-D_GLFW_WIN32=1" };
    const glfw_src_dir = "c/glfw-3.3.4/src/";
    inline for (glfw_src) |src| {
        exe.addCSourceFile(glfw_src_dir ++ src, &glfw_flags);
    }
    exe.addIncludeDir("c/glfw-3.3.4/include");

    exe.addIncludeDir("c");

    exe.linkSystemLibrary("user32");
    exe.linkSystemLibrary("gdi32");
    exe.linkSystemLibrary("shell32");
    exe.linkSystemLibrary("opengl32");
    exe.linkSystemLibrary("c");

    exe.install();

    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
