"""Per-process GLX selection for interactive viewers; never changes desktop configuration."""
import os
import shutil
import subprocess

def probe(env):
    result=subprocess.run(["glxinfo","-B"],env=env,capture_output=True,text=True,timeout=10)
    renderer=next((l.split(":",1)[1].strip() for l in result.stdout.splitlines()
                   if l.startswith("OpenGL renderer string:")),"")
    if result.returncode or not renderer:
        raise RuntimeError("OpenGL probe failed: "+(result.stderr.strip() or result.stdout.strip()))
    return renderer

def configure_opengl(mode="auto"):
    if mode not in ("auto","system","nvidia"):raise ValueError("invalid GL renderer selection")
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("Viewer needs a graphical desktop DISPLAY; run on the desktop or configure X forwarding")
    if not shutil.which("glxinfo"):
        if mode=="nvidia":
            os.environ["__NV_PRIME_RENDER_OFFLOAD"]="1"
            os.environ["__GLX_VENDOR_LIBRARY_NAME"]="nvidia"
            os.environ.pop("LIBGL_ALWAYS_SOFTWARE",None)
        print("OpenGL: glxinfo unavailable; renderer not verified",flush=True)
        return {"renderer":"UNVERIFIED","mode":mode}
    env=dict(os.environ)
    original=probe(env)
    software=any(s in original.lower() for s in ("llvmpipe","softpipe","software rasterizer","swrast"))
    if mode=="nvidia" or (mode=="auto" and software):
        env["__NV_PRIME_RENDER_OFFLOAD"]="1"
        env["__GLX_VENDOR_LIBRARY_NAME"]="nvidia"
        env.pop("LIBGL_ALWAYS_SOFTWARE",None)
        try:
            selected=probe(env)
            if "nvidia" not in selected.lower():raise RuntimeError("NVIDIA renderer not selected: "+selected)
        except (RuntimeError,subprocess.TimeoutExpired) as exc:
            if mode=="nvidia":raise
            print("OpenGL WARNING: software renderer remains active; NVIDIA offload unavailable: "+str(exc),flush=True)
            return {"renderer":original,"software":True,"mode":mode}
        os.environ["__NV_PRIME_RENDER_OFFLOAD"]="1"
        os.environ["__GLX_VENDOR_LIBRARY_NAME"]="nvidia"
        os.environ.pop("LIBGL_ALWAYS_SOFTWARE",None)
        print(f"OpenGL: {original} -> {selected}",flush=True)
        return {"renderer":selected,"previous_renderer":original,"software":False,"mode":mode}
    print("OpenGL: "+original+(" (SOFTWARE RENDERING)" if software else ""),flush=True)
    return {"renderer":original,"software":software,"mode":mode}
