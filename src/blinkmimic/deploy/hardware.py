"""Explicit hardware launch helpers; no DDS initialization on import."""
import subprocess
import numpy as np

def gain_scales(spec):
    values=np.ones(29)
    seen=set()
    if not spec:return values
    for group in spec.split(";"):
        ids,scale=group.split("=")
        scale=float(scale)
        if not np.isfinite(scale) or scale<=0:raise ValueError("gain scale must be finite and positive")
        for item in ids.split(","):
            i=int(item)
            if not 0<=i<29 or i in seen:raise ValueError("invalid or repeated motor index")
            seen.add(i);values[i]=scale
    return values

def apply_gains(bundle,kp_spec,kd_spec):
    kp=gain_scales(kp_spec);kd=gain_scales(kd_spec)
    bundle.kp=bundle.kp*kp;bundle.kd=bundle.kd*kd
    return dict(kp_scale=kp.tolist(),kd_scale=kd.tolist(),
                kp=bundle.kp.tolist(),kd=bundle.kd.tolist())

def release_motion(bridge_dir,iface,domain,sim=False,read_only=False):
    if sim or read_only:return {"released":False,"reason":"simulation or read-only"}
    if domain!=0 or not iface or iface=="lo":raise ValueError("explicit robot NIC and domain 0 required")
    try:
        result=subprocess.run([str(bridge_dir/"g1_release_mode"),iface],check=True,
                              timeout=35,capture_output=True,text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Motion service release failed: "+(exc.stdout or "")+(exc.stderr or "")) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Motion service release timed out after 35 seconds; no motor publisher started") from exc
    return {"released":True,"output":result.stdout}
