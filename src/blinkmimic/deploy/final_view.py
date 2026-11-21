"""Display-only final simulation window. Keys travel through LowState; no direct controller calls."""
import argparse,json,socket,time
from pathlib import Path
import numpy as np
import mujoco
from .graphics import configure_opengl
from .core import Bundle
from .sim_ipc import SimIPC
from .final_session import read
from .viewer_utils import RenderMirror
from .buttons import WindowInput

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--session",required=True,type=Path)
    ap.add_argument("--gl-renderer",choices=["auto","system","nvidia"],default="auto")
    ap.add_argument("--render-hz",type=float,default=30.)
    ap.add_argument("--frames",type=int,help="test only: close after N rendered frames")
    ap.add_argument("--hidden",action="store_true",help="test only: create an invisible GLFW window")
    a=ap.parse_args();meta=read(a.session/"session.json")
    if meta.get("kind")!="blink_final_sim" or meta.get("domain") not in (1,2):raise ValueError("simulation session required")
    configure_opengl(a.gl_renderer)
    import glfw
    if not glfw.init():raise RuntimeError("GLFW initialization failed")
    if a.hidden:glfw.window_hint(glfw.VISIBLE,glfw.FALSE)
    win=glfw.create_window(1280,800,"BlinkMimic FINAL SIM — shared real runner",None,None)
    if not win:raise RuntimeError("could not create GLFW window")
    glfw.make_context_current(win);glfw.swap_interval(1)
    b=Bundle(meta["bundle"]);ipc=SimIPC(a.session/"sim.mmap")
    model=mujoco.MjModel.from_xml_path(meta["unitree_xml"]);mirror=RenderMirror(model)
    m,d=mirror.m,mirror.d;ref=mujoco.MjData(m)
    free=int(np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)[0]);fq=int(m.jnt_qposadr[free])
    qidx=np.array([m.jnt_qposadr[m.joint(n).id] for n in b.cfg["joint_names"]])
    cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.distance=3.;cam.azimuth=135.;cam.elevation=-15.
    opt=mujoco.MjvOption();mujoco.mjv_defaultOption(opt)
    scene=mujoco.MjvScene(m,maxgeom=5000);context=mujoco.MjrContext(m,mujoco.mjtFontScale.mjFONTSCALE_150)
    flags=dict(ghost=True,follow=True,last_key="",mouse=None)
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM)
    def send(event):
        try:sock.sendto(json.dumps(event).encode(),str(a.session/"buttons.sock"))
        except OSError:flags["last_key"]="simulation input unavailable"
    window_input=WindowInput(send)
    def key_cb(window,key,scan,action,mods):
        if action!=glfw.PRESS:return # ignore OS repeat events
        if window_input.key(key,action):
            flags["last_key"]=chr(key)
        elif key==glfw.KEY_G:flags["ghost"]=not flags["ghost"]
        elif key==glfw.KEY_F:flags["follow"]=not flags["follow"]
        # Intentionally no pause/step/reset commands in final mode.
    def focus_cb(window,focused):
        if not focused:window_input.focus(False);flags["last_key"]="released (focus lost)"
    def mouse_cb(window,button,action,mods):
        flags["mouse"]=glfw.get_cursor_pos(window) if action==glfw.PRESS else None
    def cursor_cb(window,x,y):
        if flags["mouse"] is None:return
        oldx,oldy=flags["mouse"];flags["mouse"]=(x,y)
        height=max(glfw.get_window_size(window)[1],1)
        shift=glfw.get_key(window,glfw.KEY_LEFT_SHIFT)==glfw.PRESS
        if glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS:
            action=mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_RIGHT)==glfw.PRESS:
            action=mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:action=mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(m,action,(x-oldx)/height,(y-oldy)/height,cam)
    def scroll_cb(window,x,y):mujoco.mjv_moveCamera(m,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*y,cam)
    glfw.set_key_callback(win,key_cb);glfw.set_window_focus_callback(win,focus_cb)
    glfw.set_mouse_button_callback(win,mouse_cb);glfw.set_cursor_pos_callback(win,cursor_cb);glfw.set_scroll_callback(win,scroll_cb)
    frames=0;deadline=time.monotonic()
    try:
        while not glfw.window_should_close(win):
            glfw.poll_events()
            s=ipc.state();rs=read(a.session/"runner_status.json");ps=read(a.session/"physics_status.json")
            if s["tick"]==0:time.sleep(.01);continue
            d.qpos[:]=s["qpos"];d.qvel[:]=s["qvel"];d.time=s["sim_time"]
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_camlight(m,d)
            if flags["follow"]:cam.lookat[:]=d.qpos[fq:fq+3]
            mujoco.mjv_updateScene(m,d,opt,None,cam,mujoco.mjtCatBit.mjCAT_ALL,scene)
            frame=min(int(rs.get("frame",0)),b.T-1)
            if flags["ghost"]:
                origin=np.array(read(a.session/"plant.json").get("initial_root",b.motion["root_p"][0]))
                heading=np.array(rs.get("heading") or np.eye(3))
                ref.qpos[:]=d.qpos
                ref.qpos[fq:fq+3]=origin+heading@(b.motion["root_p"][frame]-b.motion["root_p"][0])
                mujoco.mju_mat2Quat(ref.qpos[fq+3:fq+7],(heading@b.motion["root_R"][frame]).reshape(9))
                ref.qpos[qidx]=b.motion["q"][frame];mujoco.mj_kinematics(m,ref)
                for i in range(1,m.nbody):
                    parent=m.body_parentid[i]
                    if parent==0 or scene.ngeom>=scene.maxgeom:continue
                    x,y=ref.xpos[parent],ref.xpos[i]
                    if np.linalg.norm(y-x)<1e-6:continue
                    g=scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,[.012,0,0],[0,0,0],np.eye(3).reshape(9),[.1,.7,1,.7])
                    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.012,x,y);scene.ngeom+=1
            w,h=glfw.get_framebuffer_size(win);viewport=mujoco.MjrRect(0,0,w,h)
            mujoco.mjr_render(viewport,scene,context)
            text=(f"FINAL SIM / CPU official G1 / real runner via DDS domain {meta['domain']}\n"
                  f"Policy inference: {meta.get('inference_device','cpu')}\n"
                  f"State: {rs.get('state','WAITING')} | frame {frame}/{b.T-1} | sim {s['sim_time']:.2f}s\n"
                  f"Support: {bool(s['support'])} | input: {flags['last_key']} | remote bits: {s['keys']}\n"
                  f"Physics-wall lag: {ps.get('lag_s',0)*1000:.1f}ms | sensor model: SuperMimic reference (uncalibrated)\n"
                  f"{rs.get('reason') or rs.get('reject_reason','')}\n"
                  "1 prepare | 2 standby | 3 play | D stop | B support\n"
                  "G reference | F camera | no pause/step in FINAL mode")
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,viewport,text,"",context)
            glfw.swap_buffers(win);frames+=1
            if a.frames and frames>=a.frames:break
            deadline+=1/a.render_hz
            time.sleep(max(0,deadline-time.monotonic()))
            if time.monotonic()-deadline>.1:deadline=time.monotonic()
    finally:
        send(dict(type="release"));sock.close();ipc.close()
        context.free();glfw.destroy_window(win);glfw.terminate()
if __name__=="__main__":main()
