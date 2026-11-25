"""Render three-point caches over G1 and check every frame against MuJoCo."""
import argparse,json,os,subprocess
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
import mujoco
import numpy as np
import torch
from PIL import Image,ImageDraw
from blinkmimic.g1.g1_constants import SPEC
from blinkmimic.motion import resample_motion
from blinkmimic.rotations import quat_to_mat
from blinkmimic.threepoint import BODIES,OFFSETS
from validate_bones import make_model

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--data-root",type=Path,required=True)
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines()]
    index={}
    for split in ("train","val","test"):
        index.update({r["motion_id"]:r for r in map(json.loads,(a.cache/(split+".jsonl")).read_text().splitlines())})
    model=make_model();data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,480,640)
    cam=mujoco.MjvCamera();cam.distance=3.;cam.azimuth=135;cam.elevation=-18
    adr=[model.jnt_qposadr[model.joint(n).id] for n in SPEC.joint_names]
    report=[];thumbnails=[]
    for row in rows:
        mid=row["motion_id"];name=Path(mid).name
        z=np.load(a.cache/index[mid]["cache"],allow_pickle=False)
        p,q,j=resample_motion(*SPEC.load_motion(str(a.data_root/row["g1_path"])))
        n=len(z["time_s"])
        if n!=row["pairing"]["common_frames50"]:raise ValueError("Frame count mismatch")
        np.testing.assert_allclose(z["time_s"],np.arange(n)*.02,atol=1e-5)
        wr=quat_to_mat(torch.from_numpy(z["quaternion_world"])).numpy()
        lr=quat_to_mat(torch.from_numpy(z["quaternion_local"])).numpy()
        err=0.
        ff=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
            "-s","640x480","-r","25","-i","-","-c:v","libx264","-threads","2","-pix_fmt","yuv420p",
            "-crf","23",str(a.out/(name+".mp4"))],stdin=subprocess.PIPE)
        snapshots=[];snap=set((np.linspace(.1,.9,4)*(n-1)/2).astype(int)*2)
        for f in range(n):
            data.qpos[:3]=p[f];data.qpos[3:7]=q[f];data.qpos[adr]=j[f]
            mujoco.mj_forward(model,data)
            ab=model.body("pelvis").id; ar=data.xmat[ab].reshape(3,3);ap_=data.xpos[ab]
            for k,bname in enumerate(BODIES):
                bid=model.body(bname).id;R=data.xmat[bid].reshape(3,3)
                w=data.xpos[bid]+R@np.array(OFFSETS[k])
                err=max(err,float(np.max(abs(w-z["position_world"][f,k]))),
                    float(np.max(abs(ar.T@(w-ap_)-z["position_local"][f,k]))),
                    float(np.max(abs(R-wr[f,k]))),float(np.max(abs(ar.T@R-lr[f,k]))))
            if f%2:continue
            cam.lookat[:]=[p[f,0],p[f,1],.8]
            renderer.update_scene(data,cam)
            scene=renderer.scene
            for k in range(3):
                point=z["position_world"][f,k]
                g=scene.geoms[scene.ngeom];scene.ngeom+=1
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.025]*3),point,np.eye(3).flatten(),np.array([1.,.85,0.,1.]))
                for axis in range(3):
                    g=scene.geoms[scene.ngeom];scene.ngeom+=1
                    color=np.r_[np.eye(3)[axis],1.]
                    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).flatten(),color)
                    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.008,point,point+.12*wr[f,k,:,axis])
            im=Image.fromarray(renderer.render())
            draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,48),fill=(12,18,24))
            draw.text((7,5),name[:85],fill="white")
            draw.text((7,24),f"REFERENCE ONLY | hand/hand/virtual head | RGB=XYZ | {f*.02:.2f}s",fill="white")
            ff.stdin.write(im.tobytes())
            if f in snap:snapshots.append(im.resize((320,240)))
        ff.stdin.close()
        if ff.wait()!=0:raise RuntimeError("ffmpeg failed")
        thumbnails.append(snapshots)
        report.append(dict(motion_id=mid,frames=n,max_position_or_matrix_error=err,passed=err<=1e-4))
        print(json.dumps(report[-1]),flush=True)
    renderer.close()
    for group in range(5):
        sheet=Image.new("RGB",(1280,960))
        for r in range(4):
            for c,im in enumerate(thumbnails[group*4+r]):sheet.paste(im,(320*c,240*r))
        sheet.save(a.out/f"sheet_{group}.jpg")
    summary=dict(clips=len(rows),passed=all(x["passed"] for x in report),details=report,
                 scope="Kinematic reference playback, no dynamics or VR hardware")
    (a.out/"report.json").write_text(json.dumps(summary,indent=2))
    if not summary["passed"]:raise SystemExit(1)
if __name__=="__main__":main()
