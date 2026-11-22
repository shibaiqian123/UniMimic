"""Validate BONES conversion and render kinematic reference playback (not policy rollouts)."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from blinkmimic.g1.g1_constants import SPEC, G1_JOINT_NAMES, load_bones_csv
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib


def make_model():
    root = ET.parse(SPEC.mjcf_path).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((Path(SPEC.mjcf_path).parent / compiler.get("meshdir", ".")).resolve()))
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="validation_floor", type="plane", size="0 0 .05",
                  rgba=".23 .28 .32 1", contype="0", conaffinity="0")
    ET.SubElement(world, "light", pos="0 0 6", dir="0 0 -1", directional="true")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    entries = json.loads(a.manifest.read_text())["clips"]
    kin = Kinematics(SPEC, "cpu")
    model = make_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.0, 135, -18
    addresses = [model.jnt_qposadr[model.joint(n).id] for n in G1_JOINT_NAMES]
    report, sheets = [], {}
    for entry in entries:
        path = a.manifest.parent / entry["local_file"]
        name = path.stem
        p, q, j, fps = load_bones_csv(path)
        lib = MotionLib([str(path)], kin, "cpu", SPEC.load_motion)
        n = lib.num_frames
        raw = np.genfromtxt(path, delimiter=",", names=True)
        source_rot = Rotation.from_euler("xyz", np.stack([raw["root_rotate"+x] for x in "XYZ"], -1), degrees=True)
        t = np.arange(n) * .02
        ts = np.arange(len(p)) / fps
        expected_p = np.stack([np.interp(t, ts, raw["root_translate"+x]/100) for x in "XYZ"], -1)
        expected_j = np.stack([np.interp(t, ts, np.deg2rad(raw[x+"_dof"])) for x in G1_JOINT_NAMES], -1)
        expected_R = Slerp(ts, source_rot)(np.clip(t, ts[0], ts[-1])).as_matrix()
        actual_p, actual_R, actual_j = lib.root_p.numpy(), lib.root_R.numpy(), lib.q.numpy()
        errors = dict(position_m=float(abs(actual_p-expected_p).max()),
                      rotation_matrix=float(abs(actual_R-expected_R).max()),
                      joint_rad=float(abs(actual_j-expected_j).max()))
        source_quat = source_rot.as_quat()[:, [3,0,1,2]]
        quat_error = float(np.minimum(np.linalg.norm(q-source_quat,axis=-1),np.linalg.norm(q+source_quat,axis=-1)).max())
        limits = np.maximum(np.maximum(kin.q_lo.numpy()-actual_j, actual_j-kin.q_hi.numpy()),0)
        yaw = np.unwrap(np.arctan2(actual_R[:,1,0],actual_R[:,0,0]))
        displacement = actual_p[-1]-actual_p[0]
        local_displacement = actual_R[0].T @ displacement
        duration_error = float(ts[-1]-t[-1])
        passed = (max(errors.values()) < 1e-4 and quat_error < 1e-10
                  and -1e-12 <= duration_error < .020001
                  and np.isfinite(lib.body_p.numpy()).all())
        row = dict(**entry,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   source_frames=len(p),source_fps=fps,frames_50hz=n,
                   source_seconds=float(ts[-1]),converted_seconds=float(t[-1]),duration_loss_s=duration_error,
                   metadata_frames_match=len(p)==entry["metadata_frames"],
                   parity_errors=errors,quaternion_error=quat_error,conversion_passed=bool(passed),
                   height_m=[float(actual_p[:,2].min()),float(actual_p[:,2].max())],
                   path_length_m=float(np.linalg.norm(np.diff(actual_p[:,:2],axis=0),axis=-1).sum()),
                   displacement_world_m=displacement.tolist(),displacement_initial_body_m=local_displacement.tolist(),
                   heading_change_deg=float(np.rad2deg(yaw[-1]-yaw[0])),
                   heading_span_deg=float(np.rad2deg(np.ptp(yaw))),
                   joint_range_deg=np.rad2deg(np.ptp(actual_j,axis=0)).tolist(),
                   max_joint_limit_excess_rad=float(limits.max()),
                   joints_outside_model_limits=[G1_JOINT_NAMES[k] for k in np.where(limits.max(axis=0)>1e-4)[0]])
        np.savez_compressed(a.out/(name+".npz"),fps=np.array(50.),root_pos=actual_p,
                            root_quat_wxyz=Rotation.from_matrix(actual_R).as_quat()[:,[3,0,1,2]],
                            joint_pos=actual_j,joint_names=G1_JOINT_NAMES,source=str(path))
        ff = subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
                               "-s","640x480","-r","25","-i","-","-c:v","libx264","-threads","2",
                               "-pix_fmt","yuv420p","-crf","22",str(a.out/(name+".mp4"))],stdin=subprocess.PIPE)
        snapshot_frames = set((np.linspace(.15,.85,4)*(n-1)/2).astype(int)*2)
        snapshots = []
        for f in range(0,n,2):
            data.qpos[:3] = actual_p[f]
            data.qpos[3:7] = Rotation.from_matrix(actual_R[f]).as_quat()[[3,0,1,2]]
            data.qpos[addresses] = actual_j[f]
            mujoco.mj_forward(model,data)
            cam.lookat[:] = [actual_p[f,0],actual_p[f,1],.75]
            renderer.update_scene(data,cam)
            im = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(im)
            draw.rectangle((0,0,640,58),fill=(12,18,24))
            draw.text((8,6),name[:85],fill="white")
            draw.text((8,23),f"REFERENCE ONLY | {f*.02:.2f}/{t[-1]:.2f}s | 50Hz motion / 25fps video",fill="white")
            draw.text((8,40),f"z={actual_p[f,2]:.3f}m  yaw delta={np.rad2deg(yaw[f]-yaw[0]):.1f}deg",fill="white")
            ff.stdin.write(im.tobytes())
            if f in snapshot_frames:
                snapshots.append(im.resize((480,360)))
        ff.stdin.close()
        if ff.wait() != 0:
            raise RuntimeError("ffmpeg failed: "+name)
        sheets.setdefault(entry["category"], []).append((name,snapshots))
        report.append(row)
        print(json.dumps(dict(clip=name,conversion_passed=bool(passed),duration=round(float(t[-1]),2),
                              limit_excess=round(float(limits.max()),4))),flush=True)
    renderer.close()
    for category, clips in sheets.items():
        sheet = Image.new("RGB",(1920,360*len(clips)),(30,30,30))
        for r,(_,images) in enumerate(clips):
            for c,im in enumerate(images):
                sheet.paste(im,(480*c,360*r))
        sheet.save(a.out/(category+"_contact_sheet.jpg"))
    summary = dict(scope="Kinematic reference conversion/playback; no physics stepping or trained policy",
                   clips=len(report),all_conversion_passed=all(r["conversion_passed"] for r in report),
                   source_fps=120,target_fps=50,video_fps=25,
                   joint_limit_warning_clips=sum(r["max_joint_limit_excess_rad"]>1e-4 for r in report),
                   clips_detail=report)
    (a.out/"report.json").write_text(json.dumps(summary,indent=2))
    concat = a.out/"concat.txt"
    concat.write_text("".join("file '"+r["local_file"][:-4]+".mp4'\n" for r in entries))
    subprocess.run(["ffmpeg","-y","-loglevel","error","-f","concat","-safe","0","-i",str(concat),
                    "-c","copy",str(a.out/"all15_reference.mp4")],check=True)
    print(json.dumps({k:v for k,v in summary.items() if k!="clips_detail"},indent=2))


if __name__ == "__main__":
    main()
