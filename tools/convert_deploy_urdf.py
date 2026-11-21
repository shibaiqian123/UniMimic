"""Convert a G1 URDF to a standalone 29-body-DoF deployment scene.
Finger joints are explicitly fixed at zero; their link inertias remain in the model.
Source URDF/assets are never modified. No tuned inertia or gravity compensation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as E
import mujoco
import numpy as np

def convert(src,out,dynamics_xml):
    src=src.resolve();out=out.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    r=E.parse(src).getroot()
    from blinkmimic.deploy.core import JOINTS
    joints={j.get("name"):j for j in r.findall("joint")}
    if any(n not in joints for n in JOINTS):raise ValueError("missing controlled joint")
    fixed=[]
    for j in joints.values():
        if j.get("type")!="fixed" and j.get("name") not in JOINTS:
            if "dex1_finger_joint" not in j.get("name"):raise ValueError("unexpected joint "+j.get("name"))
            fixed.append(j.get("name"));j.set("type","fixed")
            for tag in ("axis","limit","mimic","dynamics"):
                for child in j.findall(tag):j.remove(child)
    for mesh in r.findall(".//mesh"):
        name=mesh.get("filename")
        if name.startswith("package://"):raise ValueError("resolve package URI explicitly: "+name)
        path=(src.parent/name).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        mesh.set("filename",str(path))
    for node in r.findall("mujoco"):r.remove(node)
    ext=E.SubElement(r,"mujoco")
    E.SubElement(ext,"compiler",discardvisual="false",fusestatic="false",strippath="false",balanceinertia="false")
    E.SubElement(r,"link",name="world")
    j=E.SubElement(r,"joint",name="floating_base_joint",type="floating")
    E.SubElement(j,"parent",link="world");E.SubElement(j,"child",link="pelvis")
    with tempfile.TemporaryDirectory() as td:
        urdf=Path(td)/"model.urdf";compiled=Path(td)/"model.xml"
        E.ElementTree(r).write(urdf)
        model=mujoco.MjModel.from_xml_path(str(urdf))
        mujoco.mj_saveLastXML(str(compiled),model)
        mj=E.parse(compiled).getroot()
    world=mj.find("worldbody");pelvis=world.find("./body[@name='pelvis']")
    if pelvis is None:raise ValueError("pelvis body missing")
    # URDF floating joint compilation differs by MuJoCo release; require one free joint.
    if not pelvis.findall("freejoint") and not pelvis.findall("./joint[@type='free']"):
        E.SubElement(pelvis,"freejoint",name="floating_base_joint")
    E.SubElement(world,"geom",name="floor",type="plane",size="0 0 .05",rgba=".3 .35 .4 1",friction="1 .005 .0001")
    E.SubElement(world,"light",pos="0 0 3",dir="0 0 -1",directional="true")
    E.SubElement(pelvis,"site",name="imu",pos="0 0 0",size=".005")
    sensors=E.SubElement(mj,"sensor")
    E.SubElement(sensors,"framequat",name="imu_quat",objtype="site",objname="imu")
    E.SubElement(sensors,"gyro",name="imu_gyro",site="imu")
    E.SubElement(sensors,"accelerometer",name="imu_acc",site="imu")
    actuator=E.SubElement(mj,"actuator")
    for name in JOINTS:
        effort=float(joints[name].find("limit").get("effort"))
        E.SubElement(actuator,"motor",name=name+"_motor",joint=name,gear="1",ctrllimited="true",ctrlrange=f"{-effort} {effort}")
    option=mj.find("option")
    if option is None:option=E.SubElement(mj,"option")
    option.set("timestep","0.001")
    dynamics=mujoco.MjModel.from_xml_path(str(dynamics_xml.resolve()))
    dynamics_values={}
    for name in JOINTS:
        dof=int(dynamics.jnt_dofadr[dynamics.joint(name).id])
        values={key:float(getattr(dynamics,"dof_"+key)[dof]) for key in ("armature","damping","frictionloss")}
        dynamics_values[name]=values
        joint=mj.find(".//joint[@name='"+name+"']")
        for key,value in values.items():joint.set(key,str(value))
    E.indent(mj)
    E.ElementTree(mj).write(out,encoding="unicode")
    model=mujoco.MjModel.from_xml_path(str(out))
    if (model.nq,model.nv,model.nu)!=(36,35,29):raise ValueError("unexpected dimensions")
    mass=sum(float(l.find("inertial/mass").get("value")) for l in r.findall("link") if l.find("inertial/mass") is not None)
    if not np.isclose(mass,model.body_mass.sum(),atol=1e-5):raise ValueError("mass not preserved")
    report=dict(source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        scene=str(out),scene_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        mass_kg=float(model.body_mass.sum()),urdf_mass_kg=mass,
        dynamics_source=str(dynamics_xml.resolve()),dynamics_sha256=hashlib.sha256(dynamics_xml.read_bytes()).hexdigest(),motor_dynamics=dynamics_values,
        fixed_at_zero=fixed,nq=model.nq,nv=model.nv,nu=model.nu,
        imu="pelvis origin, aligned axes; reference sensor placement, not hardware calibrated",
        scope="both Dex1-1 hands; fingers fixed at zero; source URDF inertias/collisions; explicit reference motor dynamics")
    out.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--dynamics-xml",type=Path,required=True,help="reference motor armature/damping/frictionloss by joint name")
    a=ap.parse_args();convert(a.urdf,a.out,a.dynamics_xml)
