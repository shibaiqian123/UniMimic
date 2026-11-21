"""Independent wall-clock CPU plant and modeled sensors. No policy/goal evaluation here."""
import gc
import argparse,copy,json,os,signal,socket,time
from pathlib import Path
import numpy as np
import mujoco
from .core import Bundle
from .sim2sim import Plant
from .imu_model import ImuModel
from .buttons import ButtonPulses
from .sim_ipc import SimIPC,STATE
from .session import atomic_json
from .runtime_io import RuntimeLog,cpu_list

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--session",required=True,type=Path)
    a=ap.parse_args();meta=json.loads((a.session/"session.json").read_text())
    if meta["kind"]!="blink_final_sim" or meta["domain"] not in (1,2):raise ValueError("simulation session required")
    bundle=Bundle(meta["bundle"])
    # Only model/actuator parameters are used. Initial pose is q0, not a motion frame.
    cfg=copy.copy(bundle)
    cfg.motion=dict(q=np.repeat(bundle.q0[None],2,axis=0),root_p=np.array([[0,0,.9],[0,0,.9]]),
                    root_R=np.repeat(np.eye(3)[None],2,axis=0))
    p=Plant(cfg,meta["unitree_xml"],"g2")
    if p.m.nq!=36 or p.m.nv!=35:raise ValueError("full 29DoF G1 required")
    sensor={}
    for name,dim in (("imu_quat",4),("imu_gyro",3),("imu_acc",3)):
        sid=p.m.sensor(name).id
        if p.m.sensor_dim[sid]!=dim:raise ValueError("sensor dimension mismatch")
        sensor[name]=slice(p.m.sensor_adr[sid],p.m.sensor_adr[sid]+dim)
    site=p.m.site("imu").id
    if p.m.site_bodyid[site]!=p.pelvis:raise ValueError("primary IMU must be rigidly on pelvis")
    mujoco.mj_forward(p.m,p.d)
    imu=ImuModel(p.d.sensordata[sensor["imu_quat"]].copy())
    ipc=SimIPC(a.session/"sim.mmap")
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM)
    sock.bind(str(a.session/"buttons.sock"));sock.setblocking(False)
    support_anchor=p.d.qpos[p.fq:p.fq+3].copy()
    from .core import rotation
    support_rotation=rotation(p.d.qpos[p.fq+3:p.fq+7])
    adapter=ButtonPulses();support=True;stop=[False];temperature=30.;drop=False;motor_fault=0
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    atomic_json(a.session/"plant.json",dict(initial_root=p.d.qpos[p.fq:p.fq+3].tolist(),
        initial_quaternion=p.d.qpos[p.fq+3:p.fq+7].tolist(),model_fingerprint=p.hash,
        primary_imu_site="imu",sensor_model="SuperMimic saturated gyro / integrated attitude",
        calibrated=False,gyro_limit_rad_s=imu.limit,physics_dt=.001))
    rec=[];state=np.zeros(1,STATE);row=state[0]
    ui=RuntimeLog(a.session/"physics_events.jsonl",cpu_list(os.environ.get("BLINK_LOG_CPUS")))
    gc_was_enabled=gc.isenabled();gc.collect();gc.disable()
    wall0=time.monotonic();tick=0;lags=[];display_lags=[];input_log=[];last_ui=0.
    cmd=ipc.command();last_read=0.;last_toggle=-1.
    try:
        while not stop[0]:
            now=time.monotonic()
            while True:
                try:packet=json.loads(sock.recv(2048))
                except BlockingIOError:break
                except (ValueError,UnicodeDecodeError):continue
                typ=packet.get("type")
                if typ=="key":
                    key=str(packet.get("key","")).upper()
                    if key=="B" and now-last_toggle>.25:
                        support=not support;last_toggle=now;accepted=True
                        if support:
                            support_anchor=p.d.qpos[p.fq:p.fq+3].copy()
                            support_rotation=rotation(p.d.qpos[p.fq+3:p.fq+7])

                    else:accepted=adapter.send(key,now)
                    input_log.append(dict(t=now,key=key,accepted=accepted))
                elif typ=="release":adapter.release(now)
                elif typ=="fault":
                    # Local simulation fault injection only; no equivalent hardware command.
                    temperature=float(packet.get("temperature",temperature))
                    drop=bool(packet.get("drop_state",drop));motor_fault=int(packet.get("motor_fault",motor_fault))
                elif typ=="quit":stop[0]=True
            # Robot drivers retain the last received command between 500 Hz messages.
            if now-last_read>=.001:
                cmd=ipc.command();last_read=now
            q,dq,gyro,quat=p.state()
            if cmd["seq"] and cmd["mode_machine"]==5 and cmd["mode_pr"]==0:
                raw=cmd["kp"]*(cmd["q"]-q)+cmd["kd"]*(cmd["dq"]-dq)+cmd["tau"]
                raw=np.where(cmd["mode"]!=0,raw,0.)
            else:
                raw=-3.*dq  # same passive damping as the shared runner, not a hidden stand controller
            tau=np.clip(raw,-bundle.effort,bundle.effort)
            limits=p.m.actuator_ctrlrange[p.act];limited=p.m.actuator_ctrllimited[p.act].astype(bool)
            tau[limited]=np.clip(tau[limited],limits[limited,0],limits[limited,1])
            p.d.ctrl[p.act]=tau;p.d.xfrc_applied[:]=0
            if support:
                R=rotation(quat);E=support_rotation@R.T
                err=np.array([E[2,1]-E[1,2],E[0,2]-E[2,0],E[1,0]-E[0,1]])*.5
                anchor=support_anchor
                p.d.xfrc_applied[p.pelvis,:3]=2000*(anchor-p.d.qpos[p.fq:p.fq+3])-100*p.d.qvel[p.fv:p.fv+3]+[0,0,p.m.body_mass.sum()*9.81]
                p.d.xfrc_applied[p.pelvis,3:]=500*err-50*(R@gyro)
            # Model estimator evolves from the primary-site gyro, not from refreshed true attitude.
            imu.integrate(p.d.sensordata[sensor["imu_gyro"]].copy(),.001)
            mujoco.mj_step(p.m,p.d)
            mujoco.mj_forward(p.m,p.d)  # sample joint/IMU channels at one consistent post-step time
            tick+=1;wall=time.monotonic();lag=(wall-wall0)-float(p.d.time)
            lags.append(lag)
            truth_quat=p.d.sensordata[sensor["imu_quat"]].copy()
            measured_gyro,estimated_quat=imu.observe(p.d.sensordata[sensor["imu_gyro"]],truth_quat,tick)
            row["sample_wall"]=0. if drop else wall;row["sim_time"]=p.d.time;row["tick"]=tick
            row["keys"]=adapter.sample(wall)
            row["q"]=p.d.qpos[p.qidx];row["dq"]=p.d.qvel[p.vidx];row["tau"]=p.d.actuator_force[p.act]
            row["gyro"]=measured_gyro;row["quat"]=estimated_quat;row["acc"]=p.d.sensordata[sensor["imu_acc"]]
            row["temp"]=temperature;row["motor_fault"]=motor_fault
            row["qpos"]=p.d.qpos;row["qvel"]=p.d.qvel;row["support"]=int(support)
            ipc.write(state)
            if tick%20==0:
                rec.append(dict(t=wall,sim_time=float(p.d.time),qpos=p.d.qpos.copy(),qvel=p.d.qvel.copy(),
                    imu_quat=estimated_quat.copy(),imu_gyro=measured_gyro.copy(),imu_truth=truth_quat.copy(),
                    support=support,keys=int(row["keys"]),command_seq=int(cmd["seq"])))
                display_lags.append(lag)
            if wall-last_ui>.1:
                ui.status(a.session/"physics_status.json",dict(t=wall,sim_time=float(p.d.time),lag_s=lag,
                    support=support,keys=int(row["keys"]),key_label=adapter.label,temperature=temperature,drop_state=drop,
                    command_seq=int(cmd["seq"])))
                last_ui=wall
            deadline=wall0+tick*.001
            remaining=deadline-time.monotonic()
            if remaining>0:time.sleep(remaining)
    finally:
        sock.close();ipc.close()
        ui.close()
        if gc_was_enabled:gc.enable()
        if rec:np.savez_compressed(a.session/"physics.npz",**{k:np.array([x[k] for x in rec]) for k in rec[0]})
        (a.session/"input.json").write_text(json.dumps(input_log,indent=2))
        lag99=float(np.percentile(np.abs(lags),99)) if lags else None
        atomic_json(a.session/"physics_report.json",dict(steps=tick,sim_seconds=float(p.d.time),
            wall_seconds=time.monotonic()-wall0,lag_p99_s=lag99,lag_max_s=max(map(abs,lags)) if lags else None,end_lag_s=lags[-1] if lags else None,
            timing_valid=bool(lags and lag99<=.02 and max(map(abs,lags))<=.02 and abs(lags[-1])<=.02),imu=imu.summary(),
            sensor_calibrated=False,temperature_model="synthetic input, not a calibrated thermal model"))
if __name__=="__main__":main()
