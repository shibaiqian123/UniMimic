"""Validate actual paired samples against independent SciPy/NumPy formulas.

This checks features and interfaces, not learned tracking or live VR calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from blinkmimic.g1.g1_constants import SPEC
from blinkmimic.omni_contract import CONTRACT,MODALITIES
from blinkmimic.omni_data import ReferenceStore
from blinkmimic.motion import resample_motion
from blinkmimic.rotations import quat_to_mat


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    store=ReferenceStore(a.index,SPEC.joint_names)
    records=[json.loads(l) for l in a.manifest.read_text().splitlines() if l.strip()]
    results=[];errors={k:0. for k in ['smpl_root_R','smpl_local','motion','smpl','teleop']}
    C=Rotation.from_rotvec([np.pi/2,0,0]).as_matrix()
    base=Rotation.from_quat([.5,.5,.5,.5]).as_matrix()
    for r in records:
        cid=store.by_id[r['motion_id']];n=int(store.lengths[cid])
        source=a.data_root/r['smpl_path']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==r['pairing']['smpl_sha256']
        data=joblib.load(source)
        aa=np.asarray(data['pose_aa'])[:n];joints=np.asarray(data['smpl_joints'])[:n]
        assert aa.shape==(n,72) and joints.shape==(n,24,3) and data['fps']==50
        assert np.isfinite(aa).all() and np.isfinite(joints).all()
        reference_R=C@Rotation.from_rotvec(aa[:,:3]).as_matrix()@base.T
        local=np.einsum('tji,tkj->tki',reference_R,joints)
        x=store.gather(np.full(n,cid),np.arange(n))
        errors['smpl_root_R']=max(errors['smpl_root_R'],float(np.abs(reference_R-x[:,139:148].reshape(n,3,3)).max()))
        errors['smpl_local']=max(errors['smpl_local'],float(np.abs(local-x[:,67:139].reshape(n,24,3)).max()))
        _,quat,q=resample_motion(*SPEC.load_motion(str(a.data_root/r['g1_path'])))
        gR=quat_to_mat(quat[:n]).numpy()
        np.testing.assert_allclose(q[:n].numpy(),x[:,:29],atol=1e-5)
        # Every frame, including the common endpoint; nonzero roll/pitch/yaw.
        robot=gR@Rotation.from_euler('xyz',[.21,-.13,.72]).as_matrix()
        ids=np.arange(n);fm=np.minimum(ids[:,None]+np.arange(10)*5,n-1)
        fs=np.minimum(ids[:,None]+np.arange(10),n-1)
        relative=robot[:,None].transpose(0,1,3,2)@gR[fm]
        motion=np.concatenate((np.concatenate((x[fm,:29].reshape(n,-1),x[fm,29:58].reshape(n,-1)),-1).reshape(n,10,58),relative[...,:2].reshape(n,10,6)),-1).reshape(n,-1)
        smpl=np.concatenate((local[fs].reshape(n,10,72),(robot[:,None].transpose(0,1,3,2)@reference_R[fs])[...,:2].reshape(n,10,6),x[fs,:29][...,store.wrists]),-1).reshape(n,-1)
        teleop=np.concatenate((x[fm,:29][...,store.lower].reshape(n,-1),x[fm,29:58][...,store.lower].reshape(n,-1),x[:,148:157],x[:,157:184].reshape(n,3,3,3)[...,:2].reshape(n,18),relative[:,0,:,:2].reshape(n,6)),-1)
        inputs=store.inputs(store.prefetch(np.full(n,cid),ids,'cpu'),torch.tensor(robot,dtype=torch.float32))
        for m,expected in zip(MODALITIES,[motion,smpl,teleop]):
            errors[m]=max(errors[m],float(np.abs(inputs[m].numpy()-expected).max()))
        delta=gR.transpose(0,2,1)@reference_R
        angles=np.rad2deg(np.arccos(np.clip((np.trace(delta,axis1=-2,axis2=-1)-1)/2,-1,1)))
        results.append(dict(motion_id=r['motion_id'],split=r['split'],mirror=r['mirror'],frames=n,
            pelvis_position_span=float(np.ptp(joints[:,0],axis=0).max()),
            pelvis_to_head_m=float(np.linalg.norm(joints[:,15]-joints[:,0],axis=-1).mean()),
            local_left_minus_right_hip_y_m=float(np.median(local[:,1,1]-local[:,2,1])),
            g1_smpl_root_difference_deg=dict(median=float(np.median(angles)),maximum=float(angles.max()))))
    passed=max(errors.values())<=1e-4
    report=dict(contract=CONTRACT,clips=len(results),frames=sum(r['frames'] for r in results),
                max_abs_error=errors,passed=passed,records=results,
                scope='reference features, shared endpoint, and official formulas; not motion tracking quality',
                coordinate_note='SMPL joints are consumed as stored, in meters, without transl subtraction. G1/SMPL root difference includes retargeting error and is diagnostic only.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
    if not passed:raise SystemExit('Feature mismatch exceeds 1e-4')

if __name__=='__main__':main()
