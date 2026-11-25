"""ONNX token interfaces. Export excludes the auxiliary reconstruction branch."""
from pathlib import Path
import json
import hashlib
import torch
from torch import nn
from .omni import OmniPolicy,CONTRACT,DIMS,MODALITIES

def load_policy(checkpoint,device="cpu"):
    a=checkpoint["args"]
    if a.get("omni_contract")!=CONTRACT:raise ValueError("Not a compatible omni checkpoint")
    hidden=tuple(int(v) for v in a.get("dec_hidden","2048,1024,512,256").split(","))
    model=OmniPolicy(dec_hidden=hidden).to(device)
    model.load_state_dict(checkpoint["pi"])
    return model.eval()

class Encoder(nn.Module):
    def __init__(self,policy,modality):super().__init__();self.policy=policy;self.modality=modality
    def forward(self,features):return self.policy.encode_tokens(self.modality,features)

class Decoder(nn.Module):
    def __init__(self,policy):super().__init__();self.policy=policy
    def forward(self,obs_history,tokens):return self.policy.decode_tokens(obs_history,tokens)

def export(policy,out):
    import numpy as np
    import onnxruntime as ort
    from . import networks
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    policy=policy.cpu().eval()
    old=networks.AMP_DTYPE;networks.AMP_DTYPE=None
    modules={m+"_encoder":(Encoder(policy,m),(torch.randn(2,d),),["features"],"tokens") for m,d in DIMS.items()}
    modules["decoder"]=(Decoder(policy),(torch.randn(2,10,93),torch.randn(2,2,32)),["obs_history","tokens"],"actions")
    hashes={};errors={}
    try:
        for name,(module,inputs,names,target) in modules.items():
            path=out/(name+".onnx")
            torch.onnx.export(module,inputs,str(path),input_names=names,output_names=[target],
                              dynamic_axes={k:{0:"batch"} for k in names+[target]},opset_version=17,dynamo=False)
            session=ort.InferenceSession(str(path),providers=["CPUExecutionProvider"])
            error=0.
            for batch in (1,3,7):
                xs=tuple(torch.randn(batch,*x.shape[1:]) for x in inputs)
                with torch.no_grad():expected=module(*xs).numpy()
                actual=session.run(None,{k:x.numpy() for k,x in zip(names,xs)})[0]
                error=max(error,float(np.max(np.abs(expected-actual))))
            # Encoder rounding can differ at a bin boundary; action agreement
            # is additionally checked through the complete split chain below.
            errors[name]=error
            hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        decoder=ort.InferenceSession(str(out/"decoder.onnx"),providers=["CPUExecutionProvider"])
        for m,d in DIMS.items():
            encoder=ort.InferenceSession(str(out/(m+"_encoder.onnx")),providers=["CPUExecutionProvider"])
            err=0.
            for batch in (1,3,7):
                h=torch.randn(batch,10,93);x=torch.randn(batch,d)
                with torch.no_grad():expected=policy.decode_tokens(h,policy.encode_tokens(m,x)).numpy()
                z=encoder.run(None,{"features":x.numpy()})[0]
                actual=decoder.run(None,{"obs_history":h.numpy(),"tokens":z})[0]
                err=max(err,float(np.max(np.abs(expected-actual))))
            errors[m+"_action"]=err
            if err>1e-3:raise RuntimeError(f"{m} ONNX action error {err}")
        report=dict(contract=policy.metadata(),sha256=hashes,errors=errors,passed=True)
        (out/"omni_config.json").write_text(json.dumps(report,indent=2))
        return report
    finally:networks.AMP_DTYPE=old
