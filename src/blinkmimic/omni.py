"""Three independent encoders and one FSQ action decoder.

Feature/loss design references NVIDIA GR00T-WholeBodyControl (Apache-2.0),
commit 2f07b35b3f3b9053b1f4fd2adeaf56b6bce32366, sonic_release.
No pretrained NVIDIA weights are used. See docs/omni.md for adaptation boundaries.
"""
import torch
from torch import nn
from .networks import mlp, run_mlp, RunningNorm
from .latent import bound, quantize_bounded, spec

from .omni_contract import MODALITIES,DIMS,CONTRACT,WRISTS

from .omni_features import future_indices, smpl_reference, assemble

class OmniPolicy(nn.Module):
    policy_type = "omni"
    observation_mode = "proprio93"
    latent_mode = "fsq64"
    def __init__(self, enc_hidden=(2048,1024,512,512), dec_hidden=(2048,1024,512,256)):
        super().__init__()
        self.enc_hidden, self.dec_hidden = tuple(enc_hidden), tuple(dec_hidden)
        self.encoders = nn.ModuleDict({m:mlp(d,list(enc_hidden),64,act=nn.SiLU) for m,d in DIMS.items()})
        self.input_norms = nn.ModuleDict({m:RunningNorm(d) for m,d in DIMS.items()})
        self.obs_norm = RunningNorm(93)
        self.dec = mlp(994,list(dec_hidden),29)
        self.reconstruction = mlp(64,list(enc_hidden[::-1]),640,act=nn.SiLU)
        self.latent_spec = spec("fsq64",False)
        with torch.no_grad():
            self.dec[-1].weight.mul_(.01)
            self.dec[-1].bias.zero_()

    def enc_raw(self, modality, inputs):
        return run_mlp(self.encoders[modality],self.input_norms[modality](inputs)).float()

    def encode_tokens(self, modality, inputs):
        return quantize_bounded(bound(self.enc_raw(modality,inputs))).unflatten(-1,(2,32))

    def decode_tokens(self, obs_history, tokens):
        return run_mlp(self.dec,torch.cat((tokens.flatten(-2),self.obs_norm(obs_history).flatten(-2)),-1))

    def forward(self, obs_history, inputs, modality):
        # Fixed layout for CUDA graph capture: no data-dependent slicing.
        tokens = torch.stack([self.encode_tokens(m,inputs[m]) for m in MODALITIES],-3)
        mask = torch.nn.functional.one_hot(modality,3).to(tokens.dtype)
        selected = (tokens*mask[...,None,None]).sum(-3)
        return self.decode_tokens(obs_history,selected)

    @torch.no_grad()
    def update_norms(self, obs_history, inputs):
        self.obs_norm.update(obs_history)
        for m in MODALITIES:self.input_norms[m].update(inputs[m])

    def auxiliary(self, inputs):
        """Release non-compliance MSE terms on RAW (pre-FSQ) latent.

        Standard release losses are symmetric, including the cycle target.
        Compliance-aware detached teacher losses are NOT enabled.
        Return per-sample terms so distributed callers can weight valid counts.
        """
        raw = {m:self.enc_raw(m,inputs[m]) for m in MODALITIES}
        token = {m:quantize_bounded(bound(raw[m])) for m in MODALITIES}
        reconstruction = {m:run_mlp(self.reconstruction,token[m]).float() for m in MODALITIES}
        # Official G1ReconLoss uses physical-feature MSE, without normalization.
        mse=lambda x,y:(x-y).square().mean(-1)
        recon = torch.stack([mse(reconstruction[m],inputs["motion"]) for m in MODALITIES]).mean(0)
        cycle = self.enc_raw("motion",reconstruction["smpl"])
        return dict(g1_recon=recon,
                    g1_smpl_latent=mse(raw["motion"],raw["smpl"]),
                    g1_teleop_latent=mse(raw["motion"],raw["teleop"]),
                    teleop_smpl_latent=mse(raw["teleop"],raw["smpl"]),
                    reencoded_smpl_g1_latent=mse(cycle,raw["motion"]))

    def metadata(self):
        return dict(CONTRACT,enc_hidden=list(self.enc_hidden),dec_hidden=list(self.dec_hidden),
                    latent_spec=self.latent_spec,
                    auxiliary_weights={"g1_recon":.01,"g1_smpl_latent":1.,"g1_teleop_latent":1.,
                                       "teleop_smpl_latent":1.,"reencoded_smpl_g1_latent":1.})

def auxiliary_total(terms, valid=None):
    total=sum(v*(.01 if k=="g1_recon" else 1.) for k,v in terms.items())
    valid=torch.ones_like(total) if valid is None else valid.to(total.dtype)
    return (total*valid).sum(),valid.sum()
