"""SONIC-style scalar bottleneck, without a learned codebook.
Formula follows vector-quantize-pytorch FSQ (MIT, lucidrains) and
GR00T-WholeBodyControl all_mlp_v1: two tokens, 32 scalars, 32 levels.
Uses torch.round ties-to-even, including ONNX Round. No packed code indices.
"""
import torch
from .latent_contract import RULE

MODES = ("continuous32", "continuous64", "fsq64")
LEVELS = 32
HALF = LEVELS // 2
HALF_L = float(((torch.tensor(LEVELS) - 1) * (1 + 1e-3) / 2).item())
OFFSET = .5
SHIFT = float(torch.atanh(torch.tensor(OFFSET) / HALF_L).item())

def bound(raw):
    # Quantizer arithmetic stays fp32 under autocast.
    return torch.tanh(raw.float() + SHIFT) * HALF_L - OFFSET

def quantize_bounded(value):
    """Bounded scalar coordinates -> normalized lattice values, with STE."""
    rounded = value.round()
    return (value + (rounded - value).detach()) / HALF

def quantize_tokens(tokens):
    """Snap external normalized values to the legal lattice (explicit, not in decoder)."""
    x = tokens.float().clamp(-1., (HALF - 1) / HALF) * HALF
    return quantize_bounded(x)

def spec(mode="continuous32", ref_skip=False):
    if mode not in MODES:
        raise ValueError("unknown latent mode: " + str(mode))
    if mode != "continuous32" and ref_skip:
        raise ValueError("64D token policies require pi_ref_skip=False")
    return dict(version=1, mode=mode, latent_dim=32 if mode=="continuous32" else 64,
                token_shape=[1,32] if mode=="continuous32" else [2,32],
                levels=32 if mode=="fsq64" else None,
                bound="tanh" if mode=="continuous32" else RULE,
                quantization=RULE if mode=="fsq64" else None, ref_skip=bool(ref_skip))

def validate_args(args):
    expected = spec(args.get("latent_mode", "continuous32"), args.get("pi_ref_skip", False))
    if "latent_spec" in args and args["latent_spec"] != expected:
        raise ValueError("checkpoint latent_spec disagrees with architecture arguments")
    return expected

def check_resume(saved, current):
    for key in ("omni_contract","omni_index_sha256"):
        if saved.get(key)!=current.get(key):
            raise ValueError("resume omni architecture/data mismatch: "+key)
    from .policy_inputs import validate_args as validate_observation
    if validate_observation(saved)!=validate_observation(current):
        raise ValueError("resume observation architecture mismatch; start a separate run")
    if validate_args(saved) != validate_args(current):
        raise ValueError("resume latent architecture mismatch; start a separate run")

@torch.no_grad()
def statistics(policy, goal, orientation6d_future=None):
    raw = policy.enc_raw(goal,orientation6d_future)
    u = bound(raw)
    z = u / HALF
    q = quantize_bounded(u)
    ids = (q * HALF + HALF).long().clamp(0, LEVELS-1).reshape(-1,64)
    counts = torch.stack([(ids == k).sum(0) for k in range(LEVELS)], -1)
    saturation = ((ids == 0) | (ids == LEVELS-1)).float().mean(0)
    grads = [p.grad.float().square().sum() for p in policy.enc.parameters() if p.grad is not None]
    return dict(observation_mode=getattr(policy,"observation_mode","legacy99"),
                orientation_condition="current_reference_aligned" if getattr(policy,"observation_mode","legacy99")=="proprio93" else "reference_only",
                quantization_mse=float((q-z).square().mean()),
                used_levels=counts.gt(0).sum(-1).cpu().tolist(),
                utilization=(counts.gt(0).float().mean(-1)).cpu().tolist(),
                saturation=saturation.cpu().tolist(),
                encoder_grad_norm=float(torch.stack(grads).sum().sqrt()) if grads else None)
