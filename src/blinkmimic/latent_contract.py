"""NumPy-runtime-compatible token metadata; no torch dependency."""
RULE = "fsq32_eps1e-3_evenoffset_round_even_v1"

def validate_v2(config):
    s = config.get("latent_spec", {})
    mode = s.get("mode")
    expected = dict(version=1, mode=mode, latent_dim=64, token_shape=[2,32],
                    levels=32 if mode=="fsq64" else None, bound=RULE,
                    quantization=RULE if mode=="fsq64" else None, ref_skip=False)
    if mode not in ("continuous64","fsq64") or s != expected:
        raise ValueError("unsupported v2 latent contract")
