"""Policy observation ABI. Importable by NumPy-only deployment."""
MODES=("legacy99","proprio93")
def contract(mode="legacy99"):
    if mode not in MODES:raise ValueError("unknown observation mode: "+str(mode))
    if mode=="legacy99":
        return dict(version=1,mode=mode,obs_dim=99,orientation_dim=0,orientation="none",
                    history_order="oldest_first",orientation_history="latest_only")
    return dict(version=2,mode=mode,obs_dim=93,goal_dim=730,encoder_input_dim=790,
                orientation_shape=[10,6],orientation_dim=60,
                orientation="current_robot_T_aligned_future_reference_first2columns_rowmajor",
                future_indices="f+1..f+10_clip_last_clamped",
                history_order="oldest_first",orientation_history="future_window_current_robot_frame")
def validate_args(args):
    result=contract(args.get("observation_mode","legacy99"))
    if result["mode"]=="proprio93":
        if args.get("latent_mode","continuous32")=="continuous32" or args.get("pi_ref_skip",False):
            raise ValueError("proprio93 requires a 64D policy without pi_ref_skip")
        if args.get("hist",10)!=10 or args.get("goal_frames",10)!=10:
            raise ValueError("proprio93 requires 10 history and 10 future frames")
    if "observation_spec" in args and args["observation_spec"]!=result:
        raise ValueError("checkpoint observation_spec mismatch (current-only 6D is unsupported)")
    return result
