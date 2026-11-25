"""NumPy-runtime-compatible multimodal feature contract."""
MODALITIES = ("motion", "smpl", "teleop")
DIMS = {"motion": 640, "smpl": 840, "teleop": 273}
CONTRACT = dict(version=1, policy_type="omni", reference_commit="2f07b35b3f3b9053b1f4fd2adeaf56b6bce32366",
                observation_shape=[10,93], token_shape=[2,32], levels=32,
                motion_offsets=list(range(0,50,5)), smpl_offsets=list(range(10)),
                rotation="first_two_columns_row_major", dims=DIMS,
                joint_order="BlinkMimic RobotSpec by name", fps=50)
WRISTS = tuple(f"{side}_wrist_{axis}_joint" for axis in ("roll","pitch","yaw") for side in ("left","right"))

