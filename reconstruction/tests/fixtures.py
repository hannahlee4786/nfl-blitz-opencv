from nfl_reconstruction.schemas import CameraState


def synthetic_state(**changes):
    values = {
        "target_x": 55.0,
        "target_y": 26.6665,
        "distance": 55.0,
        "yaw_degrees": 0.0,
        "pitch_degrees": 48.0,
        "focal_px": 720.0,
        "roll_degrees": 0.0,
    }
    values.update(changes)
    return CameraState(**values)

