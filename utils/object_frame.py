"""Object-frame <-> camera-frame transform for PushVIBES push actions.

The DiffDef model runs inference in an *object frame*: a PCA-aligned, origin-
centered frame built from the start (current) tissue point cloud. The 500
candidate pushes in ``d["preds"]`` (``start_point`` / ``displacement``) live in
that object frame, so they do NOT overlay the camera-frame point clouds. The
single chosen push (``pred_start`` / ``pred_end``) is already in the camera
frame -- it was the chosen candidate mapped back out.

This module reconstructs the camera->object transform so the candidate fan can
be plotted alongside the clouds. It mirrors DiffDef-core's
``world_to_object_frame_PCA`` (utils/object_frame_utils.py): PCA on the cloud,
take the first two principal axes as object x/y and ``z = x cross y``, then the
world->object map is ``R_w_o = [x y z]^T`` about the cloud centroid.

Two details, both verified empirically against this data:

* PCA sign convention. sklearn's ``PCA`` runs ``svd_flip`` so the component
  signs are deterministic. We replicate it (numpy ``svd`` leaves signs
  arbitrary) so the reconstructed axes match the ones used at inference.

* Z reflection. The frame DiffDef stored for these pushes has its z-axis
  pointing opposite to ``x cross y``. Applying ``diag(1, 1, -1)`` in the object
  frame is what makes the chosen camera-frame push (``pred_start``/``pred_end``)
  land back on its candidate to <1.5 mm across all three versions; without it
  the error is 6-19 mm and the displacement direction is wrong. See the module
  test at the bottom for the check.

The transform is built from the *full* start cloud (deterministic -- no random
downsample needed); the result is stable to sub-millimeter vs. the 512-point
downsample DiffDef used.
"""

import numpy as np

# Object-frame reflection that matches DiffDef's stored push frame (see above).
_Z_FLIP = np.diag([1.0, 1.0, -1.0, 1.0])


def _pca_axes(points):
    """First two principal axes (x, y) and centroid of ``points``.

    Replicates DiffDef ``find_pca_axes`` + sklearn ``PCA(svd_solver='full')``,
    including ``svd_flip`` so the axis signs are deterministic and match what
    ran at inference. ``z = x cross y`` is added by the caller.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    u, _, vt = np.linalg.svd(centered, full_matrices=False)
    # svd_flip (u_based_decision=True): pin each component's sign to the entry
    # with the largest magnitude in the corresponding left singular vector.
    max_rows = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_rows, range(u.shape[1])])
    vt = vt * signs[:, None]
    return vt[0], vt[1], centroid


def camera_to_object_transform(start_pc):
    """4x4 homogeneous map from camera frame to DiffDef's object frame.

    ``start_pc`` is the camera-frame current/start tissue cloud
    (``camera_frame_current_pc``). Returns the same transform DiffDef applied to
    the push actions before inference, including the z reflection.
    """
    start_pc = np.asarray(start_pc, dtype=float)
    x_axis, y_axis, centroid = _pca_axes(start_pc)
    z_axis = np.cross(x_axis, y_axis)

    r_o_w = np.column_stack((x_axis, y_axis, z_axis))  # object axes in camera frame
    r_w_o = r_o_w.T                                    # camera -> object rotation

    mat = np.eye(4)
    mat[:3, :3] = r_w_o
    mat[:3, 3] = -r_w_o @ centroid
    return _Z_FLIP @ mat


def object_to_camera_transform(start_pc):
    """4x4 homogeneous map from DiffDef's object frame back to the camera frame."""
    return np.linalg.inv(camera_to_object_transform(start_pc))


def transform_points(points, matrix):
    """Apply a 4x4 homogeneous ``matrix`` to an (N, 3) array of points."""
    points = np.asarray(points, dtype=float)
    homogeneous = np.hstack((points, np.ones((len(points), 1))))
    return (homogeneous @ matrix.T)[:, :3]


def candidate_pushes_in_camera_frame(d):
    """Map the 500 object-frame candidate pushes in ``d`` into the camera frame.

    ``d`` is a loaded ``thanks_joe_vibes_v*.pickle`` dict. Returns
    ``(starts, ends)``, each (N, 3) in the camera frame, ready to overlay the
    point clouds next to ``pred_start`` / ``pred_end``.
    """
    starts_obj = np.asarray(d["preds"]["start_point"], dtype=float)
    ends_obj = starts_obj + np.asarray(d["preds"]["displacement"], dtype=float)
    obj_to_cam = object_to_camera_transform(d["camera_frame_current_pc"])
    return transform_points(starts_obj, obj_to_cam), transform_points(ends_obj, obj_to_cam)


if __name__ == "__main__":
    # Sanity check: the chosen push (pred_start/pred_end, already in camera
    # frame) must map into the object frame onto one of the candidates.
    import pickle
    from pathlib import Path

    data_dir = Path("data/PushVIB3S_procedure_level_plan_example")
    for v in (1, 2, 3):
        d = pickle.load(open(data_dir / f"thanks_joe_vibes_v{v}.pickle", "rb"))
        cam_to_obj = camera_to_object_transform(d["camera_frame_current_pc"])
        ps = np.asarray(d["pred_start"]).ravel()
        pe = np.asarray(d["pred_end"]).ravel()
        so = transform_points(ps[None], cam_to_obj)[0]
        eo = transform_points(pe[None], cam_to_obj)[0]
        sp = np.asarray(d["preds"]["start_point"])
        ep = sp + np.asarray(d["preds"]["displacement"])
        err = np.linalg.norm(sp - so, axis=1) + np.linalg.norm(ep - eo, axis=1)
        j = int(err.argmin())
        print(f"v{v}: chosen push -> candidate {j}: "
              f"start_err={np.linalg.norm(sp[j] - so) * 1000:.1f} mm, "
              f"end_err={np.linalg.norm(ep[j] - eo) * 1000:.1f} mm")
