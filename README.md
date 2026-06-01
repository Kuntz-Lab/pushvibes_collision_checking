# PushVIBES — Collision Checking & Visualization

Tools for ranking and visualizing the candidate "push" actions that the PushVIBES /
DiffDef policy predicts for a deformable-tissue manipulation scene, scored by how
close each push comes to colliding with an artery (the obstacle).

For each scene the policy emits ~500 candidate pushes (start point + displacement)
plus the single chosen push. Each push is modelled as the **capsule** swept by the
robot tip (a sphere) as it travels from a start point to an end point. The artery is
modelled as a **tube** of constant radius around a centerline spline fit to its point
cloud. A push collides when the capsule and the tube overlap; we score every push by
its *signed surface clearance* (positive = room to spare, negative = penetration
depth) and rank them safest-first.

## Quick start

```bash
# Visualize the chosen push + the full candidate fan + collision status
python visualize.py            # version 1 (default)
python visualize.py 2          # version 2
python visualize.py 1 2 3      # all three side by side

# Rank all ~500 candidates by clearance and draw the 5 safest vs 5 riskiest
python visualize_ranked.py 1 2 3
```

Both scripts open an interactive Matplotlib (TkAgg) window. To render headless,
change `matplotlib.use("TkAgg")` to `"Agg"` near the top of `visualize.py` and save
the figure instead of `plt.show()`.

## Dependencies

Python 3.11. Install:

```bash
pip install numpy scipy scikit-learn networkx matplotlib torch sympy
# only needed to re-extract data from rosbags:
pip install rosbags pillow
```

`torch` is used for the batched collision ranker (uses CUDA automatically when
available, otherwise CPU). Everything else runs on CPU.

## Repository layout

```
visualize.py              # chosen push + candidate fan + per-push collision status
visualize_ranked.py       # rank all candidates, draw N safest vs N riskiest
utils/
  object_frame.py         # object-frame <-> camera-frame transform for the pushes
  spline_fit.py           # artery point cloud -> ordered centerline -> B-spline
  capsule_collision.py    # single capsule-vs-tube distance/collision (reference loop)
  collision_ranking.py    # batched (GPU) capsule-vs-tube scoring & ranking
  extract_rosbag_data.py  # rebuild obstacle_pc / tissue_pc / image from the rosbags
  inspect_data.py         # print the structure of the pickle files
data/                     # input data (git-ignored), see below
```

## Data structure

All data lives under `data/PushVIB3S_procedure_level_plan_example/` and is
**git-ignored** (`.gitignore` excludes `data/*`) — you must obtain it separately and
drop it in place. The scripts hard-code `DATA_DIR` to this folder.

There are three scene "versions" (`v1`, `v2`, `v3`), each a different predicted plan
for the same goal. Files per version:

| File | Type | Contents |
| --- | --- | --- |
| `thanks_joe_vibes_v{N}.pickle` | dict | The policy output for version N (see keys below). |
| `obstacle_pc_v{N}.npy` | `(M, 3)` float | Artery point cloud (the obstacle), camera frame, meters. |
| `tissue_pc_v{N}.npy` | `(M, 3)` float | Tissue point cloud, camera frame (not used by the visualizers — they read the clouds from the pickle). |
| `image_v{N}.png` / `.npy` | image | RGB scene photo for the side panel. |
| `rosbags/thanks_joe_v{N}_*.bag` | rosbag | Raw capture the `.npy`/`.png` above were extracted from. |

Shared (not per-version):

| File | Type | Contents |
| --- | --- | --- |
| `thanks_joe_goal.pickle` | `(~22868, 3)` float64 | The goal tissue point cloud. |
| `rosbags/thanks_joe_expert_demo_*.bag` | rosbag | Expert demonstration capture. |

### `thanks_joe_vibes_v{N}.pickle` keys

A single dict with 14 keys:

| Key | Shape / type | Frame | Meaning |
| --- | --- | --- | --- |
| `pred_start` | `(3,)` float32 | **camera** | Chosen push start point (on the current tissue surface). |
| `pred_disp` | `(3,)` float32 | camera | Chosen push displacement vector. |
| `pred_end` | `(3,)` float32 | camera | Chosen push end point (`= pred_start + pred_disp`, on the goal surface). |
| `original_pred_start/disp/end` | `(3,)` float32 | camera | Pre-adjustment copies of the above (identical in this dataset). |
| `preds` | dict (see below) | **object** | The full candidate set the chosen push was selected from. |
| `table_plane_equation` | sympy `Add` | camera (mm) | Plane `ax + by + cz + d = 0` for the table. **Fit in millimeters** — divide `d` by 1000 to use with the meter-scale clouds. |
| `camera_frame_current_pc` | `(M, 3)` float64 | camera | Current ("start") tissue point cloud, meters. |
| `camera_frame_goal_pc` | `(M, 3)` float64 | camera | Goal tissue point cloud, meters. |
| `pre_push_chamfer` | float | — | Chamfer distance current→goal before the push. |
| `post_push_chamfer` | float or `None` | — | Chamfer after the push (`None` when not executed). |
| `executed_push` | bool | — | Whether the push was physically executed. |
| `push_method` | str | — | e.g. `"PushVIBES"`. |

`preds` holds the ~500 candidate pushes, each a sub-array:

| `preds[...]` | Shape | Frame | Meaning |
| --- | --- | --- | --- |
| `start_point` | `(500, 3)` float32 | **object** | Candidate start points. |
| `displacement` | `(500, 3)` float32 | object | Candidate displacement vectors (`end = start + disp`). |
| `normalized_start_point` | `(500, 3)` float32 | object, normalized | Network-space (unit-scaled) start points. |
| `normalized_displacement` | `(500, 3)` float32 | object, normalized | Network-space displacement. |

### Coordinate frames (important)

There are **two frames** in play:

- **Camera frame** (meters): the point clouds, the table plane (offset in mm), and
  the *chosen* push (`pred_start` / `pred_end`) all live here. These overlay each
  other directly.
- **Object frame**: a PCA-aligned, origin-centered frame built from the current
  tissue cloud — what the DiffDef network runs inference in. The **candidate fan**
  (`preds["start_point"]` / `displacement`) lives here, so it does *not* overlay the
  camera-frame clouds as stored.

`utils/object_frame.py` reconstructs the object↔camera transform (replicating
DiffDef's `world_to_object_frame_PCA`, including the sklearn PCA sign convention and a
z-reflection) and maps the candidates back into the camera frame. This is verified:
the chosen push round-trips onto its matching candidate to < 1.5 mm. Use
`candidate_pushes_in_camera_frame(d)` to get camera-frame `(starts, ends)`.

## Geometry pipeline

1. **Artery centerline** (`utils/spline_fit.py`): Mean Shift on the obstacle cloud →
   skeleton nodes → minimum spanning tree longest path to order them → regularized
   cubic B-spline. The spline is extrapolated past its visible ends to estimate where
   the artery continues underneath the tissue.
2. **Collision model**: the robot tip is a sphere of `ROBOT_TIP_RADIUS`; swept along a
   push it forms a capsule. The artery is a tube of `SPLINE_RADIUS` (= the Mean Shift
   `ARTERY_BANDWIDTH`) around the spline. Signed clearance = (closest core distance) −
   (capsule radius + tube radius).
3. **Scoring**: `utils/capsule_collision.py` is the readable per-push reference;
   `utils/collision_ranking.py` is the batched/GPU equivalent that scores all 500
   candidates at once. Both compute the same signed clearance (cross-checked in the
   module `__main__` self-tests).

Tuning constants live at the top of `visualize.py` (`ARTERY_BANDWIDTH`,
`ARTERY_SMOOTHING`, `ARTERY_EXTRAPOLATE`, `ROBOT_TIP_RADIUS`, `SPLINE_RADIUS`, and the
`CLEARANCE_NORM` heat-map range) and are imported by `visualize_ranked.py`.

## Regenerating data from rosbags

`obstacle_pc_v{N}.npy`, `tissue_pc_v{N}.npy`, and `image_v{N}.npy/.png` are extracted
from the rosbags (first message of each topic):

```bash
python utils/extract_rosbag_data.py
```

Topics: `/obstacle_pointcloud`, `/tissue_pointcloud`, `/camera/color/image_raw`.

## Inspecting the pickles

```bash
python utils/inspect_data.py        # prints shapes/dtypes/ranges of every key
python utils/object_frame.py        # round-trip check: chosen push -> candidate (<1.5mm)
```
