"""Manual grasp binding helpers: asset-local X closing and Z approach."""
from __future__ import annotations

import numpy as np

CONVENTION = "manual_x_closing_z_approach"


class GraspBindingError(ValueError):
    """标注/读取错误，不能计作物理失败。"""


def column_matrix(value):
    """Return a validated column-vector 4x4 matrix from USD matrix4d/legacy array."""
    array = np.asarray(value, dtype=float)
    if array.shape == (4, 4):
        array = array.T.copy()
    elif array.shape == (16,):
        array = array.reshape(4, 4)
    else:
        raise GraspBindingError("invalid_grasp_matrix_shape")
    rotation = array[:3, :3]
    if (
        not np.isfinite(array).all()
        or not np.allclose(array[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise GraspBindingError("invalid_grasp_rigid_transform")
    return array


def local_axes_from_matrix(matrix):
    return matrix[:3, 0].copy(), matrix[:3, 2].copy()


def sync_stage(stage, asset_path, paths, apply=False):
    from pxr import Gf, Sdf, UsdGeom
    cache = UsdGeom.XformCache()
    root = stage.GetPrimAtPath(asset_path)
    if not root:
        raise ValueError("missing_asset_root")
    inverse = cache.GetLocalToWorldTransform(root).GetInverse()
    prepared = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not str(path).startswith(asset_path + "/grasps/"):
            raise ValueError("missing_or_out_of_scope_grasp:" + path)
        relative = cache.GetLocalToWorldTransform(prim) * inverse
        matrix = column_matrix(relative)
        prepared.append((prim, relative, matrix))
    report = []
    for prim, relative, matrix in prepared:
        closing, approach = local_axes_from_matrix(matrix)
        report.append(dict(prim=str(prim.GetPath()), closing=closing.tolist(), approach=approach.tolist()))
        if apply:
            for name, kind, value in [
                ("grasp:pose_matrix", Sdf.ValueTypeNames.Matrix4d, relative),
                ("grasp:finger_closing", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*closing)),
                ("grasp:approach", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*approach)),
                ("grasp:frame_convention", Sdf.ValueTypeNames.Token, CONVENTION),
            ]:
                attr = prim.GetAttribute(name)
                if attr and attr.GetTypeName() != kind:
                    prim.RemoveProperty(name)
                prim.CreateAttribute(name, kind, custom=True).Set(value)
    return report
