"""Pose conversions shared by training transforms and inference adapters."""

import numpy as np
import scipy.spatial.transform as st


def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros((*shape, 4, 4), dtype=pos.dtype)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rot.as_matrix()
    mat[..., 3, 3] = 1
    return mat


def mat_to_pos_rot(mat):
    pos = (mat[..., :3, 3].T / mat[..., 3, 3].T).T
    rot = st.Rotation.from_matrix(mat[..., :3, :3])
    return pos, rot


def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros((*shape, 6), dtype=pos.dtype)
    pose[..., :3] = pos
    pose[..., 3:] = rot.as_rotvec()
    return pose


def pose_to_pos_rot(pose):
    pos = pose[..., :3]
    rot = st.Rotation.from_rotvec(pose[..., 3:])
    return pos, rot


def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))


def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))


def convert_pose_mat_rep(pose_mat, base_pose_mat, pose_rep="abs", *, backward=False):
    if not backward:
        # training transform
        if pose_rep == "abs":
            return pose_mat
        if pose_rep == "rel":
            # legacy buggy implementation
            # for compatibility
            pos = pose_mat[..., :3, 3] - base_pose_mat[:3, 3]
            rot = pose_mat[..., :3, :3] @ np.linalg.inv(base_pose_mat[:3, :3])
            out = np.copy(pose_mat)
            out[..., :3, :3] = rot
            out[..., :3, 3] = pos
            return out
        if pose_rep == "relative":
            return np.linalg.inv(base_pose_mat) @ pose_mat
        if pose_rep == "delta":
            all_pos = np.concatenate([base_pose_mat[None, :3, 3], pose_mat[..., :3, 3]], axis=0)
            out_pos = np.diff(all_pos, axis=0)

            all_rot_mat = np.concatenate([base_pose_mat[None, :3, :3], pose_mat[..., :3, :3]], axis=0)
            prev_rot = np.linalg.inv(all_rot_mat[:-1])
            curr_rot = all_rot_mat[1:]
            out_rot = np.matmul(curr_rot, prev_rot)

            out = np.copy(pose_mat)
            out[..., :3, :3] = out_rot
            out[..., :3, 3] = out_pos
            return out
        raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")

    # eval transform
    if pose_rep == "abs":
        return pose_mat
    if pose_rep == "rel":
        # legacy buggy implementation
        # for compatibility
        pos = pose_mat[..., :3, 3] + base_pose_mat[:3, 3]
        rot = pose_mat[..., :3, :3] @ base_pose_mat[:3, :3]
        out = np.copy(pose_mat)
        out[..., :3, :3] = rot
        out[..., :3, 3] = pos
        return out
    if pose_rep == "relative":
        return base_pose_mat @ pose_mat
    if pose_rep == "delta":
        output_pos = np.cumsum(pose_mat[..., :3, 3], axis=0) + base_pose_mat[:3, 3]

        output_rot_mat = np.zeros_like(pose_mat[..., :3, :3])
        curr_rot = base_pose_mat[:3, :3]
        for i in range(len(pose_mat)):
            curr_rot = pose_mat[i, :3, :3] @ curr_rot
            output_rot_mat[i] = curr_rot

        out = np.copy(pose_mat)
        out[..., :3, :3] = output_rot_mat
        out[..., :3, 3] = output_pos
        return out
    raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")
