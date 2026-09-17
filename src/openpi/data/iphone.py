"""
Utility functions for processing iPhone recording data.
"""

from enum import Enum
from enum import auto
import json
import logging
import os
import tarfile

import bson
import cv2
import numpy as np
from tqdm import tqdm
import transforms3d as t3d
import zarr

from openpi.shared.archive import extract_recording

# ============================================================================
# Data Models
# ============================================================================


class ActionType(Enum):
    LEFT_ARM_6DOF_GRIPPER_WIDTH = auto()  # (x, y, z, rot6d, gripper_width)
    LEFT_ARM_3D_TRANSLATION_GRIPPER_WIDTH = auto()  # (x, y, z, gripper_width)
    RIGHT_ARM_6DOF_GRIPPER_WIDTH = auto()  # (x, y, z, rot6d, gripper_width)
    DUAL_ARM_6DOF_GRIPPER_WIDTH = (
        auto()
    )  # (l_x, l_y, l_z, l_rot6d, r_x, r_y, r_z, r_rot6d, l_gripper_width, r_gripper_width)
    DUAL_ARM_3D_TRANSLATION_GRIPPER_WIDTH = auto()  # (l_x, l_y, l_z, r_x, r_y, r_z, l_gripper_width, r_gripper_width)


# ============================================================================
# Image Utils
# ============================================================================


def center_crop_and_resize_image(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Center crop the image to square using min(h, w), then resize to target size.
    """
    h, w = image.shape[:2]
    min_dim = min(h, w)

    y_start = (h - min_dim) // 2
    x_start = (w - min_dim) // 2

    cropped_img = image[y_start : y_start + min_dim, x_start : x_start + min_dim]
    return cv2.resize(cropped_img, target_size)


# ============================================================================
# Space Utils
# ============================================================================


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector (batch * 3)."""
    v_mag = np.linalg.norm(v, axis=1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


def pose_7d_to_pose_6d(pose: np.ndarray) -> np.ndarray:
    """Convert 7D pose (x, y, z, qw, qx, qy, qz) to 6D pose (x, y, z, r, p, y)."""
    quat = pose[3:]
    euler = t3d.euler.quat2euler(quat)
    return np.concatenate([pose[:3], euler])


def pose_6d_to_4x4matrix(pose: np.ndarray) -> np.ndarray:
    """Convert 6D pose (x, y, z, r, p, y) to 4x4 transformation matrix."""
    mat = np.eye(4)
    quat = t3d.euler.euler2quat(pose[3], pose[4], pose[5])
    mat[:3, :3] = t3d.quaternions.quat2mat(quat)
    mat[:3, 3] = pose[:3]
    return mat


def ortho6d_to_rotation_matrix(ortho6d: np.ndarray) -> np.ndarray:
    """Compute rotation matrix from ortho6d representation."""
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
    x = normalize_vector(x_raw)
    z = np.cross(x, y_raw)
    z = normalize_vector(z)
    y = np.cross(z, x)

    x = x[:, :, np.newaxis]
    y = y[:, :, np.newaxis]
    z = z[:, :, np.newaxis]

    return np.concatenate((x, y, z), axis=2)


def pose_6d_to_pose_9d(pose: np.ndarray) -> np.ndarray:
    """
    Convert 6D state to 9D state.
    :param pose: np.ndarray (6,), (x, y, z, rx, ry, rz)
    :return: np.ndarray (9,), (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3)
    """
    rot_6d = pose_6d_to_4x4matrix(pose)[:3, :2].T.flatten()
    return np.concatenate((pose[:3], rot_6d), axis=0)


def pose_3d_9d_to_homo_matrix_batch(pose: np.ndarray) -> np.ndarray:
    """
    Convert 3D / 9D states to 4x4 matrix.
    :param pose: np.ndarray (N, 9) or (N, 3)
    :return: np.ndarray (N, 4, 4)
    """
    assert pose.shape[1] in [3, 9], "pose should be (N, 3) or (N, 9)"
    mat = np.eye(4)[None, :, :].repeat(pose.shape[0], axis=0)
    mat[:, :3, 3] = pose[:, :3]
    if pose.shape[1] == 9:
        mat[:, :3, :3] = ortho6d_to_rotation_matrix(pose[:, 3:9])
    return mat


def homo_matrix_to_pose_9d_batch(mat: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 matrix to 9D state.
    :param mat: np.ndarray (N, 4, 4)
    :return: np.ndarray (N, 9)
    """
    assert mat.shape[1:] == (4, 4), "mat should be (N, 4, 4)"
    pose = np.zeros((mat.shape[0], 9))
    pose[:, :3] = mat[:, :3, 3]
    pose[:, 3:9] = mat[:, :3, :2].swapaxes(1, 2).reshape(mat.shape[0], -1)
    return pose


# ============================================================================
# Data Post Processing Manager
# ============================================================================


class DataPostProcessingManageriPhone:
    def __init__(
        self,
        *,
        image_resize_shape: tuple = (320, 240),
        use_6d_rotation: bool = True,
        debug: bool = False,
    ):
        self.use_6d_rotation = use_6d_rotation
        self.resize_shape = image_resize_shape
        self.debug = debug

    @staticmethod
    def load_bson_file(file_path: str) -> dict | None:
        try:
            with open(file_path, "rb") as f:
                bson_data = f.read()
            try:
                bson_dict = bson.loads(bson_data)
            except AttributeError:
                bson_dict = bson.decode(bson_data)
            return bson_dict
        except Exception as e:
            logging.warning(f"Failed to load bson file {file_path}: {e}")
            return None

    @staticmethod
    def get_numpy_arrays(data: dict) -> tuple:
        timestamps = np.array(data.get("timestamps", []))
        arkit_poses = np.array(data.get("arkitPose", []))
        gripper_widths = np.array(data.get("gripperWidth", []))
        return timestamps, arkit_poses, gripper_widths

    def read_bson(self, file_path: str) -> tuple:
        data = self.load_bson_file(file_path)
        if data is None:
            return None, None, None
        return self.get_numpy_arrays(data)

    def load_video_frames(self, video_path: str) -> np.ndarray | None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning(f"Error: Could not open video {video_path}")
            return None
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        cap.release()
        return np.array(frames)

    def extract_msg_to_obs_dict(
        self,
        session: dict,
        clip_head_seconds: float = 0.0,
        clip_tail_seconds: float = 0.0,
    ) -> dict[str, np.ndarray] | None:
        obs_dict = {}
        timestamps = {}
        arkit_poses = {}
        gripper_widths = {}

        for record in session.values():
            metadata_path = os.path.join(record, "metadata.json")
            with open(metadata_path) as f:
                metadata = json.load(f)
            camera_position = metadata.get("camera_position", "left_wrist")

            bson_path = os.path.join(record, "frame_data.bson")
            t, a, g = self.read_bson(bson_path)

            if t is None or len(t) == 0:
                continue

            timestamps[camera_position] = t
            arkit_poses[camera_position] = a
            gripper_widths[camera_position] = g

        if not timestamps:
            return None

        latest_start_time = max([t[0] for t in timestamps.values()])
        earliest_end_time = min([t[-1] for t in timestamps.values()])

        clipped_start_time = latest_start_time + clip_head_seconds
        clipped_end_time = earliest_end_time - clip_tail_seconds

        if clipped_start_time >= clipped_end_time:
            return None

        start_frame_indices = {}
        end_frame_indices = {}
        for k, v in timestamps.items():
            start_frame_indices[k] = np.searchsorted(v, clipped_start_time, side="left")
            end_frame_indices[k] = np.searchsorted(v, clipped_end_time, side="right")

        num_frames = min([end_frame_indices[k] - start_frame_indices[k] for k in timestamps])

        if num_frames <= 0:
            return None

        def project_data(data_dict, start_indices, num_frames):
            return {k: v[start_indices[k] : start_indices[k] + num_frames] for k, v in data_dict.items()}

        timestamps = project_data(timestamps, start_frame_indices, num_frames)
        arkit_poses = project_data(arkit_poses, start_frame_indices, num_frames)
        gripper_widths = project_data(gripper_widths, start_frame_indices, num_frames)

        # Convert quat format from [x, y, z, w] to [w, x, y, z]
        for k in arkit_poses:
            arkit_poses[k][:, 3:] = arkit_poses[k][:, [6, 3, 4, 5]]

        # Extend dims
        for k in timestamps:
            timestamps[k] = timestamps[k][:, np.newaxis]
            gripper_widths[k] = gripper_widths[k][:, np.newaxis]

        if "left_wrist" not in timestamps and "right_wrist" not in timestamps:
            return None

        obs_dict["timestamp"] = timestamps.get("left_wrist", timestamps.get("right_wrist"))

        for k in arkit_poses:
            key_prefix = "right" if "right" in k else "left"
            if self.use_6d_rotation:
                a_9d = [pose_6d_to_pose_9d(pose_7d_to_pose_6d(pose)) for pose in arkit_poses[k]]
                obs_dict[f"{key_prefix}_robot_tcp_pose"] = a_9d
            else:
                obs_dict[f"{key_prefix}_robot_tcp_pose"] = arkit_poses[k]

            obs_dict[f"{key_prefix}_robot_gripper_width"] = gripper_widths[k]

            images_array = self.load_video_frames(os.path.join(session[k], "recording.mp4"))
            if images_array is None:
                return None

            obs_dict[f"{key_prefix}_wrist_img"] = [
                cv2.resize(image, self.resize_shape)
                for image in images_array[start_frame_indices[k] : start_frame_indices[k] + num_frames]
            ]

        return obs_dict


# ============================================================================
# Convert Data to Zarr
# ============================================================================


def convert_data_to_zarr(
    input_dir: str,
    output_dir: str,
    *,
    temporal_downsample_ratio: int = 3,
    use_absolute_action: bool = True,
    action_type: ActionType = ActionType.LEFT_ARM_6DOF_GRIPPER_WIDTH,
    debug: bool = False,
    overwrite: bool = True,
    use_dino: bool = False,
    gripper_width_bias: float = 0.0,
    gripper_width_scale: float = 1.0,
    tcp_transform: np.ndarray | None = None,
    episode_clip_head_seconds: float = 0.0,
    episode_clip_tail_seconds: float = 0.0,
) -> str:
    """
    Convert raw data to zarr format.

    Args:
        input_dir: Input data directory containing .tar.gz files or .bson files
        output_dir: Output directory for saving zarr file
        temporal_downsample_ratio: Temporal downsampling ratio
        use_absolute_action: Whether to use absolute action values
        action_type: Action type enum
        debug: Whether to enable debug mode
        overwrite: Whether to overwrite existing data
        use_dino: Whether to use DINO preprocessing
        gripper_width_bias: Gripper width bias
        gripper_width_scale: Gripper width scale factor
        tcp_transform: TCP transformation matrix
        episode_clip_head_seconds: Seconds to clip from episode start
        episode_clip_tail_seconds: Seconds to clip from episode end

    Returns:
        Path to saved zarr file
    """
    if tcp_transform is None:
        tcp_transform = np.eye(4, dtype=np.float32)
    save_data_path = os.path.join(output_dir, "replay_buffer.zarr")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(save_data_path):
        if not overwrite:
            logging.info(f"Data already exists at {save_data_path}")
            return save_data_path
        logging.warning(f"Overwriting {save_data_path}")
        import shutil

        shutil.rmtree(save_data_path)

    data_processing_manager = DataPostProcessingManageriPhone(use_6d_rotation=True)

    # Initialize data arrays
    timestamp_arrays: list[np.ndarray] = []
    left_wrist_img_arrays: list[np.ndarray] = []
    left_robot_tcp_pose_arrays: list[np.ndarray] = []
    left_robot_gripper_width_arrays: list[np.ndarray] = []
    right_wrist_img_arrays: list[np.ndarray] = []
    right_robot_tcp_pose_arrays: list[np.ndarray] = []
    right_robot_gripper_width_arrays: list[np.ndarray] = []
    episode_ends_arrays: list[int] = []
    total_count = 0

    # Process all compressed data files
    data_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".tar.gz")])
    for seq_idx, data_file in enumerate(data_files):
        if debug and seq_idx <= 5:
            continue

        data_path = os.path.join(input_dir, data_file)
        abs_path = os.path.abspath(data_path)
        dst_path = abs_path.split(".tar.gz")[0]

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
            logging.info(f"Extracting {abs_path}...")
            with tarfile.open(abs_path, "r:gz") as tar:
                extract_recording(tar, dst_path)

    # Get directories containing .bson files
    with os.scandir(input_dir) as entries:
        subfolders = [entry.path for entry in entries if entry.is_dir()]
    dst_paths = [folder for folder in subfolders if any(name.endswith(".bson") for name in os.listdir(folder))]

    if not dst_paths:
        logging.warning(f"No .bson files found in subdirectories of {input_dir}")
        return save_data_path

    record_sessions: dict[str, dict[str, str]] = {}
    for dst_path in dst_paths:
        meta_path = os.path.join(dst_path, "metadata.json")
        with open(meta_path) as metadata_file:
            metadata = json.load(metadata_file)
        uuid = metadata["uuid"]
        session_uuid = metadata.get("parent_uuid", uuid)
        if session_uuid not in record_sessions:
            record_sessions[session_uuid] = {}
        camera_position = metadata.get("camera_position", "left_wrist")
        record_sessions[session_uuid][camera_position] = dst_path

    is_dual_arm = action_type in [
        ActionType.DUAL_ARM_6DOF_GRIPPER_WIDTH,
        ActionType.DUAL_ARM_3D_TRANSLATION_GRIPPER_WIDTH,
    ]
    skipped_sessions = 0

    for session in tqdm(record_sessions.items(), dynamic_ncols=True):
        obs_dict = data_processing_manager.extract_msg_to_obs_dict(
            session[1],
            clip_head_seconds=episode_clip_head_seconds,
            clip_tail_seconds=episode_clip_tail_seconds,
        )
        if obs_dict is None:
            logging.warning(f"obs_dict is None for {session[0]}")
            continue

        if is_dual_arm:
            has_left = "left_robot_tcp_pose" in obs_dict and "left_robot_gripper_width" in obs_dict
            has_right = "right_robot_tcp_pose" in obs_dict and "right_robot_gripper_width" in obs_dict
            if not (has_left and has_right):
                skipped_sessions += 1
                continue

        timestamp_arrays.append(obs_dict["timestamp"])

        for i in range(len(obs_dict["left_robot_tcp_pose"])):
            pose_array = obs_dict["left_robot_tcp_pose"][i][np.newaxis, :]
            pose_homo_matrix = pose_3d_9d_to_homo_matrix_batch(pose_array)
            transformed_tcp_matrix = tcp_transform @ pose_homo_matrix
            transformed_9d_pose = homo_matrix_to_pose_9d_batch(transformed_tcp_matrix).squeeze()
            left_robot_tcp_pose_arrays.append(transformed_9d_pose)

        total_count += len(obs_dict["timestamp"])
        episode_ends_arrays.append(total_count)

        gripper_width = obs_dict["left_robot_gripper_width"]
        for i in range(1, len(gripper_width) - 2):
            if abs(gripper_width[i] - gripper_width[i - 1]) > 0.15:
                gripper_width[i] = (gripper_width[i - 1] + gripper_width[i + 2]) / 2
        left_robot_gripper_width_arrays.append(gripper_width)

        while len(left_robot_gripper_width_arrays[-1]) < len(left_robot_tcp_pose_arrays[-1]):
            left_robot_gripper_width_arrays[-1] = np.concatenate(
                [left_robot_gripper_width_arrays[-1], left_robot_gripper_width_arrays[-1][-1][np.newaxis, :]]
            )

        if use_dino:
            processed_images = [center_crop_and_resize_image(img) for img in obs_dict["left_wrist_img"]]
            left_wrist_img_arrays.append(np.array(processed_images))
        else:
            left_wrist_img_arrays.append(np.array(obs_dict["left_wrist_img"]))

        if "right_robot_tcp_pose" in obs_dict:
            for i in range(len(obs_dict["right_robot_tcp_pose"])):
                pose_array = obs_dict["right_robot_tcp_pose"][i][np.newaxis, :]
                pose_homo_matrix = pose_3d_9d_to_homo_matrix_batch(pose_array)
                transformed_tcp_matrix = tcp_transform @ pose_homo_matrix
                transformed_9d_pose = homo_matrix_to_pose_9d_batch(transformed_tcp_matrix).squeeze()
                right_robot_tcp_pose_arrays.append(transformed_9d_pose)

            gripper_width = obs_dict["right_robot_gripper_width"]
            for i in range(1, len(gripper_width) - 2):
                if abs(gripper_width[i] - gripper_width[i - 1]) > 0.15:
                    gripper_width[i] = (gripper_width[i - 1] + gripper_width[i + 2]) / 2
            right_robot_gripper_width_arrays.append(gripper_width)

            while len(right_robot_gripper_width_arrays[-1]) < len(right_robot_tcp_pose_arrays[-1]):
                right_robot_gripper_width_arrays[-1] = np.concatenate(
                    [right_robot_gripper_width_arrays[-1], right_robot_gripper_width_arrays[-1][-1][np.newaxis, :]]
                )

            if use_dino:
                processed_images = [center_crop_and_resize_image(img) for img in obs_dict["right_wrist_img"]]
                right_wrist_img_arrays.append(np.array(processed_images))
            else:
                right_wrist_img_arrays.append(np.array(obs_dict["right_wrist_img"]))

    if not episode_ends_arrays:
        logging.warning("No valid episodes found")
        return save_data_path

    # Convert lists to arrays
    episode_ends_np = np.array(episode_ends_arrays)
    timestamp_np = np.vstack(timestamp_arrays)
    left_wrist_img_np = np.vstack(left_wrist_img_arrays)
    left_robot_tcp_pose_np = np.vstack(left_robot_tcp_pose_arrays)
    left_robot_gripper_width_np = np.vstack(left_robot_gripper_width_arrays)
    left_robot_gripper_width_np = (left_robot_gripper_width_np + gripper_width_bias) * gripper_width_scale

    right_wrist_img_np = None
    right_robot_tcp_pose_np = None
    right_robot_gripper_width_np = None

    if right_wrist_img_arrays:
        right_wrist_img_np = np.vstack(right_wrist_img_arrays)
        right_robot_tcp_pose_np = np.vstack(right_robot_tcp_pose_arrays)
        right_robot_gripper_width_np = np.vstack(right_robot_gripper_width_arrays)
        right_robot_gripper_width_np = (right_robot_gripper_width_np + gripper_width_bias) * gripper_width_scale
    elif is_dual_arm:
        raise ValueError(f"{action_type.name} requires both left and right arm data")

    logging.info(f"episodes: {len(episode_ends_np)}")

    # Temporal downsampling
    if temporal_downsample_ratio > 1:
        (
            timestamp_np,
            left_wrist_img_np,
            left_robot_tcp_pose_np,
            left_robot_gripper_width_np,
            right_wrist_img_np,
            right_robot_tcp_pose_np,
            right_robot_gripper_width_np,
            episode_ends_np,
        ) = _downsample_temporal_data(
            temporal_downsample_ratio,
            timestamp_np,
            episode_ends_np,
            left_wrist_img_np,
            left_robot_tcp_pose_np,
            left_robot_gripper_width_np,
            right_wrist_img_np,
            right_robot_tcp_pose_np,
            right_robot_gripper_width_np,
        )

    # Build state arrays
    if action_type == ActionType.LEFT_ARM_6DOF_GRIPPER_WIDTH:
        state_arrays = np.concatenate([left_robot_tcp_pose_np, left_robot_gripper_width_np], axis=-1)
    elif action_type == ActionType.LEFT_ARM_3D_TRANSLATION_GRIPPER_WIDTH:
        state_arrays = np.concatenate([left_robot_tcp_pose_np[:, :3], left_robot_gripper_width_np], axis=-1)
    elif action_type == ActionType.RIGHT_ARM_6DOF_GRIPPER_WIDTH:
        state_arrays = np.concatenate([right_robot_tcp_pose_np, right_robot_gripper_width_np], axis=-1)
    elif action_type == ActionType.DUAL_ARM_6DOF_GRIPPER_WIDTH:
        state_arrays = np.concatenate(
            [
                left_robot_tcp_pose_np,
                right_robot_tcp_pose_np,
                left_robot_gripper_width_np,
                right_robot_gripper_width_np,
            ],
            axis=-1,
        )
    elif action_type == ActionType.DUAL_ARM_3D_TRANSLATION_GRIPPER_WIDTH:
        state_arrays = np.concatenate(
            [
                left_robot_tcp_pose_np[:, :3],
                right_robot_tcp_pose_np[:, :3],
                left_robot_gripper_width_np,
                right_robot_gripper_width_np,
            ],
            axis=-1,
        )

    # Build action arrays
    if use_absolute_action:
        action_arrays = _create_absolute_actions(state_arrays, episode_ends_np)
    else:
        raise NotImplementedError("Only absolute actions are supported")

    # Create zarr storage
    _create_zarr_storage(
        save_data_path,
        timestamp_np,
        left_robot_tcp_pose_np,
        left_robot_gripper_width_np,
        state_arrays,
        action_arrays,
        episode_ends_np,
        left_wrist_img_np,
        right_robot_tcp_pose_np,
        right_robot_gripper_width_np,
        right_wrist_img_np,
    )

    logging.info(f"Total count after filtering: {action_arrays.shape[0]}")
    logging.info(f"Save data at {save_data_path}")

    return save_data_path


def _downsample_temporal_data(
    downsample_ratio: int,
    timestamp_arrays: np.ndarray,
    episode_ends_arrays: np.ndarray,
    left_wrist_img_arrays: np.ndarray,
    left_robot_tcp_pose_arrays: np.ndarray,
    left_robot_gripper_width_arrays: np.ndarray,
    right_wrist_img_arrays: np.ndarray | None = None,
    right_robot_tcp_pose_arrays: np.ndarray | None = None,
    right_robot_gripper_width_arrays: np.ndarray | None = None,
) -> tuple:
    """Temporal downsampling processing function."""
    keep_indices = []
    current_episode_start = 0

    for episode_end in episode_ends_arrays:
        episode_indices = np.arange(current_episode_start, episode_end)
        if len(episode_indices) > 2:
            middle_indices = episode_indices[1:-1]
            downsampled_middle_indices = middle_indices[::downsample_ratio]
            episode_keep_indices = np.concatenate(
                [[episode_indices[0]], downsampled_middle_indices, [episode_indices[-1]]]
            )
        else:
            episode_keep_indices = episode_indices
        keep_indices.extend(episode_keep_indices)
        current_episode_start = episode_end

    keep_indices = np.array(keep_indices)

    timestamp_arrays = timestamp_arrays[keep_indices]
    left_wrist_img_arrays = left_wrist_img_arrays[keep_indices]
    left_robot_tcp_pose_arrays = left_robot_tcp_pose_arrays[keep_indices]
    left_robot_gripper_width_arrays = left_robot_gripper_width_arrays[keep_indices]

    if right_wrist_img_arrays is not None and len(right_wrist_img_arrays) > 0:
        right_wrist_img_arrays = right_wrist_img_arrays[keep_indices]
    if right_robot_tcp_pose_arrays is not None and len(right_robot_tcp_pose_arrays) > 0:
        right_robot_tcp_pose_arrays = right_robot_tcp_pose_arrays[keep_indices]
    if right_robot_gripper_width_arrays is not None and len(right_robot_gripper_width_arrays) > 0:
        right_robot_gripper_width_arrays = right_robot_gripper_width_arrays[keep_indices]

    # Recalculate episode_ends
    new_episode_ends = []
    count = 0
    current_episode_start = 0

    for episode_end in episode_ends_arrays:
        episode_indices = np.arange(current_episode_start, episode_end)
        if len(episode_indices) > 2:
            middle_indices = episode_indices[1:-1]
            downsampled_middle_indices = middle_indices[::downsample_ratio]
            count += len(downsampled_middle_indices) + 2
        else:
            count += len(episode_indices)
        new_episode_ends.append(count)
        current_episode_start = episode_end

    return (
        timestamp_arrays,
        left_wrist_img_arrays,
        left_robot_tcp_pose_arrays,
        left_robot_gripper_width_arrays,
        right_wrist_img_arrays,
        right_robot_tcp_pose_arrays,
        right_robot_gripper_width_arrays,
        np.array(new_episode_ends),
    )


def _create_absolute_actions(state_arrays: np.ndarray, episode_ends_arrays: np.ndarray) -> np.ndarray:
    """Create absolute action arrays."""
    new_action_arrays = state_arrays[1:, ...].copy()
    action_arrays = np.concatenate([new_action_arrays, new_action_arrays[-1][np.newaxis, :]], axis=0)

    for i in range(len(episode_ends_arrays)):
        action_arrays[episode_ends_arrays[i] - 1] = action_arrays[episode_ends_arrays[i] - 2]

    return action_arrays


def _create_zarr_storage(
    save_data_path: str,
    timestamp_arrays: np.ndarray,
    left_robot_tcp_pose_arrays: np.ndarray,
    left_robot_gripper_width_arrays: np.ndarray,
    state_arrays: np.ndarray,
    action_arrays: np.ndarray,
    episode_ends_arrays: np.ndarray,
    left_wrist_img_arrays: np.ndarray,
    right_robot_tcp_pose_arrays: np.ndarray | None = None,
    right_robot_gripper_width_arrays: np.ndarray | None = None,
    right_wrist_img_arrays: np.ndarray | None = None,
) -> None:
    """Create zarr storage."""
    zarr_root = zarr.group(save_data_path)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    wrist_img_chunk_size = (100, *left_wrist_img_arrays.shape[1:])
    action_chunk_size = (10000, action_arrays.shape[1])
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    zarr_data.create_dataset(
        "timestamp", data=timestamp_arrays, chunks=(10000,), dtype="float32", overwrite=True, compressor=compressor
    )
    zarr_data.create_dataset(
        "left_robot_tcp_pose",
        data=left_robot_tcp_pose_arrays,
        chunks=(10000, 9),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "left_robot_gripper_width",
        data=left_robot_gripper_width_arrays,
        chunks=(10000, 1),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "target", data=state_arrays, chunks=action_chunk_size, dtype="float32", overwrite=True, compressor=compressor
    )
    zarr_data.create_dataset(
        "action", data=action_arrays, chunks=action_chunk_size, dtype="float32", overwrite=True, compressor=compressor
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        chunks=(10000,),
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset("left_wrist_img", data=left_wrist_img_arrays, chunks=wrist_img_chunk_size, dtype="uint8")

    if right_robot_tcp_pose_arrays is not None and len(right_robot_tcp_pose_arrays) > 0:
        zarr_data.create_dataset(
            "right_robot_tcp_pose",
            data=right_robot_tcp_pose_arrays,
            chunks=(10000, 9),
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
    if right_robot_gripper_width_arrays is not None and len(right_robot_gripper_width_arrays) > 0:
        zarr_data.create_dataset(
            "right_robot_gripper_width",
            data=right_robot_gripper_width_arrays,
            chunks=(10000, 1),
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
    if right_wrist_img_arrays is not None and len(right_wrist_img_arrays) > 0:
        zarr_data.create_dataset(
            "right_wrist_img", data=right_wrist_img_arrays, chunks=wrist_img_chunk_size, dtype="uint8"
        )
