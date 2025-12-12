import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset


def relative_T_lidar(T_world_lidar):
    T_world_lidar = np.asarray(T_world_lidar)
    assert T_world_lidar.ndim == 3 and T_world_lidar.shape[1:] == (4, 4)
    T_rel = []
    T0 = T_world_lidar[0]
    for t in range(T_world_lidar.shape[0]):
        Tt = T_world_lidar[t]
        T = np.linalg.inv(Tt) @ T0
        T_rel.append(T.astype(np.float32))
    return np.stack(T_rel, axis=0)


def collate_seq(batch):
    if len(batch) != 1:
        raise RuntimeError("Use batch_size=1 with collate_seq for now")
    return batch[0]


class NuScenesSeqDataset(Dataset):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        training,
        logger,
        seq_len,
        stride,
        nusc_version,
        nusc_dataroot,
        root_path=None,
    ):
        super().__init__()
        self.base = NuScenesDataset(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.seq_len = int(seq_len)
        self.stride = int(stride)

        self.class_names = self.base.class_names

        dataroot_for_nusc = Path(nusc_dataroot) / nusc_version

        self.nusc = NuScenes(
            version=nusc_version,
            dataroot=str(dataroot_for_nusc),
            verbose=False,
        )

        self.sequence_indices = self._build_sequences()

    def _build_sequences(self):
        by_scene = {}
        for idx, info in enumerate(self.base.infos):
            sample_token = info.get("token", None)
            if sample_token is None:
                raise KeyError("token not found in info; cannot derive scene_id")
            sample = self.nusc.get("sample", sample_token)
            scene_id = sample["scene_token"]
            by_scene.setdefault(scene_id, []).append(idx)

        sequences = []
        for scene_id, idxs in by_scene.items():
            idxs_sorted = sorted(
                idxs,
                key=lambda i: self.base.infos[i]["timestamp"],
            )
            L = len(idxs_sorted)
            if L < self.seq_len:
                continue
            for start in range(0, L - self.seq_len + 1, self.stride):
                seq = idxs_sorted[start:start + self.seq_len]
                sequences.append(seq)

                # if len(sequences) >= 100:
                #     return sequences
        return sequences

    def __len__(self):
        return len(self.sequence_indices)

    def _world_T_lidar_from_info(self, info):
        sample = self.nusc.get("sample", info["token"])
        sd_lidar = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ep = self.nusc.get("ego_pose", sd_lidar["ego_pose_token"])
        cs = self.nusc.get("calibrated_sensor", sd_lidar["calibrated_sensor_token"])
        world_T_ego = transform_matrix(ep["translation"], Quaternion(ep["rotation"]), inverse=False)
        ego_T_lidar = transform_matrix(cs["translation"], Quaternion(cs["rotation"]), inverse=False)
        return world_T_ego @ ego_T_lidar


    def __getitem__(self, index):
        idx_seq = self.sequence_indices[index]
        infos_seq = [self.base.infos[i] for i in idx_seq]

        T_world_lidar = []
        sample_tokens = []
        timestamps = []

        for info in infos_seq:
            T = self._world_T_lidar_from_info(info)
            T_world_lidar.append(T.astype(np.float32))
            sample_tokens.append(info["token"])
            timestamps.append(info["timestamp"])

        T_world_lidar = np.stack(T_world_lidar, axis=0)

        frames_raw = [self.base.__getitem__(i) for i in idx_seq]
        frames = [self.base.collate_batch([f]) for f in frames_raw]


        sample = {
            "frames": frames,
            "T_world_lidar": T_world_lidar,
            "sample_tokens": sample_tokens,
            "timestamps": np.array(timestamps, dtype=np.int64),
        }
        return sample

