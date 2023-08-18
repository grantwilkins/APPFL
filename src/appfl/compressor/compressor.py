from collections import OrderedDict
from copy import deepcopy
import zlib
from . import pysz
from ..config import Config
from typing import Tuple, Any
import numpy as np
import pickle
from . import pyszx
import zfpy
import scipy.sparse as sparse
import zstd
import torch.nn as nn
import torch
import xz
import gzip
import blosc


class Compressor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sz_error_mode_dict = {
            "ABS": 0,
            "REL": 1,
            "ABS_AND_REL": 2,
            "ABS_OR_REL": 3,
            "PSNR": 4,
            "NORM": 5,
            "PW_REL": 10,
        }
        self.lossless_compressor = cfg.lossless_compressor
        self.compression_layers = []

    def compress(self, ori_data: np.ndarray):
        """
        Compress data with chosen compressor
        :param ori_data: compressed data, numpy array format
        :return: decompressed data,numpy array format
        """
        if self.cfg.compressor == "SZ3" or self.cfg.compressor == "SZ2":
            self.cfg.flat_model_size = ori_data.shape
            compressor = pysz.SZ(szpath=self.cfg.compressor_lib_path)
            error_mode = self.sz_error_mode_dict[self.cfg.compressor_error_mode]
            error_bound = self.cfg.compressor_error_bound
            compressed_arr, comp_ratio = compressor.compress(
                data=ori_data,
                eb_mode=error_mode,
                eb_abs=error_bound,
                eb_rel=error_bound,
                eb_pwr=error_bound,
            )
            return compressed_arr.tobytes()
        elif self.cfg.compressor == "SZx":
            self.cfg.flat_model_size = ori_data.shape
            compressor = pyszx.SZx(szxpath=self.cfg.compressor_lib_path)
            error_mode = self.sz_error_mode_dict[self.cfg.compressor_error_mode]
            error_bound = self.cfg.compressor_error_bound
            compressed_arr, comp_ratio = compressor.compress(
                data=ori_data,
                eb_mode=error_mode,
                eb_abs=error_bound,
                eb_rel=error_bound,
            )
            return compressed_arr.tobytes()
        elif self.cfg.compressor == "ZFP":
            if self.cfg.compressor_error_mode == "ABS":
                return zfpy.compress_numpy(
                    ori_data, tolerance=self.cfg.compressor_error_bound
                )
            elif self.cfg.compressor_error_mode == "REL":
                range_data = abs(np.max(ori_data) - np.min(ori_data))
                return zfpy.compress_numpy(
                    ori_data, tolerance=self.cfg.compressor_error_bound * range_data
                )
            else:
                raise NotImplementedError
        elif self.cfg.compressor == "Prune":
            k = len(ori_data)
            data = np.zeros(k, dtype="float32")
            index = np.zeros(k, dtype="uint8")

            k = 0
            kk = 0
            bit = 0

            for i in range(len(ori_data)):
                if (bit == 255) and (ori_data[i] == 0):
                    index[k] = 0
                    k = k + 1
                    bit = 0
                if ori_data[i] != 0:
                    data[kk] = ori_data[i]
                    index[k] = bit
                    k = k + 1
                    kk = kk + 1
                    bit = 0
                bit = bit + 1
            a = np.zeros(kk, dtype="float32")
            b = np.zeros(k, dtype="uint8")
            for i in range(kk):
                a[i] = data[i]
            for i in range(k):
                b[i] = index[i]
            dict_weights = {"weights": a, "idx": b}
            return pickle.dumps(dict_weights)
        else:
            raise NotImplementedError

    def compress_model(
        self, model: nn.Module, param_count_threshold: int
    ) -> (bytes, int):
        compressed_weights = {}
        lossy_compressed_size = 0
        lossy_original_size = 0
        lossy_elements = 0
        lossless_compressed_size = 0
        lossless_original_size = 0
        for name, param in model.state_dict().items():
            param_flat = param.flatten().detach().cpu().numpy()
            if "weight" in name and param_flat.size > param_count_threshold:
                lossy_original_size += param_flat.size * 4
                lossy_elements += param_flat.size
                if self.cfg.pruning:
                    sparse_arr = sparse.coo_matrix(param_flat.reshape(1, -1))
                    sparse_data = sparse_arr.data
                    sparse_row = sparse_arr.row
                    sparse_col = sparse_arr.col
                    if sparse_data.size != 0:
                        compressed_weights[name] = self.compress(ori_data=sparse_data)
                        compressed_weights[name + "_row"] = zstd.compress(
                            sparse_row, 10
                        )
                        compressed_weights[name + "_col"] = zstd.compress(
                            sparse_col, 10
                        )
                        compressed_weights[name + "_shape"] = sparse_data.shape
                    lossy_compressed_size += (
                        len(compressed_weights[name])
                        + len(compressed_weights[name + "_row"])
                        + len(compressed_weights[name + "_col"])
                        + len(compressed_weights[name + "_shape"])
                    )
                else:
                    compressed_weights[name] = self.compress(ori_data=param_flat)
                    lossy_compressed_size += len(compressed_weights[name])
            else:
                lossless_original_size += param_flat.size * 4
                lossless = b""
                if self.lossless_compressor == "zstd":
                    lossless = zstd.compress(param_flat, 10)
                elif self.lossless_compressor == "xz":
                    lossless = xz.compress(param_flat.tobytes())
                elif self.lossless_compressor == "gzip":
                    lossless = gzip.compress(param_flat.tobytes())
                elif self.lossless_compressor == "zlib":
                    lossless = zlib.compress(param_flat.tobytes())
                elif self.lossless_compressor == "blosc":
                    lossless = blosc.compress(param_flat.tobytes(), typesize=4)
                else:
                    raise NotImplementedError
                lossless_compressed_size += len(lossless)
                compressed_weights[name] = lossless
        if lossy_compressed_size != 0:
            print(
                "Lossy Compression Ratio: "
                + str(lossy_original_size / lossy_compressed_size)
            )
        print(
            "Total Compression Ratio: "
            + str(
                (lossy_original_size + lossless_original_size)
                / (lossy_compressed_size + lossless_compressed_size)
            )
        )
        print(
            "Lossless Compression Ratio: "
            + str(lossless_original_size / lossless_compressed_size)
        )
        return (
            pickle.dumps(compressed_weights),
            lossy_elements,
        )

    def decompress_model(
        self, compressed_model: bytes, model: nn.Module, param_count_threshold
    ) -> nn.Module:
        model_copy = deepcopy(model)
        new_dict = OrderedDict()
        decomp_weights = pickle.loads(compressed_model)
        for name, param in model_copy.state_dict().items():
            if "weight" in name and param.numel() > param_count_threshold:
                if self.cfg.pruning:
                    shape = decomp_weights[name + "_shape"]
                    decomp_weights[name] = self.decompress(
                        cmp_data=decomp_weights[name],
                        ori_shape=shape,
                        ori_dtype=np.float32,
                    )
                    decomp_weights[name + "_row"] = np.frombuffer(
                        zstd.decompress(decomp_weights[name + "_row"]), dtype=np.int32
                    )
                    decomp_weights[name + "_col"] = np.frombuffer(
                        zstd.decompress(decomp_weights[name + "_col"]), dtype=np.int32
                    )
                    csr_arr = (
                        sparse.coo_matrix(
                            (
                                decomp_weights[name],
                                (
                                    decomp_weights[name + "_row"],
                                    decomp_weights[name + "_col"],
                                ),
                            ),
                            shape=(1, param.numel()),
                        )
                        .toarray()
                        .reshape(-1)
                    ).astype(np.float32)
                    decomp_weights[name] = csr_arr
                else:
                    decomp_weights[name] = self.decompress(
                        cmp_data=decomp_weights[name],
                        ori_shape=param.shape,
                        ori_dtype=np.float32,
                    )
            else:
                if self.lossless_compressor == "zstd":
                    decomp_weights[name] = zstd.decompress(decomp_weights[name])
                elif self.lossless_compressor == "xz":
                    decomp_weights[name] = xz.decompress(decomp_weights[name])
                elif self.lossless_compressor == "gzip":
                    decomp_weights[name] = gzip.decompress(decomp_weights[name])
                elif self.lossless_compressor == "zlib":
                    decomp_weights[name] = zlib.decompress(decomp_weights[name])
                elif self.lossless_compressor == "blosc":
                    decomp_weights[name] = blosc.decompress(
                        decomp_weights[name], as_bytearray=True
                    )
                else:
                    raise NotImplementedError
                decomp_weights[name] = np.frombuffer(
                    decomp_weights[name], dtype=np.float32
                )
            if param.shape == torch.Size([]):
                copy_arr = deepcopy(decomp_weights[name])
                copy_tensor = torch.from_numpy(copy_arr)
                new_dict[name] = torch.tensor(copy_tensor)
            else:
                copy_arr = deepcopy(decomp_weights[name])
                copy_tensor = torch.from_numpy(copy_arr)
                new_dict[name] = copy_tensor.reshape(param.shape)
        model_copy.load_state_dict(new_dict)
        return model_copy

    def compress_error_control(
        self, ori_data: np.ndarray, error_bound: float, error_mode: str
    ):
        """
        Compress data with chosen compressor
        :param ori_data: compressed data, numpy array format
        :return: decompressed data,numpy array format
        """
        if self.cfg.compressor == "SZ3" or self.cfg.compressor == "SZ2":
            self.cfg.flat_model_size = ori_data.shape
            compressor = pysz.SZ(szpath=self.cfg.compressor_lib_path)
            error_mode = self.sz_error_mode_dict[error_mode]
            compressed_arr, comp_ratio = compressor.compress(
                data=ori_data,
                eb_mode=error_mode,
                eb_abs=error_bound,
                eb_rel=error_bound,
                eb_pwr=error_bound,
            )
            return compressed_arr.tobytes()
        elif self.cfg.compressor == "SZx":
            self.cfg.flat_model_size = ori_data.shape
            compressor = pyszx.SZx(szxpath=self.cfg.compressor_lib_path)
            error_mode = self.sz_error_mode_dict[error_mode]
            compressed_arr, comp_ratio = compressor.compress(
                data=ori_data,
                eb_mode=error_mode,
                eb_abs=error_bound,
                eb_rel=error_bound,
            )
            return compressed_arr.tobytes()
        else:
            raise NotImplementedError

    def decompress(
        self, cmp_data, ori_shape: Tuple[int, ...], ori_dtype: np.dtype
    ) -> np.ndarray:
        """
        Decompress data with chosen compressor
        :param cmp_data: compressed data, numpy array format, dtype should be np.uint8
        :param ori_shape: the shape of original data
        :param ori_dtype: the dtype of original data
        :return: decompressed data,numpy array format
        """
        if self.cfg.compressor == "SZ3" or self.cfg.compressor == "SZ2":
            compressor = pysz.SZ(szpath=self.cfg.compressor_lib_path)
            cmp_data = np.frombuffer(cmp_data, dtype=np.uint8)
            decompressed_arr = compressor.decompress(
                data_cmpr=cmp_data,
                original_shape=ori_shape,
                original_dtype=ori_dtype,
            )
            return decompressed_arr
        elif self.cfg.compressor == "SZx":
            compressor = pyszx.SZx(szxpath=self.cfg.compressor_lib_path)
            cmp_data = np.frombuffer(cmp_data, dtype=np.uint8)
            decompressed_arr = compressor.decompress(
                data_cmpr=cmp_data,
                original_shape=ori_shape,
                original_dtype=ori_dtype,
            )
            return decompressed_arr
        elif self.cfg.compressor == "ZFP":
            return zfpy.decompress_numpy(cmp_data)
        elif self.cfg.compressor == "Prune":
            data_dict = pickle.loads(cmp_data)
            index = data_dict["idx"]
            weights = data_dict["weights"]
            data = np.zeros(ori_shape, dtype=ori_dtype)
            k = 0
            q = 0
            for j in range(len(index)):
                if (index[j] == 0) and (j != 0):
                    k = k + 255
                else:
                    k = k + index[j]
                    data[k] = weights[q]
                    q = q + 1
            return data
        else:
            raise NotImplementedError

    def verify(self, ori_data, dec_data) -> Tuple[float, ...]:
        if self.cfg.compressor == "SZ3" or "SZ2":
            return pysz.SZ(szpath=self.cfg.compressor_lib_path).verify(
                ori_data, dec_data
            )
        else:
            raise NotImplementedError
