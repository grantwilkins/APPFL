from . import pysz
from ..config import Config
from typing import Tuple, Any
import numpy as np
import pickle


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
        self.compressor_class = None

    def compress(self, ori_data: np.ndarray):
        """
        Compress data with chosen compressor
        :param ori_data: compressed data, numpy array format
        :return: decompressed data,numpy array format
        """
        if self.cfg.compressor == "SZ3" or self.cfg.compressor == "SZ2":
            self.cfg.flat_model_size = ori_data.shape
            if self.compressor_class is None:
                self.compressor_class = pysz.SZ(szpath=self.cfg.compressor_lib_path)
            error_mode = self.sz_error_mode_dict[self.cfg.compressor_error_mode]
            error_bound = self.cfg.compressor_error_bound
            compressed_arr, comp_ratio = self.compressor_class.compress(
                data=ori_data,
                eb_mode=error_mode,
                eb_abs=error_bound,
                eb_rel=error_bound,
                eb_pwr=error_bound,
            )
            return compressed_arr.tobytes()
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

    def compress_error_control(
        self, ori_data: np.ndarray, error_bound: float, error_mode: str
    ):
        """
        Compress data with chosen compressor
        :param ori_data: compressed data, numpy array format
        :return: decompressed data,numpy array format
        """
        if self.cfg.compressor == "SZ3" or "SZ2":
            self.cfg.flat_model_size = ori_data.shape
            if self.compressor_class is None:
                self.compressor_class = pysz.SZ(szpath=self.cfg.compressor_lib_path)
            error_mode = self.sz_error_mode_dict[error_mode]
            compressed_arr, comp_ratio = self.compressor_class.compress(
                data=ori_data,
                eb_mode=error_mode,
                eb_abs=error_bound,
                eb_rel=error_bound,
                eb_pwr=error_bound,
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
            if self.compressor_class is None:
                self.compressor_class = pysz.SZ(szpath=self.cfg.compressor_lib_path)
            cmp_data = np.frombuffer(cmp_data, dtype=np.uint8)
            decompressed_arr = self.compressor_class.decompress(
                data_cmpr=cmp_data,
                original_shape=ori_shape,
                original_dtype=ori_dtype,
            )
            return decompressed_arr
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
            if self.compressor_class is None:
                self.compressor_class = pysz.SZ(szpath=self.cfg.compressor_lib_path)
            return self.compressor_class.verify(ori_data, dec_data)
        else:
            raise NotImplementedError
