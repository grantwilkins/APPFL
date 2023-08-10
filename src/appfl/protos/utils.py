from .federated_learning_pb2 import DataBuffer
from .federated_learning_pb2 import TensorRecord
from appfl.misc.utils import flatten_primal_or_dual
from appfl.compressor import Compressor
from collections import OrderedDict


def construct_tensor_record(name, nparray):
    return TensorRecord(
        name=name,
        data_shape=list(nparray.shape),
        data_bytes=nparray.tobytes(order="C"),
        data_dtype="np." + str(nparray.dtype),
    )


def construct_compressed_tensor_record(name, primal_or_dual, cfg):
    return TensorRecord(
        name=name,
        data_shape=(1,),
        data_bytes=primal_or_dual,
        data_dtype="np.float32",
    )


def proto_to_databuffer(proto, max_message_size=(2 * 1024 * 1024)):
    data_bytes = proto.SerializeToString()
    data_bytes_size = len(data_bytes)
    message_size = (
        data_bytes_size if max_message_size > data_bytes_size else max_message_size
    )

    for i in range(0, data_bytes_size, message_size):
        chunk = data_bytes[i : i + message_size]
        msg = DataBuffer(size=message_size, data_bytes=chunk)
        yield msg
