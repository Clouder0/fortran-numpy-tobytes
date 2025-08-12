import numpy as np
import time
import ctypes
import contextlib


@contextlib.contextmanager
def record_time(msg: str):
    start = time.time()

    yield
    end = time.time()
    print(f"{msg}: Elapsed time: {end - start} seconds")


with record_time("init"):
    arr_f = np.ones((1024, 1024, 1024), dtype=np.float32, order="F")

test_sum = np.sum(arr_f)
print("arr fortran sum", test_sum)

size = arr_f.nbytes
addr = arr_f.ctypes.data

with record_time("create view"):
    byte_buffer = (ctypes.c_uint8 * size).from_address(addr)
    view = memoryview(byte_buffer)

# verify zero copy by modifying the underlying array
print(view[0:8].hex())
arr_f[0][0][0] = 0.0
print(view[0:8].hex())

print(type(view))

# now let's try split chunks, and combine back
chunk_num = 8
chunk_size = len(byte_buffer) // chunk_num
lhs = 0
rhs = chunk_size
cur_chunks: list[memoryview] = []
while lhs < len(byte_buffer):
    cur_chunks.append(view[lhs:rhs])
    lhs = rhs
    rhs = min(rhs + chunk_size, len(byte_buffer))

# now let's simulate receiver
with record_time("recv buf init"):
    recv_buf = np.empty(arr_f.shape, arr_f.dtype, order="F")
    recv_buf_ptr = recv_buf.ctypes.data
lhs = 0
rhs = chunk_size
with record_time("recv copy"):
    while lhs < len(byte_buffer):
        ctypes.memmove(recv_buf_ptr + lhs, addr + lhs, rhs - lhs)
        lhs = rhs
        rhs = min(rhs + chunk_size, len(byte_buffer))
loss = np.sum((recv_buf - arr_f) ** 2)
print("total loss", loss)
