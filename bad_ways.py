import asyncio
import traceback
import numpy as np
import time
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

# bad1: view
try:
    arr_f.view(np.uint8)
except Exception as e:
    traceback.print_exception(e)


# bad2: convert to C
def try_convert():
    return np.ascontiguousarray(arr_f)


async def with_timeout():
    f_convert = asyncio.get_running_loop().run_in_executor(None, try_convert)
    # let's wait for 5 seconds
    await asyncio.sleep(5)
    if not f_convert.done():
        print("Still not done after 5s!")
        return None
    else:
        return await f_convert


convert_res = asyncio.run(with_timeout())
if convert_res is None:
    print("Convert to C format timeout.")


# bad3: memory view and cast
try:
    arr_f.data.cast("B")
except Exception as e:
    traceback.print_exception(e)