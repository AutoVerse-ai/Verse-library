import ray
import time
import itertools

ray.init()

lens=1000000
numbers = list(range(100000))


def double(numbers):
    for i in range(len(numbers)):
        numbers[i]=2*numbers[i]
    return numbers


start_time = time.time()
serial_doubled_numbers = double(numbers)
end_time = time.time()
ser_time = end_time - start_time
print(f"Ordinary funciton call takes {end_time - start_time} seconds")
# Ordinary funciton call takes 0.16506004333496094 seconds


@ray.remote
def remote_double(number):
    for i in range(len(numbers)):
        numbers[i]=2*numbers[i]
    return numbers


start_time = time.time()
doubled_number_refs = [remote_double.remote(numbers)]
parallel_doubled_numbers = ray.get(doubled_number_refs)
end_time = time.time()
par_time = end_time - start_time
print(f"Parallelizing tasks takes {end_time - start_time} seconds")
# Parallelizing tasks takes 1.6061789989471436 seconds
print(f"time diff: {par_time - ser_time}, {par_time / ser_time}")