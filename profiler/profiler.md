# Profiling
Initialized by the `.profile()` method of the provided `HyCache` class. Profiling is done for selection of cache steps on the memory, on the disk, and to scale the number of fetchers (Python processes used for fetching data from disk).

## Intermediate steps: Compute and Memory

To determine the *optimal* caching step(s), there needs to be some profiling that provides data about each step, namely, the compute saving per step if it's chosen to cache and the size of the materialized results to be cached. The dataset subset that is used for profiling would vary given the time constraint. The collected data is a python dictionary, for example:

```json
{
    "0": [0.47, 0.42],
    "1": [0.58, 1.92],
    "2": [0.28, 4.63],
    "3": [0.57, 5.12]

}
```
The key represents a **caching step** and the value is a tuple. The first value of the tuple is the average **size** of a tensor in that step, denoted in MBs. The second value denotes the average **compute saving** per tensor in that step in milliseconds.

The amount of time this profiling would take depends on the pipeline, dataset size, and the time_limit (if provided).

> A similar profiling is done for the disk, the sole difference being that the caching store is the disk instead of the memory, and the **compute savings** per tensor would now be different because of the longer access latency of disk-fetch.

For these two tasks, an internal function `profile_device()` is invoked. 

Additionally, a function `profile_steps()` is also made available to the users, which captures the total time it takes to solely preprocess a single step, along with the size of the tensors produced.  `profile_steps()` can be optionally used by the user to understand pipeline characteristics further, or to understand the caching decisions of the ILP in detail.