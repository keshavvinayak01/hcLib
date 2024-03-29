import shutil
import os
import re
import numpy as np
from typing import Any, Optional
from dataclasses import dataclass
from nvidia.dali.pipeline import Pipeline
from pipegen import create_pipe, merge_and_collect_pipes
from annotate import get_annotated_computational_graph, filter_important_steps
from profiler import profile_device, profile_steps
from memory_estimator import adjust_for_activemem
from iterator import HyCacheGenericIterator
from math import ceil
from utils import get_mem_size, get_presto_solution
from solver import get_cache_solution
import time

"""
# 1. Profile the pipe to get memory statistics
# 2. Profile the pipe to get disk_statistics
# 2.5 Get the maximum workers required for throughput saturation + memory used by them
# 2.5.5 Calculate the memory required for these many workers, and adjust the max possible memory size.    
# 3. Get ILP solution for both disk and memory
# 4. Pipeline merge should be enabled by default
# 5. Create all pipelines as per available solutions
# 6. get_iterator() function should return the HyCacheGenericIterator
"""

class BasePipeline(Pipeline):
    """
    For every preprocessing step defined, 
    use self.condition(self.profile_helpers, step_number),
    where step_number is unique to each pre-processing pipeline.
    
    """
    def __init__(
        self, batch_size, num_threads, num_workers, device_id, seed,prefetch_queue_depth=2):
        super(BasePipeline, self).__init__(
            batch_size,
            num_threads,
            device_id,
            seed=seed,
            prefetch_queue_depth=prefetch_queue_depth,
            py_start_method='spawn',
            py_num_workers=num_workers,
            enable_memory_stats=False,
        )

    def set_required_params(self, input_func, condition, profile_helpers, samples=[], label_func=None, args=None):
        # print(input_func, condition, profile_helpers, label_func)
        self.input = input_func(samples, self.max_batch_size, label_func=label_func, args=args)
        self.condition = condition
        self.profile_helpers = profile_helpers

    def define_graph(self):
        """Left to the user for inmplementation"""
        raise NotImplementedError

@dataclass
class CacheConfig:
    memcache_size: int
    diskcache_size: int
    cache_steps: int
    dataset: list
    disk_cloc: str = os.getcwd() + "/disk_cache/"
    label_func: Optional[Any] = None
    
    def __post_init__(self):
        if not isinstance(self.memcache_size, int) or self.memcache_size <= 0:
            raise ValueError("memcache_size must be a positive integer")
        if not isinstance(self.diskcache_size, int) or self.diskcache_size <= 0:
            raise ValueError("diskcache_size must be a positive integer")
        if self.disk_cloc == "":
            raise ResourceWarning("Using current disk location for cache")
        if not isinstance(self.cache_steps, int) or self.cache_steps <= 0:
            raise ValueError("cache_steps must be a positive integer")
        if not isinstance(self.dataset, list) or not self.dataset:
            raise ValueError("Dataset should be a non-empty list of files")
        if not self.label_func:
            raise UserWarning("Using no labels or random values for labels")

class HyCache:
    """
    Takes in a defined and annotated pipeline as an input, and internally 
    creates optimized pipelines which can be invoked for iteration.
    """
    def __init__(self,
                 pipe: Pipeline,
                 dataset: list,
                 memory_budget: int,
                 disk_budget: int,
                 cache_steps: int,
                 disk_cloc: str = os.getcwd() + "/disk_cache/",
                 logdir: str = "./",
                 batch_size: int = 64,
                 profile_factor: float = 0.01,
                 threads: int = 32,
                 rank: int = 0,
                 world_size: int = 1,
                 directio:bool = 0,
                 merge:bool = 1,
                 label_func = None,
                 ):
        assert (profile_factor <= 1 and profile_factor > 0), "profile_factor has to be in (0, 1]"
        self.config = CacheConfig(
            memcache_size=memory_budget,
            diskcache_size=disk_budget,
            cache_steps=cache_steps,
            dataset=dataset,
            label_func=label_func,
            disk_cloc=disk_cloc
        )

        ####################### Implementation:  1. Annotate #######################
        self.pipe=get_annotated_computational_graph(pipe)
        ####################### Implementation:  1. Annotate #######################
        self.built_pipes = []
        self.pipe_name = re.sub(r'[^a-zA-Z]', '', str(pipe).split('.')[1].split(' ')[0])
        self.logdir = os.path.join(logdir, f'logs/{self.pipe_name}')
        disk_cloc += self.pipe_name
        self.args = {
            "pipe": self.pipe,
            "batch_size": batch_size,
            "directio": 1,
            "logdir": os.path.join(logdir, f'logs/{self.pipe_name}'),
            "profile_factor": profile_factor,
            "threads": threads,
            "disk_csize": 0,
            "disk_cstep": -1,
            "cache_size": 0,
            "cache_step": -1,
            "num_workers": 0,
            "num_cworkers": 0,
            "pagecache_size": 0,
            "directio": directio,
            "rank": rank,
            "merge": merge,
            "world_size": world_size
        }
        ####################### Implementation:  2. Step Filter #######################
        self.args["cc_steps"], self.args["bloatups"] = \
            filter_important_steps(self.args, self.config.dataset)
        ####################### Implementation:  2. Step Filter #######################
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        if not os.path.exists(disk_cloc):
            os.makedirs(disk_cloc)

    def build(self, pipetype='cached', same_step=False):
        self.args['experiment'] = pipetype
        profile_begin = time.time()
        if pipetype.lower() == 'presto':
            # create pipe
            disk_profile = profile_device(self.args.copy(), self.config.dataset, 'disk')
            self.config.diskcache_size = self.config.diskcache_size // self.args['world_size']
            disk_cstep = get_presto_solution(self.args.copy(), disk_profile, len(self.config.dataset))
            if disk_cstep == -2:
                pipetype = 'vanilla'
            else:
                self.args['num_cworkers'] = self.args['threads']
                self.args['file_map'] = {file: i for i, file in enumerate(self.config.dataset)}
                self.args['idx_map'] = {i: file for i, file in enumerate(self.config.dataset)}
                self.args['global_cache_map'] = {i: 0 for i in range(len(self.config.dataset))}
                pipe, uncached_samples = \
                    create_pipe(self.args, self.config.dataset, disk_cstep, disk_csize=self.config.diskcache_size, pipetype='cached')
                self.built_pipes.append(pipe)
        if pipetype == 'vanilla':
            self.args['pagecache_size'] = self.config.memcache_size
            self.args['num_workers'] = self.args['threads']
            pipe, _ = create_pipe(self.args.copy(), self.config.dataset, pipetype='vanilla')
            self.built_pipes.append(pipe)
        elif pipetype == 'cached':
            ####################### Implementation:  3. Profiling #######################
            print('*'*30, "Profiling devices", '*'*30)
            # Profiling to get memory, disk, workers, and `profile_steps` statistics
            mem_profile = profile_device(self.args.copy(), self.config.dataset, 'memory')
            disk_profile = profile_device(self.args.copy(), self.config.dataset, 'disk')
            ####################### Implementation:  3. Profiling #######################
            ########################################### Track the global cache map ###########################################
            self.args['file_map'] = {file: i for i, file in enumerate(self.config.dataset)}
            self.args['idx_map'] = {i: file for i, file in enumerate(self.config.dataset)}
            self.args['global_cache_map'] = {i: 0 for i in range(len(self.config.dataset))}

            """Assume initially :
            - cache_size = mem_budget - 2 x Batch memsize - 16 x (workers overhead)
            - Get the mem and disk solution using this size
            - Merge the solutions if same steps in both disk and mem
            - Calculate worker assignment based on disk solution
            - Update cache size:
                cache_size = cache_size + (16 - Total workers used) x (Worker overhead)
            """
            ####################### Implementation:  4. ILP Solver #######################

            ######################################### Finding a Memory Cache solution #########################################
            memcache_steps, memcache_sizes = [-1],[0]
            cache_size = min(get_mem_size('free')*1024, self.config.memcache_size*1024)
            remaining_samples = len(self.config.dataset)
            if self.config.memcache_size > 0:
                mem_profile.pop(-1)
                memcache_sizes, memcache_steps, memcached_tensors = \
                    get_cache_solution(self.args['world_size']*len(self.config.dataset), mem_profile, cache_size/1024)
                remaining_samples -= memcached_tensors
            memcache_sizes = memcache_sizes[::-1]
            memcache_steps = memcache_steps[::-1]
            ######################################### Finding a Disk Cache solution #########################################
            disk_csteps, disk_csizes = [-1],[0]
            if(remaining_samples > 0) and self.config.diskcache_size > 0:
                raw_size = np.median([os.path.getsize(sample) for sample in self.config.dataset]) / (1 << 20)
                # Considering raw data to be cached as well.
                disk_profile[-1][0] = raw_size
                disk_csizes, disk_csteps, diskcached_tensors = \
                    get_cache_solution(self.args['world_size']*remaining_samples, disk_profile, self.config.diskcache_size, tight=False)
                if -1 in disk_csteps:
                    disk_csteps = disk_csteps[1:]
                    disk_csizes = disk_csizes[1:]
            if same_step:
                disk_csteps = memcache_steps
                disk_csizes = [(size/sum(memcache_sizes))*self.config.diskcache_size for size in memcache_sizes]
            print(f"(Appcache steps, Diskcache steps) | (Appcache size, diskcache size): ({memcache_steps}, {disk_csteps}) | ({memcache_sizes}, {disk_csizes})")
            ####################### Implementation:  4. ILP Solver #######################
            ####################### Implementation:  5. Active Memory Estimator #######################
            memcache_sizes, memcache_steps, disk_csizes, disk_csteps = adjust_for_activemem(
                self.args, memcache_steps, disk_csteps, mem_profile, memcached_tensors, diskcached_tensors, remaining_samples, self.config)
            self.config.diskcache_size = sum(disk_csizes)
            self.config.memcache_size = sum(memcache_sizes)
            ####################### Implementation:  5. Active Memory Estimator #######################
            ####################### Implementation:  6. Pipeline Generator #######################
            self.built_pipes = merge_and_collect_pipes(memcache_steps, disk_csteps, memcache_sizes, disk_csizes, self.config, self.args)
            # Merge common step pipelines

        print(f"Using pipes: {self.built_pipes}, {[i.py_num_workers for i in self.built_pipes]}, {[pipe.input.full_iterations for pipe in self.built_pipes]}")
        iter = HyCacheGenericIterator(self.built_pipes)
        ####################### Implementation:  6. Pipeline Generator #######################
        return iter

    def _profile_steps(self):
        profile_steps(self.args.copy(), self.config.dataset)
    
    def __del__(self):
        try:
            shutil.rmtree(self.config.disk_cloc + "/*")
        except:
            pass
        for pipe in self.built_pipes:
            try:
                pipe.input.unlink()
            except:
                pass
