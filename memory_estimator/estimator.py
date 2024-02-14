import os
import numpy as np
import shutil
from math import ceil
import time
import ast
from utils import setup_logging, get_used_memory
from pipegen import create_pipe
from iterator import HyCacheGenericIterator

def adjust_for_activemem(args, memcache_steps, disk_csteps, mem_profile, memcached_tensors, diskcached_tensors, remaining_samples, config):
    args['max_workers'] = get_worker_profile(args.copy(), config.dataset)
    if config.memcache_size > 0:
        max_prefetch_queue_size = max([2*args['batch_size']*i[0] for i in mem_profile.values()])
        num_pipes = len(np.unique(memcache_steps + disk_csteps))
        num_pipes = num_pipes + 1 if memcached_tensors + diskcached_tensors < len(config.dataset) else num_pipes
        required_workers = sum([args['max_workers'][step] for step in disk_csteps])
        required_workers = required_workers + args['max_workers'][-1] if memcached_tensors + diskcached_tensors < len(config.dataset) else required_workers
        offset = (required_workers * args['max_workers']['per_worker_mem'] + num_pipes*max_prefetch_queue_size) / 1024
        memcache_sizes[0] -= offset
        # This addition should actually be multiplied with the bloatup too. Let's see if we can make it work without doing so.
        if(remaining_samples > args['batch_size']) and disk_csizes[-1] > 0:
            disk_csizes[-1] = disk_csizes[-1] if sum(disk_csizes) >= config.diskcache_size \
                else min(disk_csizes[-1] + config.diskcache_size - sum(disk_csizes), disk_csizes[-1] + offset)
            disk_csizes = [ceil(i) for i in disk_csizes]
        else:
            disk_csizes = [0]
        memcache_sizes = [int(ceil(i/args['world_size'])) for i in memcache_sizes]
        print(f"Sizes after adjustment with offset {offset}GB: (Appcache size, diskcache size): ({memcache_sizes}, {disk_csizes})")
    disk_csizes = [int(ceil(i/args['world_size'])) for i in disk_csizes]
    return memcache_sizes, memcache_steps, disk_csizes, disk_csteps

def get_worker_profile(args, filenames):
    # Getting maximum achievable throughput at all disk-cached steps in the pipeline
    logfile = os.path.join(args['logdir'], 'workers.log')
    if os.path.exists(logfile):
        with open(logfile, 'r') as file:
            text = file.read().split('\n')
            try:
                return ast.literal_eval(text[0])
            except:
                print(f"Profiling maximum throughput at disk steps...")
    else:
        print(f"Profiling maximum throughput at disk steps...")

    logger = setup_logging('workers', args['logdir'], logfile)
    samples = filenames.copy()[ : int(len(filenames) * args['profile_factor'])]
    # Calculate the fetch rate at each cache step for 
    data = {}
    data['per_worker_mem'] = round(get_worker_memory(args, samples) + (3*np.array(filenames).nbytes) / (1 << 20), 2)
    args['disk_csize'] = min(1000, args['disk_budget'])
    for step in args['cc_steps']:
        step_time = []
        for num_workers in [0, args['threads']//2]:
            args['file_map'] = {file: i for i, file in enumerate(samples)}
            args['idx_map'] = {i: file for i, file in enumerate(samples)}
            args['global_cache_map'] = {i: 0 for i in range(len(samples))}
            args['disk_cstep'] = step
            args['num_cworkers'] = num_workers
            disk_creation = time.time()
            pipe, _ = \
                create_pipe(args, samples, step=step, disk_csize=args['disk_csize'], profile_helpers={'disk_cstep': step, 'cache_steps': 100}, pipetype='cached')
            iter = HyCacheGenericIterator([pipe])
            print("Redundant disk_creation time:", time.time() - disk_creation)
            begin = time.time()
            for batch in iter:
                batch=None
            step_time.append(round(time.time()-begin, 2))
        data[step] = ceil(2 ** (step_time[0] / step_time[1]))
    for num_workers in [0, args['threads']//2]:
        args['num_workers'] = num_workers
        disk_creation = time.time()
        pipe, _ = \
            create_pipe(args, samples, pipetype='vanilla')
        iter = HyCacheGenericIterator([pipe])
        print("Redundant disk_creation time:", time.time() - disk_creation)
        begin = time.time()
        for batch in iter:
            batch=None
        step_time.append(round(time.time()-begin, 2))
    data[-1] = ceil(2 ** (step_time[0] / step_time[1]))
    try:
        shutil.rmtree(args['disk_cloc'] + "/*")
    except:
        pass
    logger.critical(str(data))
    return data

def get_worker_memory(args, filenames):
    print("Profiling per-worker memory..")
    active_mem = []
    metadata = 0
    args['directio'] = 0
    for num_workers in [4, 8]:
        args['num_workers'] = num_workers
        pipe, _ = \
            create_pipe(args, filenames, profile_helpers={'profile_step': args['cache_steps']+100}, pipetype='vanilla')

        metadata = pipe.input.metadata_size
        iter = HyCacheGenericIterator([pipe])
        begin = time.time()
        for i, batch in enumerate(iter):
            batch=None
            if(i % 10 and (time.time() - begin) > 25):
                break
        active_mem.append(get_used_memory())
    args['directio'] = 1
    return max(100, (active_mem[1] - active_mem[0]) // 4 + metadata) 
