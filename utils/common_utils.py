import re
import numpy as np
import os, mmap, sys
import math
import logging
import subprocess
formatter = logging.Formatter('%(message)s')

def total_size_dict(obj):
    total_size = 0
    for key in obj.keys():
        if(type(obj[key]) == dict):
            total_size += total_size_dict(obj[key])
        elif(type(obj[key]) == list):
            total_size += np.array(obj[key]).nbytes
        else:
            total_size += sys.getsizeof(obj[key]) + sys.getsizeof(key)
    return total_size

def pagecache_fill(samples, cache_size, bs, random=True):
    # cache_size fed directly to open() O_DIRECT, hence by default on.
    if cache_size == 0:
        return {}, 0
    pagecached_data = {i: 1 for i in range(len(samples))}
    fill_size = 0
    indices = [i for i in range(len(samples))]
    count = 1
    batch = []
    bs = max(512, bs)
    for i in indices:
        batch.append(samples[i])
        if(i%bs or i == 0):
            continue
        else:
            print(f"Filling pagecache {round(fill_size, 2)}/{cache_size} | Marking indices {i-bs} to {i}", end="\r", flush=True)
            if fill_size >= cache_size:
                break
            for j in range(i-bs,i):
                pagecached_data[j] = 0
            os.system('vmtouch -t ' + ' '.join(batch) + ' > /dev/null 2>&1')
            fill_size += sum([os.path.getsize(item) for item in batch]) / float(1 << 30)
            count += bs
            batch = []
    return pagecached_data, fill_size

def dataloader(filename, directio=1, dtype=np.uint8):
    flags = os.O_RDONLY | os.O_DIRECT if directio else os.O_RDONLY
    file_fd = os.open(filename, flags)
    fo = os.fdopen(file_fd, 'rb+', buffering=0)
    filesize = 512 * math.ceil(os.path.getsize(filename) / 512)
    file_mmap = mmap.mmap(-1, filesize)
    fo.readinto(file_mmap)
    tensor = np.frombuffer(file_mmap.read(), dtype)
    return tensor

def __setup_logging(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    try:
        os.remove(log_file)
    except:
        pass
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def setup_logging(name, basedir, logfile):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    return __setup_logging(name, logfile)

def get_batch(pipe, pipe_hit, finished):
    batch = None
    try:
        batch = pipe.run()
        pipe_hit += 1
    except StopIteration:
        finished = True
    return batch, pipe_hit, finished

def clear_pagecache():
    subprocess.run(
        ["bash", "-c", "sync;sudo sysctl -w vm.drop_caches=3;sync;"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_used_memory():
    output = subprocess.check_output(['free', '-m']).decode('utf-8').split('\n')
    return int([line for line in output if line.startswith('Mem:')][0].split()[2])

def get_mem_size(stat='total'):
    output = subprocess.check_output(['free', '-tmh']).decode('utf-8')
    if stat == 'total':
        line = re.search(r'^\s*Total:\s+([\d.]+[KMGTP]i?)', output, re.MULTILINE)
    elif stat == 'used':
        line = re.search(r'^Mem:\s+\S+\s+(\S+)', output, re.MULTILINE)
    elif stat == 'free':
        line = re.search(r'^Mem:\s+\S+\s+\S+\s+(\S+)', output, re.MULTILINE)
    
    mem = line.group(1)
    return float(mem.split('G')[0])

def get_presto_solution(args, disk_profile, nfiles, thput):
    for key in disk_profile.keys():
        disk_profile[key][0] = (disk_profile[key][0]*nfiles)/(1 << 10)
#    print([(i, disk_profile[i][0]) for i in disk_profile.keys()])
    fitting_steps = [step for step in disk_profile.keys() if disk_profile[step][0] < args['disk_budget']]
    if(len(fitting_steps) == 0):
        return -2
    # Metric is pipeline of 2 batches: fetch + pre-process + (fetch > pp: fetch - pp: 0)
    times = {step: disk_profile[step][1] for step in fitting_steps}
    optimal_step = sorted(times, key = lambda y: times[y])[0]
    print(f"Choosing Step-{optimal_step} for disk cache")
    return optimal_step
