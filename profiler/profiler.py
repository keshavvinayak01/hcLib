import shutil
import os
import subprocess
import psutil
import ast
from utils import setup_logging
from pipegen import create_pipe
from iterator import HyCacheGenericIterator
import time
import numpy as np
from tqdm import tqdm
from math import ceil

def vanilla_pipeline_run(args, filenames, epochs=2, avg=True, fetchers=-1):
    num_workers = args['num_workers']
    args['directio'] = 0
    args['num_workers'] = args['threads']//2 if fetchers < 0 else fetchers
    pipe, _ = create_pipe(args, filenames, pipetype='vanilla')
    times = []
    iter = HyCacheGenericIterator([pipe])
    for _ in range(epochs):
        begin = time.time()
        for batch in iter:
            batch=None
        times.append(round(time.time() - begin, 2))
    if len(times) == 1:
        avg_epoch_time = np.round(np.mean(times), 2)
    else:
        avg_epoch_time = np.round(np.mean(times[1:]), 2)
    args['directio'] = 1
    args['num_workers'] = num_workers
    if avg:
        avg_epoch_time = round((avg_epoch_time * 1000) / len(filenames), 3)
        avg_sample_size = np.mean([os.path.getsize(sample) for sample in filenames]) / (1 << 20)
    else:
        avg_sample_size = sum([os.path.getsize(sample) for sample in filenames])/ (1 << 30)
    return [round(avg_sample_size, 3), avg_epoch_time]

def profile_device(args, filenames, profile='disk'):
    samples = filenames.copy()[ : int(len(filenames) * args['profile_factor'])]
    logfile = os.path.join(args['logdir'], f'{profile}_profile.log')
    if os.path.exists(logfile):
        with open(logfile, 'r') as file:
            text = file.read().split('\n')
            try:
                log = ast.literal_eval(text[0])
                log = {key: [log[key][0], log[-1][1] - log[key][1]] for key in sorted(log)}
                
                return log
            except:
                print(f"Profiling {profile} profile...")
    else:
        print(f"Profiling {profile} profile...")

    logger = setup_logging(f'{profile}_profile', args['logdir'], logfile)

    logging_data = {}
    if profile == 'disk':
        args['disk_csize'] = 50
        args['disk_cloc'] += '/_temp'
        if not os.path.exists(args['disk_cloc']):
            os.makedirs(args['disk_cloc'])
        args['directio'] = 1
    else:
        args['directio'] = 0
        args['cache_size'] = 50
    logging_data[-1] = vanilla_pipeline_run(args, samples)
    subprocess.run(
        ["bash", "-c", "sync;sudo sysctl -w vm.drop_caches=3;sync;"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for step in tqdm(sorted(args["cc_steps"])):
        args['file_map'] = {file: i for i, file in enumerate(samples)}
        args['idx_map'] = {i: file for i, file in enumerate(samples)}
        args['global_cache_map'] = {i: 0 for i in range(len(samples))}

        if profile == 'disk':
            args['disk_cstep'] = step
            profile_helpers = {'disk_cstep': step, 'cache_steps': args['cache_steps']}
            args['num_cworkers'] = args['threads']//4
            pipe, _ = \
                create_pipe(args, samples, step=step, disk_csize=args['disk_csize'], profile_helpers=profile_helpers, pipetype='cached')
        else:
            args['cache_step'] = step
            args['num_cworkers'] = 0
            args['num_workers'] = 0
            profile_helpers = {'cache_step': step, 'cache_steps': args['cache_steps']}
            pipe, _ = \
                create_pipe(args, samples, step=step, cache_size=args['cache_size'], profile_helpers=profile_helpers, pipetype='cached')

        if profile == 'disk':
            fill_size = (pipe.input.disk_fill_size / (1 << 20))
        else:
            fill_size = (pipe.input.fill_size / (1 << 20))
        total_batches = pipe.input.full_iterations
        average_time = []
        iter = HyCacheGenericIterator([pipe])
        for epoch in range(2):
            begin = time.time()
            for batch in iter:
                batch=None
            average_time.append(time.time() - begin)
        # Time per samaple in milliseconds, size per sample in MB
        average_time = round((min(average_time)*1000) / (total_batches* args['batch_size']), 3)
        average_size = round(fill_size / (total_batches* args['batch_size']), 3)
        logging_data[step] = [average_size, average_time]
        if profile == 'memory':
            pipe.input.unlink()
            subprocess.run(
                ["bash", "-c", "sync;sudo sysctl -w vm.drop_caches=3;sync;"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(
                ["bash", "-c", f"rm -rf {args['disk_cloc']}/*"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.critical(str(logging_data))
    if profile == 'disk':
        try:
            shutil.rmtree(args['disk_cloc'] + "/*")
        except:
            pass
        args['disk_cloc'] = args['disk_cloc'].split('_temp')[0]
    logging_data = {key: [logging_data[key][0], logging_data[-1][1] - logging_data[key][1]] for key in sorted(logging_data)}
    return logging_data

def profile_steps(args, filenames):
    samples = filenames.copy()[ : int(len(filenames) * args['profile_factor'])]
    logfile = os.path.join(args['logdir'], f'steps_profile.log')
    logger = setup_logging(f'steps_profile', args['logdir'], logfile)
    if os.path.exists(logfile):
        with open(logfile, 'r') as file:
            text = file.read().split('\n')
            try:
                log = ast.literal_eval(text[0])
                return log
            except:
                print(f"Running Profile steps...")
    else:
        print(f"Running Profile steps...")
    logging_data = {}
    logging_data[-1] = vanilla_pipeline_run(args, samples, avg=False, fetchers=0)
    subprocess.run(
        ["bash", "-c", "sync;sudo sysctl -w vm.drop_caches=3;sync;"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for step in tqdm(range(args["cache_steps"])):
        args['file_map'] = {file: i for i, file in enumerate(samples)}
        args['idx_map'] = {i: file for i, file in enumerate(samples)}
        args['global_cache_map'] = {i: 0 for i in range(len(samples))}
        args['directio'] = 0
        args['num_workers'] = 0
        pipe, _ = create_pipe(
            args, samples,
            profile_helpers={'cache_step': step, 'cache_steps': args['cache_steps']},
            pipetype='caching'
        )
        iter = HyCacheGenericIterator([pipe])
        size = 0
        epoch_t = -1
        for epoch in range(2):
            begin = time.time()
            for i, batch in enumerate(iter):
                if epoch == 0:
                    datapoints = [np.array(batch[0].at(i)) for i in range(len(batch[0]))]
                    size += sum([sample.nbytes for sample in datapoints])
            if epoch:
                epoch_t = time.time() - begin
        # Scaled approximately over the entire dataset
        logging_data[step] = [size / (1 << 30), epoch_t]

    for key in logging_data.keys():
        print(key, logging_data[key])
        logging_data[key][0] = round(logging_data[key][0] * (len(filenames) / len(samples)), 2)
        logging_data[key][1] = round(logging_data[key][1] * (len(filenames) / len(samples)), 2)

    subprocess.run(
        ["bash", "-c", f"rm -rf {args['disk_cloc']}/*"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.critical(str(logging_data))
    print(logging_data)
    return logging_data
