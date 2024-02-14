import os
import types
import nvidia.dali.types as dalitypes
import numpy as np
import shutil
from utils import total_size_dict, dataloader, pagecache_fill, clear_pagecache
from multiprocessing import shared_memory as shm
from iterator import HyCacheGenericIterator

def _caching_condition(profile_helpers, step):
    return profile_helpers['cache_step'] >= step and profile_helpers['cache_step'] < profile_helpers['cache_steps']
    
def _profiling_condition(profile_helpers, step):
    return profile_helpers['profile_step'] >= step

def _cached_condition(profile_helpers, step):
    return profile_helpers.get('disk_cstep', 100) < step or profile_helpers.get('cache_step', 100) < step

byte_sizes = {
    np.dtype('uint8'): 1,
    np.dtype('uint16'): 2,
    np.dtype('float32'): 4,
    np.dtype('float64'): 8,
    np.dtype('int32'): 4,
    np.dtype('int16'): 2,
    'uint8': 1,
    'uint16': 2,
    'float32': 4,
    'float64': 8,
    'int32': 4
}
# Logic for memory caching
def _build_memcache(args, shapes, nbytes, cache_location, uncached_samples, label_func):
    args['directio'] = 1
    pipe, _ = create_pipe(
        args, uncached_samples,
        profile_helpers={'cache_step': args['cache_step'], 'cache_steps': args['cache_steps']},
        pipetype='caching'
    )

    shm_buffer = shm.SharedMemory(create=True, size=args['cache_size']*(1 << 30))
    mem_view = shm_buffer.buf
    fill_size, offset, dtype = 0, 0, None
    # Caching logic for multi-threaded caching
    caching_iterator = HyCacheGenericIterator([pipe])
    for i, batch in enumerate(caching_iterator):
        if not batch:
            break
        samples = [np.array(batch[0].at(i)) for i in range(len(batch[0]))]
        indices = [np.array(batch[-1].at(i)).item() for i in range(len(batch[0]))]
        print(f"Filling memory {i}/{len(caching_iterator)} | fill size: {round(fill_size / (1 << 30), 2)}/{args['cache_size']} | step-{args['cache_step']}",  end="\r", flush=True)
        if fill_size + sum([sample.nbytes for sample in samples]) >= args['cache_size'] * (1 << 30):
            break
        for sample, index in zip(samples, indices):
            file = args['file_map'][uncached_samples[index]]
            args['global_cache_map'][file] = 1
            sample_nbytes = sample.nbytes

            shapes[file] = sample.shape
            nbytes[file] = sample_nbytes
            cache_location[file] = 1
            buffer = mem_view[offset:(offset + sample_nbytes)]
            dtype=sample.dtype
            shared_array = np.ndarray(sample.shape, dtype=sample.dtype, buffer=buffer)

            shared_array.ravel()[:] = sample.ravel()[:]
            offset += sample_nbytes
            fill_size += sample_nbytes
    uncached_samples = list(set(uncached_samples) - set([args['idx_map'][key] for key in cache_location.keys()]))
    del pipe
    return shm_buffer, shm_buffer.name, dtype, fill_size, uncached_samples

def _build_diskcache(args, disk_shapes, cache_location, uncached_samples, cache_loc, label_func):
    disk_cached_map = {}
    memcached_items = len(cache_location)
    disk_cached_idx = memcached_items
    args['directio'] = 1
    pipe, _ = create_pipe(
        args, uncached_samples,
        profile_helpers={'cache_step': args['cache_step'], 'cache_steps': args['cache_steps']},
        pipetype='caching'
    )
    if not os.path.exists(cache_loc):
        os.makedirs(cache_loc)
    disk_fill_size, byte_set, disk_dtype = 0, 0, None
    disk_csize = args['disk_csize'] * (1 << 30)
    disk_iterator = HyCacheGenericIterator([pipe])
    for i, batch in enumerate(disk_iterator):
        if not batch:
            break
        samples = [np.array(batch[0].at(i)) for i in range(len(batch[0]))]
        indices = [np.array(batch[-1].at(i)).item() for i in range(len(batch[0]))]
        print(f"Filling Disk {i}/{len(disk_iterator)} | fill size: {round(disk_fill_size / (1 << 30), 2)}/{args['disk_csize']} | step-{args['cache_step']}",  end="\r", flush=True)

        if disk_fill_size + sum([sample.nbytes for sample in samples]) >= disk_csize:
            break
        for sample, index in zip(samples, indices):
            # Need to make this write O_DIRECT
            file = args['file_map'][uncached_samples[index]]
            disk_cached_map[disk_cached_idx] = file
            cache_location[file] = 2
            args['global_cache_map'][file] = 2
            sample_nbytes = sample.nbytes
            disk_dtype = sample.dtype
            np.save(f"{cache_loc}/{args['rank']}_{file}.npy", sample.reshape(-1))
            if not byte_set:
                byte_sizes[disk_dtype] = sample.reshape(-1)[0].nbytes
                byte_set = 1
            disk_shapes[file] = sample.shape
            disk_fill_size += sample_nbytes
            disk_cached_idx += 1
    clear_pagecache()
    uncached_samples = list(set(uncached_samples) - \
        set([args['idx_map'][key] for key in cache_location.keys() if cache_location[key] == 2]))
    del pipe
    return disk_cached_map, memcached_items, disk_cached_idx, disk_dtype, disk_fill_size, uncached_samples

def create_pipe(args, filenames, step=-1, cache_size=0, disk_csize=0, profile_helpers=None, pipetype='vanilla'):
    to_reset = {}
    if step >= 0:
        to_reset['disk_csize'] = args['disk_csize']
        to_reset['cache_size'] = args['cache_size']
        to_reset['disk_cstep'] = args['disk_cstep']
        to_reset['cache_step'] = args['cache_step']
        to_reset['num_cworkers'] = args['num_cworkers']
        args['disk_csize'] = disk_csize
        args['cache_size'] = cache_size
        args['disk_cstep'] = step if disk_csize >= 0 else -1
        args['cache_step'] = step if cache_size >= 0 else -1
        profile_helpers = {'disk_cstep': step, 'cache_step': step,'cache_steps': args['cache_steps']}

    workers = args['num_cworkers'] if pipetype == 'cached' else args['num_workers'] if pipetype == 'vanilla' else 0
    input_func = CachedExternInputCallable if pipetype == 'cached' else RawExternInputCallable
    condition = _cached_condition if pipetype == 'cached' else _profiling_condition if pipetype == 'vanilla' else _caching_condition 
    profile_helpers = profile_helpers if profile_helpers is not None else {'profile_step': args['cache_steps']+100}
    threads = args['threads']

    pipe = args['pipe'](
        batch_size=args['batch_size'],
        num_threads=threads,
        num_workers=workers,
        device_id=dalitypes.CPU_ONLY_DEVICE_ID,
    )

    pipe.set_required_params(
        input_func=input_func,
        condition=condition,
        profile_helpers=profile_helpers,
        samples=filenames,
        label_func=args['label_func'],
        args=args
    )
    pipe.build()
    for key in to_reset.keys():
        args[key] = to_reset[key]
    return pipe, pipe.input.uncached_samples if pipetype == 'cached' else None

class CachedExternInputCallable:
    def __init__(self,
                 samples:list,
                 batch_size:int=64,
                 type=np.float32,
                 shuffle:bool=True,
                 label_func:types.FunctionType = None,
                 args: dict=None
                ):
        assert args['batch_size'] > 0, "Batch size should be non-negative"
        self.batch_size = batch_size
        self.directio = args['directio']
        self.shuffle = shuffle
        self.type = type
        self.seeds = np.arange(2**20)
        self.cache_step = args['cache_step']
        self.disk_cstep = args['disk_cstep']
        self.label_func = label_func

        args = args
        assert len(samples) > 0, "Empty Samples List!"
        self.uncached_samples = samples.copy()
        self.last_seen_epoch = None
        self.total_items = 0 
        self.cache_loc = f"{args['disk_cloc']}/{args['cache_step']}/"
        # Separate out the memcache and disk_cache creation 
        shapes = {}
        disk_shapes = {}
        nbytes = {}
        cache_location = {}
        memcached_items=0
        disk_cached_map={}
        disk_fill_size=0
        if args['cache_size'] > 0:
            _, self.shm_buffer_name, self.dtype, self.fill_size, self.uncached_samples = \
                _build_memcache(args.copy(), shapes, nbytes, cache_location, self.uncached_samples, self.label_func)
        memcached_items = len(cache_location)
        disk_cached_idx = memcached_items
        if len(self.uncached_samples) > args['batch_size'] and args['disk_csize'] > 0:
            disk_cached_map, memcached_items, disk_cached_idx, self.disk_dtype, disk_fill_size, self.uncached_samples = \
                _build_diskcache(args.copy(), disk_shapes, cache_location, self.uncached_samples, self.cache_loc, self.label_func)

        disk_cached_map_vals = list(disk_cached_map.values())
        self.memcached_items = memcached_items
        self.disk_cached_map = {memcached_items + i: val for i, val in enumerate(disk_cached_map_vals)}
        self.disk_shapes = disk_shapes
        self.disk_fill_size = disk_fill_size
        self.total_items = disk_cached_idx
        self.shapes = shapes
        self.nbytes = nbytes
        self.metadata_size = round(total_size_dict(self.__dict__) // (1 << 20), 2)
        self.full_iterations = self.total_items // self.batch_size
        self.perm = np.arange(self.total_items)
        if args['num_cworkers'] == 0:
            state = self.__getstate__()
            self.__setstate__(state)

    def __getstate__(self):
        members = self.__dict__.copy()
        del members["uncached_samples"]
        return members

    def __setstate__(self, state=None):
        if not state:
            state = self.__dict__.copy()
        nbytes, shapes, dtype, total_items, args = \
            state.get("nbytes"), state.get("shapes"), state.get("dtype"), state.get('total_items'), state.get('args')

        self.samples = []
        if args['cache_size'] > 0:
            self.__dict__.update(state)
            self.shm_buffer = shm.SharedMemory(self.shm_buffer_name)
            mem_view = self.shm_buffer.buf
            offset = 0
            assert len(nbytes) == len(shapes)
            for file in shapes.keys():
                sample_shape = shapes[file]
                sample_nbytes = nbytes[file]
                buffer = mem_view[offset:(offset + sample_nbytes)]
                sample = np.ndarray(
                    sample_shape, dtype=dtype, buffer=buffer)
                self.samples.append(sample)
                offset += sample_nbytes
        self.__dict__.update(state)
        self.update_iter_size(total_items, state.get('batch_size'))

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration()
        # Samples are shuffled every epoch
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            np.random.seed(self.seeds[self.last_seen_epoch % (2**20)])
            self.perm = np.random.permutation(self.perm)
        label = 1
        shuffled_idx = self.perm[sample_idx]
        if self.disk_cstep >= 0 and shuffled_idx >= self.memcached_items:
            file = self.disk_cached_map[shuffled_idx]
            shape = self.disk_shapes[file]
            sample = dataloader(f"{self.cache_loc}/{args['rank']}_{file}.npy", args['directio'], self.disk_dtype)
            offset = 128//byte_sizes[self.disk_dtype]
            sample = sample[offset:np.prod(shape) + offset]
            sample = sample.reshape(shape).astype(self.type)
        else:
            sample = np.array(self.samples[shuffled_idx], self.type)

        # You can define your own function to load the labels
        # CACHING LABELS REQUIRE HUGE OVERWORK TO THE LIBRARY, but is possible.
        label = self.label_func(args['idx_map'][shuffled_idx]) if self.label_func else 1
        return sample, np.array(label,dtype=np.int32), np.array(shuffled_idx,dtype=np.int32)

    def unlink(self):
        if self.shm_buffer is not None:
            self.shm_buffer.close()
            self.shm_buffer.unlink()

    def update_iter_size(self, size, bs):
        self.full_iterations = size // bs

class RawExternInputCallable:
    def __init__(self,
                 samples:list,
                 batch_size:int=64,
                 type=np.float32,
                 shuffle:bool=True,
                 label_func:types.FunctionType = None,
                 args: dict=None
                ):
        assert batch_size > 0, "Batch size should be non-negative"

        self.batch_size = batch_size
        self.directio = args['directio']
        self.seeds = np.arange(2**20)
        self.shuffle = shuffle
        assert len(samples) > 0, "Empty Samples List!"

        self.samples = samples
        self.full_iterations = len(self.samples) // batch_size
        self.last_seen_epoch = None
        self.type = type
        self.perm = np.arange(len(self.samples))
        self.pagecached_data, self.fill_size = pagecache_fill(samples, args['pagecache_size'] // args['world_size'], self.batch_size)
        self.metadata_size = round(total_size_dict(self.__dict__.copy()) / (1 << 20), 2)
        self.label_func = label_func

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration()

        # Samples are shuffled every epoch
        if self.shuffle and self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            np.random.seed(self.seeds[self.last_seen_epoch % (2**20)])
            self.perm = np.random.permutation(self.perm)

        sample_idx = self.perm[sample_idx]
        filename = self.samples[sample_idx]
        directio = self.directio and self.pagecached_data.get(sample_idx, 1)
        encoded_img = dataloader(filename, directio=directio)

        # You can define your own function to load the labels
        label = self.label_func(filename) if self.label_func else 1
        return np.array(encoded_img, self.type), np.array(label,dtype=np.int32), np.array(sample_idx, dtype=np.int32)

def merge_and_collect_pipes(memcache_steps, disk_csteps, memcache_sizes, disk_csizes, config, args):
    common_steps = [x for x in memcache_steps if x in disk_csteps]
    uncached_samples = config.dataset
    built_pipes = []
    # Create mixed pipelines
    try:
        shutil.rmtree(config.disk_cloc + "/*")
    except:
        pass
    if(len(common_steps) > 0) and args['merge']:
        for step in common_steps:
            print(f"Using merged pipeline at step-{step}", args['merge'])
            args['num_cworkers'] = args['threads'] // 2
            cache_size = memcache_sizes[memcache_steps.index(step)]
            disk_csize = disk_csizes[disk_csteps.index(step)]
            pipe, uncached_samples = create_pipe(args, uncached_samples, step, cache_size, disk_csize, pipetype='cached')
            built_pipes.append(pipe)
            disk_csizes.remove(disk_csize)
            memcache_sizes.remove(cache_size)
            disk_csteps.remove(step)
            memcache_steps.remove(step)
    # Create memcache only pipes
    for i in range(len(memcache_sizes)):
        if(len(uncached_samples) < args['batch_size']):
            break
        args['num_cworkers'] = 0
        pipe, uncached_samples = \
            create_pipe(args, uncached_samples, memcache_steps[i], cache_size=memcache_sizes[i], pipetype='cached')
        built_pipes.append(pipe)

    # Create Diskcache only pipes
    if config.diskcache_size > 0:
        for i in range(len(disk_csizes)):
            if disk_csteps[i] == -1:
                continue
            if(len(uncached_samples) < args['batch_size']):
                break
            args['num_cworkers'] = args['max_workers'][disk_csteps[i]]
            pipe, uncached_samples = \
                create_pipe(args, uncached_samples, disk_csteps[i], disk_csize=disk_csizes[i], pipetype='cached')
            built_pipes.append(pipe)

    if(len(uncached_samples) > args['batch_size']):
        print("Creating uncached pipe")
        args['num_workers'] = args['max_workers'][-1]
        pipe, _ = create_pipe(args, uncached_samples, pipetype='vanilla')
        built_pipes.append(pipe)
    return built_pipes