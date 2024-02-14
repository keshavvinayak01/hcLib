"""
We need to create our own iterator because DALI will not support multiple pipelines
running different external_source operators that may have different iterations.
In addition, we want some randomness in the batches we want from varying cache sources,
therefore, it make sense to have our own iterator.
https://github.com/NVIDIA/DALI/issues/2169#issuecomment-668527392
"""
import numpy as np
from utils import get_batch

class HyCacheGenericIterator:
    """
    DO NOT USE DIRECTLY, Should only be created indirectly from the SmartHybridCache class.
    As of now, it only supports simple iteration on a single CPU - single GPU setup.
    """
    def __init__(self, pipes):
        self.pipes = pipes
        self._init_pipe()

    def __iter__(self):
        return self
    
    def _init_pipe(self):
        self.finished = [False]*len(self.pipes)
        self.active_pipes = [i for i in range(len(self.pipes))]
        self.pipe_hits = [0]*len(self.pipes)
        self.num_batches = [pipe.input.full_iterations for pipe in self.pipes]
        self.total_batches = sum(self.num_batches)
        self.batch_num = 0

    def _get_rand_batch(self):
        rng = np.random.choice(self.active_pipes)
        batch, self.pipe_hits[rng], self.finished[rng] = get_batch(self.pipes[rng], self.pipe_hits[rng], self.finished[rng])
        return rng, batch, self.pipe_hits[rng], self.finished[rng]

    def __next__(self):
        if(self.batch_num <= self.total_batches and len(self.active_pipes) > 0):
            self.batch_num += 1
            rng, batch, self.pipe_hits[rng], self.finished[rng] = self._get_rand_batch()
            if self.finished[rng]:
                self.active_pipes.remove(rng)
                if(len(self.active_pipes) == 0):
                    self._check_and_reset_pipes()
                else:
                    rng, batch, self.pipe_hits[rng], self.finished[rng] = self._get_rand_batch()
            return batch
        else:
            self._check_and_reset_pipes()

    def __len__(self):
        return self.total_batches

    def _check_and_reset_pipes(self):
        if(sum(self.finished) == len(self.finished)):
            for pipe in self.pipes:
                pipe.reset()
            self._init_pipe()
            raise StopIteration