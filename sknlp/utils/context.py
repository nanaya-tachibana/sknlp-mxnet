import logging

import mxnet as mx
try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None


logger = logging.getLogger(__name__)


def try_gpu(i=0):
    try:
        ctx = mx.gpu(i)
        mx.nd.array([0], ctx=ctx)
    except:
        return None
    return mx.gpu(i)


def get_context(multigpu=False):
    if multigpu:
        if hvd is None:
            logger.warn('Cannot load horovod. Use single gpu instead.')
        else:
            # Initialize Horovod
            hvd.init()
            # Pin GPU to be used to process local rank
            context = mx.gpu(hvd.local_rank())
            return context, hvd.size()

    for i in range(10):
        c = try_gpu(i)
        if c is not None:
            return c, 1

    return mx.cpu(), 1
