import re
import logging

import mxnet as mx
import numpy as np

import gluonnlp as nlp


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        mx.nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def strip_whitespace(s):
    return re.sub('[\s\t\n]+', '', s)


def preprocess(data, padding=False, cut_func=word_cut_func, max_length=50):
    query = data[0]
    candidate = data[1]
    y = data[2]
    seg_query = list(cut_func(strip_whitespace(query)))[:max_length]
    query_len = len(seg_query)
    seg_candidate = list(cut_func(strip_whitespace(candidate)))[:max_length]
    candidate_len = len(seg_candidate)
    if padding:
        seg_query.extend(['<unk>'] * (max_length - len(seg_query)))
        seg_candidate.extend(['<unk>'] * (max_length - len(seg_candidate)))
    return ((seg_query, seg_candidate, query_len, candidate_len, y),
            (query_len, candidate_len))


def _pad_arrs_to_max_length(arrs, pad_axis, pad_val,
                            use_shared_mem, dtype,
                            max_length=None):
    """Inner Implementation of the Pad batchify

    Parameters
    ----------
    arrs : list
    pad_axis : int
    pad_val : number
    use_shared_mem : bool, default False

    Returns
    -------
    ret : NDArray
    original_length : NDArray
    """
    if isinstance(arrs[0], mx.nd.NDArray):
        dtype = arrs[0].dtype if dtype is None else dtype
        arrs = [arr.asnumpy() for arr in arrs]
    elif not isinstance(arrs[0], np.ndarray):
        arrs = [np.asarray(ele) for ele in arrs]
    else:
        dtype = arrs[0].dtype if dtype is None else dtype

    original_length = [ele.shape[pad_axis] for ele in arrs]
    max_size = max_length or max(original_length)
    arrs = [arr[:max_size] for arr in arrs]

    ret_shape = list(arrs[0].shape)
    ret_shape[pad_axis] = max_size
    ret_shape = (len(arrs), ) + tuple(ret_shape)

    if pad_val is None:
        ret = np.full(shape=ret_shape, fill_value=0, dtype=dtype)
    else:
        ret = np.full(shape=ret_shape, fill_value=pad_val, dtype=dtype)

    for i, arr in enumerate(arrs):
        if arr.shape[pad_axis] == max_size:
            ret[i] = arr
        else:
            slices = [slice(None) for _ in range(arr.ndim)]
            slices[pad_axis] = slice(0, arr.shape[pad_axis])
            if slices[pad_axis].start != slices[pad_axis].stop:
                row_slices = [slice(i, i + 1)] + slices
                ret[tuple(row_slices)] = arr
            if pad_val is None:
                slices[pad_axis] = slice(arr.shape[pad_axis] - 1,
                                         arr.shape[pad_axis])
                last_value = arr[tuple(slices)]
                slices[pad_axis] = slice(arr.shape[pad_axis], max_size)
                row_slices = [slice(i, i + 1)] + slices
                ret[tuple(row_slices)] = np.arange(
                    last_value + 1,
                    last_value + max_size - original_length[i] + 1,
                    dtype=dtype)

    ctx = mx.Context('cpu_shared', 0) if use_shared_mem else mx.cpu()
    ret = mx.nd.array(ret, ctx=ctx, dtype=dtype)
    original_length = mx.nd.array(original_length, ctx=ctx, dtype=np.int32)

    return ret, original_length


class Pad(nlp.data.batchify.Pad):
    """Return a callable that pads and stacks data.

    Parameters
    ----------
    axis : int, default 0
        The axis to pad the arrays. The arrays will be padded to the largest dimension at
        `axis`. For example, assume the input arrays have shape
        (10, 8, 5), (6, 8, 5), (3, 8, 5) and the `axis` is 0. Each input will be padded into
        (10, 8, 5) and then stacked to form the final output, which has shape（3, 10, 8, 5).
    pad_val : float or int, default 0
        The padding value.
    ret_length : bool, default False
        Whether to return the valid length in the output.
    dtype : str or numpy.dtype, default None
        The value type of the output. If it is set to None, the input data type is used.

    Examples
    --------
    >>> # Inputs are multiple lists
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6]
    >>> c = [8, 2]
    >>> gluonnlp.data.batchify.Pad()([a, b, c])
    <BLANKLINE>
    [[1. 2. 3. 4.]
     [4. 5. 6. 0.]
     [8. 2. 0. 0.]]
    <NDArray 3x4 @cpu_shared(0)>
    >>> # Also output the lengths
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6]
    >>> c = [8, 2]
    >>> batch, length = gluonnlp.data.batchify.Pad(ret_length=True)([a, b, c])
    >>> batch
    <BLANKLINE>
    [[1. 2. 3. 4.]
     [4. 5. 6. 0.]
     [8. 2. 0. 0.]]
    <NDArray 3x4 @cpu_shared(0)>
    >>> length
    <BLANKLINE>
    [4 3 2]
    <NDArray 3 @cpu_shared(0)>
    >>> # Inputs are multiple ndarrays
    >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = np.array([[5, 8], [1, 2]])
    >>> gluonnlp.data.batchify.Pad(axis=1, pad_val=-1)([a, b])
    <BLANKLINE>
    [[[ 1  2  3  4]
      [ 5  6  7  8]]
    <BLANKLINE>
     [[ 5  8 -1 -1]
      [ 1  2 -1 -1]]]
    <NDArray 2x2x4 @cpu_shared(0)>
    """

    def __init__(self, axis=0, pad_val=0,
                 max_length=None,
                 ret_length=False, dtype=None):
        self._axis = axis
        assert isinstance(axis, int), f'axis must be an integer! ' \
                                      f'Received axis={axis}, ' \
                                      f'type={type(axis)}.'
        self._pad_val = pad_val
        self._ret_length = ret_length
        self._dtype = dtype
        self._warned = False
        self._max_length = max_length

    def __call__(self, data):
        """Batchify the input data.

        The input can be list of numpy.ndarray, list of numbers or list of
        mxnet.nd.NDArray. Inputting mxnet.nd.NDArray is discouraged as each
        array need to be converted to numpy for efficient padding.

        The arrays will be padded to the largest dimension at `axis` and then
        stacked to form the final output. In addition, the function will output
        the original dimensions at the `axis` if ret_length is turned on.

        Parameters
        ----------
        data : List[np.ndarray] or List[List[dtype]] or List[mx.nd.NDArray]
            List of samples to pad and stack.

        Returns
        -------
        batch_data: NDArray
            Data in the minibatch. Shape is (N, ...)
        valid_length: NDArray, optional
            The sequences' original lengths at the padded axis. Shape is (N,). This will only be
            returned in `ret_length` is True.

        """

        if isinstance(data[0], mx.nd.NDArray) and not self._warned:
            self._warned = True
            logging.warning(
                'Using Pad with NDArrays is discouraged for speed reasons. '
                'Instead you should pad your data while it is still a list '
                'and before converting to an NDArray. '
                'Alternatively you can consider inputting a numpy.ndarray.')
        if isinstance(data[0], (mx.nd.NDArray, np.ndarray, list)):
            padded_arr, original_length = _pad_arrs_to_max_length(
                data, self._axis,
                self._pad_val, True,
                self._dtype, self._max_length)
            if self._ret_length:
                return padded_arr, original_length
            else:
                return padded_arr
        else:
            raise NotImplementedError
