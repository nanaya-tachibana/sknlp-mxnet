import logging

import mxnet as mx
import numpy as np

import gluonnlp


logger = logging.getLogger(__name__)


def _pad_arrs_to_max_length(arrs, pad_axis, pad_val, dtype, min_length=0):
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
    max_size = max(min_length, max(original_length))
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
                    dtype=dtype
                )
    # return numpy array
    # ctx = mx.Context('cpu_shared', 0) if use_shared_mem else mx.cpu()
    # ret = mx.nd.array(ret, ctx=ctx, dtype=dtype)
    # original_length = mx.nd.array(original_length, ctx=ctx, dtype=np.int32)
    return ret, np.asarray(original_length)


class Pad(gluonnlp.data.batchify.Pad):
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

    def __init__(
        self, axis=0, pad_val=0, min_length=0, ret_length=False, dtype=None
    ):
        super().__init__(
            axis=axis, pad_val=pad_val, ret_length=ret_length, dtype=dtype
        )
        self._min_length = min_length

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
            logger.warning(
                'Using Pad with NDArrays is discouraged for speed reasons. '
                'Instead you should pad your data while it is still a list '
                'and before converting to an NDArray. '
                'Alternatively you can consider inputting a numpy.ndarray.'
            )
        if isinstance(data[0], (mx.nd.NDArray, np.ndarray, list)):
            padded_arr, original_length = _pad_arrs_to_max_length(
                data, self._axis, self._pad_val, self._dtype, self._min_length
            )
            if self._ret_length:
                return padded_arr, original_length
            else:
                return padded_arr
        else:
            raise NotImplementedError


def _stack_arrs(arrs, dtype):
    if isinstance(arrs[0], mx.nd.NDArray):
        dtype = arrs[0].dtype if dtype is None else dtype
        return mx.nd.stack(*arrs)
    else:
        return np.asarray(arrs, dtype=dtype)


class Stack(gluonnlp.data.batchify.Stack):
    """Stack the input data samples to construct the batch.

    The N input samples must have the same shape/length and will be stacked to construct a batch.

    Parameters
    ----------
    dtype : str or numpy.dtype, default None
        The value type of the output. If it is set to None, the input data type is used.

    Examples
    --------
    >>> from gluonnlp.data import batchify
    >>> # Stack multiple lists
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6, 8]
    >>> c = [8, 9, 1, 2]
    >>> gluonnlp.data.batchify.Stack()([a, b, c])
    <BLANKLINE>
    [[1 2 3 4]
     [4 5 6 8]
     [8 9 1 2]]
    <NDArray 3x4 @cpu_shared(0)>
    >>> # Stack multiple numpy.ndarrays
    >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = np.array([[5, 6, 7, 8], [1, 2, 3, 4]])
    >>> gluonnlp.data.batchify.Stack()([a, b])
    <BLANKLINE>
    [[[1 2 3 4]
      [5 6 7 8]]
    <BLANKLINE>
     [[5 6 7 8]
      [1 2 3 4]]]
    <NDArray 2x2x4 @cpu_shared(0)>
    >>> # Stack multiple NDArrays
    >>> a = mx.nd.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = mx.nd.array([[5, 6, 7, 8], [1, 2, 3, 4]])
    >>> gluonnlp.data.batchify.Stack()([a, b])
    <BLANKLINE>
    [[[1. 2. 3. 4.]
      [5. 6. 7. 8.]]
    <BLANKLINE>
     [[5. 6. 7. 8.]
      [1. 2. 3. 4.]]]
    <NDArray 2x2x4 @cpu_shared(0)>
    """
    def __call__(self, data):
        """Batchify the input data

        Parameters
        ----------
        data : list
            The input data samples

        Returns
        -------
        batch_data : NDArray
        """
        return _stack_arrs(data, self._dtype)


class BPTTBatchify:
    """
    Transform the dataset into batches of numericalized samples, in the way
    that the recurrent states from last batch connects with the current batch
    for each sample.

    Each sample is of shape `(seq_len, batch_size)`. When `last_batch='keep'`, the first
    dimension of last sample may be shorter than `seq_len`.

    Parameters
    ----------
    vocab : gluonnlp.Vocab
        The vocabulary to use for numericalizing the dataset. Each token will be mapped to the
        index according to the vocabulary.
    seq_len : int
        The length of each of the samples for truncated back-propagation-through-time (TBPTT).
    batch_size : int
        The number of samples in each batch.
    last_batch : {'keep', 'discard'}
        How to handle the last batch if the remaining length is less than `seq_len`.

        - keep: A batch with less samples than previous batches is returned. vocab.padding_token
          is used to pad the last batch based on batch size.

        - discard: The last batch is discarded if it's smaller than `(seq_len, batch_size)`.
    """

    def __init__(
        self, padding_token, bos_token, eos_token
    ):
        self._padding_token = padding_token
        self._bos_token = bos_token
        self._eos_token = eos_token

    def __call__(self, data):
        """Batchify a dataset.

        Parameters
        ----------
        corpus : mxnet.gluon.data.Dataset
            A flat dataset to be batchified.

        Returns
        -------
        mxnet.gluon.data.Dataset
            Batches of numericalized samples such that the recurrent states
            from last batch connects with the current batch for each sample.
            Each element of the Dataset is a tuple of data and label arrays for
            BPTT. They are of shape (seq_len, batch_size) respectively.
        """
        if isinstance(data[0], mx.nd.NDArray):
            data_list = [d.asnumpy().tolist() for d in data]
        elif isinstance(data[0], np.ndarray):
            data_list = [d.tolist() for d in data]
        else:
            data_list = list(data)
        data_list = [
            [self._bos_token] + d + [self._eos_token] for d in data_list
        ]
        target_list = [d[1:] + [self._bos_token] for d in data_list]
        reversed_target_list = [[self._eos_token] + d[:-1] for d in data_list]
        _pad = Pad(pad_val=self._padding_token)
        _pad_with_length = Pad(pad_val=self._padding_token, ret_length=True)
        return (
            _pad_with_length(data_list),
            _pad(target_list),
            _pad(reversed_target_list)
        )
