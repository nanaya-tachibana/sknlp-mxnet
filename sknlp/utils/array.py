def sequence_mask(arr, length, sequence_axis=0, batch_axis=1):
    slices = [slice(None) for _ in range(arr.ndim)]
    for i, l in enumerate(length):
        slices[batch_axis] = slice(i, i + 1)
        slices[sequence_axis] = slice(l, None)
        arr[tuple(slices)] = 0
    return arr
