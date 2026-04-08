import functools

from ninetoothed import Symbol, Tensor


def arrangement(x, cos, sin, block_size=None):
    if block_size is None:
        block_size = Symbol("BLOCK_SIZE", constexpr=True)

    _, num_heads, _ = x.shape
    x_arranged = x.tile((1, 1, block_size)).tile((1, 1, 2))
    cos_arranged = cos.expand((-1, num_heads, -1)).tile((1, 1, block_size))
    sin_arranged = sin.expand((-1, num_heads, -1)).tile((1, 1, block_size))

    def _squeeze(arranged):
        for _ in range(2):
            arranged.dtype = arranged.dtype.squeeze(0)
        return arranged

    x_arranged = _squeeze(x_arranged)
    x_arranged.dtype.dtype = x_arranged.dtype.dtype.squeeze(0)
    x_arranged.dtype.dtype = x_arranged.dtype.dtype.squeeze(0)
    cos_arranged = _squeeze(cos_arranged)
    sin_arranged = _squeeze(sin_arranged)
    return x_arranged, cos_arranged, sin_arranged


def application(x, cos, sin):
    x1 = x[0]
    x2 = x[1]
    x[0] = x1 * cos - x2 * sin
    x[1] = x2 * cos + x1 * sin


def premake(block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    return arrangement_, application, (Tensor(3), Tensor(3), Tensor(3))
