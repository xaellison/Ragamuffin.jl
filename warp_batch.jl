# Code to accelerate gpu_batch but temporarily omitted for simplicity

macro merge_swap_shfl()
    q = quote
    b = shfl_sync(0xFFFFFFFF, sum, batch_floor(idx, 2 * n))
    a = n - b
    d = shfl_sync(0xFFFFFFFF, sum, batch_ceil(idx, 2 * n))
    c = n - d
    swap = value
    sum = d + b
    value = shfl_sync(0xFFFFFFFF, swap, batch_step_swap(idx, 2 * n, a, b, c))
    n *= 2
    end
    return esc(q)
end

macro repeat_inline(N, expr)
    function tail(expr, N)
        N <= 0 ? :() : :($expr; $(tail(expr, N - 1)))
    end
    out = tail(expr, N)
    return esc(out)
end

function shfl_reduce_kernel(src, dest, flags)
    i = threadIdx().x
    j = blockIdx().x
    w = blockDim().x
    floor = w * (j - 1)
    idx = floor + i
    @inbounds value = src[idx]

    @inbounds sum = 1 & flags[idx]
    n = 1
    # five merges => 2 ^ 5 = 32 = size of correct batches
    @repeat_inline 5 @merge_swap_shfl

    @inbounds dest[idx] = value
    return nothing
end

function shfl_reduce_kernel(A, flags)
    i = threadIdx().x
    j = blockIdx().x
    w = blockDim().x
    floor = w * (j - 1)
    idx = floor + i
    value = A[idx]

    sum = 1 & flags[idx]
    n = 1
    # five merges => 2 ^ 5 = 32 = size of correct batches
    @repeat_inline 5 @merge_swap_shfl

    A[idx] = value
    return nothing
end
