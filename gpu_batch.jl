"""
Given an array of values and associated array of booleans, sort the values into
true / false and move those values to separate arrays
"""

using CUDA

CUDA.allowscalar(false)
"""
For a batch of size `n` what is the lowest index of the batch `i` is in
"""
function batch_floor(idx, n)
    return idx - (idx - 1) % n
end

"""
For a batch of size `n` what is the highest index of the batch `i` is in
"""
function batch_ceil(idx, n)
    return idx + n - 1 - (idx - 1) % n
end

"""
GPU friendly step function
"""
function Θ(i)
    return 1 & (1 <= i)
end

"""
Suppose we are merging two lists of size n, each of which has all falses before
all trues. Together, they will be indexed 1:2n. This is a fast stepwise function
for the destination index of a value at index `x` in the concatenated input,
where `a` is the number of falses in the first half, b = n - a, and false is the
number of falses in the second half.
"""
function step_swap(x, a, b, c)
    return x + Θ(x - a) * b - Θ(x - (a + c)) * (b + c) + Θ(x - (a + b + c)) * c
end

"""
Generalizes `step_swap` for when the floor index is not 1
"""
function batch_step_swap(x, n, a, b, c)
    idx = (x - 1) % n + 1
    return batch_floor(x, n) - 1 + step_swap(idx, a, b, c)
end

"""
For thread `idx` with current value `value`, merge two batches of size `n` and
return the new value this thread takes. `sums` and `swap` and shared mem
"""
function merge_swap_shmem(value, idx, n, sums, swap)
    b = sums[batch_floor(idx, 2 * n)]
    a = n - b
    d = sums[batch_ceil(idx, 2 * n)]
    c = n - d
    swap[idx] = value
    sync_threads()
    sums[idx] = d + b
    return swap[batch_step_swap(idx, 2 * n, a, b, c)]
end

"""
Given an array `flags`, perform a stable sort into `dest` of the indices of
`flags`.
Eg: [True, False, False, True] => [2, 3, 1, 4]
Note, stores indices modulo 1024 (max thread block size). Assumed use is with
`index_move_kernel`
"""
function index_sort_kernel(dest, flags, batch_sums)
    i = threadIdx().x
    idx0 = (blockIdx().x - 1) * blockDim().x + i # for addressing
    idx = i # for use in shared mem swaps
    swap = @cuStaticSharedMem(Int32, 1024)
    sums = @cuStaticSharedMem(Int32, 1024)
    sums[i] = 1 & flags[idx0]
    idx = merge_swap_shmem(idx, i, 1, sums, swap)
    idx = merge_swap_shmem(idx, i, 2, sums, swap)
    idx = merge_swap_shmem(idx, i, 4, sums, swap)
    idx = merge_swap_shmem(idx, i, 8, sums, swap)
    idx = merge_swap_shmem(idx, i, 16, sums, swap)
    idx = merge_swap_shmem(idx, i, 32, sums, swap)
    idx = merge_swap_shmem(idx, i, 64, sums, swap)
    idx = merge_swap_shmem(idx, i, 128, sums, swap)
    idx = merge_swap_shmem(idx, i, 256, sums, swap)
    idx = merge_swap_shmem(idx, i, 512, sums, swap)

    if i == 1
        batch_sums[blockIdx().x] = sums[i]
    end
    dest[idx0] = idx
    return nothing
end

"""
Given an array of `values` reorder them. Denote v0 as initial `values`, v1 as
modified. The result of this kernel is v1[i] = v0[indices[i]].
Makes use of shared memory so that global mem accesses are coalesced.
WARNING: Assumes T is small enough that we won't run out of shmem
"""
function index_move_kernel(values :: CuDeviceArray{T}, indices) where T
    i = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + i
    swap = @cuStaticSharedMem(T, 1024)
    swap[i] = values[idx]
    sync_threads()
    new_value = swap[indices[idx]]
    sync_threads()
    values[idx] = new_value
    return nothing
end

function global_move_kernel(values)

end
