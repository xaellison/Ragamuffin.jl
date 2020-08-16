using Test, Logging

include("../gpu_batch.jl")

"""
For an array of length `N` * 1024 with value type `T`, test that each sequential
group of 1024 elements is stably sorted.
"""
function test_multi_batch(N, T)
    dest = CUDA.zeros(Int32, N * 1024)
    host_vals = rand(T, N * 1024)
    vals = CuArray(host_vals)
    odd = collect(filter(x -> mod(x, 2) == 1, host_vals[1 + 1024 * (i - 1):1024 * i]) for i in 1:N)
    even = collect(filter(x -> mod(x, 2) == 0, host_vals[1 + 1024 * (i - 1):1024 * i]) for i in 1:N)
    @assert Set(host_vals) == union(reduce(union, map(Set, odd)), reduce(union, map(Set, even)))
    flags = map(x -> x % 2 == 0, vals)
    batch_sums = CUDA.zeros(Int32, N)

    @cuda blocks=N threads=1024 index_sort_kernel(dest, flags, batch_sums)
    @cuda blocks=N threads=1024 index_move_kernel(vals, dest)
    synchronize()
    @test Array(vals) == reduce(vcat, collect(vcat(odd[i], even[i]) for i in 1:N))
end

test_multi_batch(1, Int16)
test_multi_batch(2, Int16)
test_multi_batch(1, Int64)
test_multi_batch(2, Int64)
test_multi_batch(64, Int64)
@info "passed"
