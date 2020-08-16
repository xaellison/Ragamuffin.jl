"""
Compute partition numbers https://en.wikipedia.org/wiki/Partition_(number_theory)
This example is not performant because for initial value `N` every step invokes
recursion on `N` subproblems but for most steps this creates many subproblems
with negative values, which are simply discarded.
"""

include("../ragamuffin.jl")

function partition(N, floor=1)
    if N == 0
        return 1
    end
    out = 0
    for i in floor:N
        out += partition(N - i, i)
    end
    return out
end

function expand_helper(x :: Cell{T}, n) :: Cell{T} where T
    if n >= x.value[2]
        return Cell{T}((x.value[1] - n, n), x.depth, x.output_index)
    else
        return Cell{T}((-1, -1) , -1, -1)
    end
end

function ragamuffin_partition(min_N, max_N)

    T = Tuple{Int32, Int32}
    program = initialize(collect((Int32(i), Int32(1)) for i in min_N:max_N), (-Int32(1), -Int32(1)), CuFrame, 1024, Int)
    add!(program, :root, (:test_base_case, :end), value_function = x -> x.value[1] >= 0 )
    add!(program, :test_base_case, (:increment, :expand), value_function = x -> x.value[1] == 0)
    add!(program, :increment, :end, dest_function = x -> 1)
    add!(program, :expand, :root, value_function = (expand_helper, collect(1:max_N)))

    @time run(program)
    return program.dest
end
max_N = 10
@time partition(max_N)
expected = collect(partition(i) for i in 1:max_N)
batched = ragamuffin_partition(1, max_N)
@info batched
@assert expected == batched
