"""
Compute fibonacci numbers by naive recursion... on the GPU. Probably the least
watt efficient calculation possible.
"""

include("../ragamuffin.jl")

function fib(N)
    if 1 <= N <= 2
        return 1
    end
    return fib(N - 1) + fib(N - 2)
end

function expand_helper(x :: Cell{T}, n) :: Cell{T} where T
    return Cell{T}(x.value - n, x.depth, x.output_index)
end

function ragamuffin_fib(min_N, max_N)
    T = Int32
    program = initialize(collect(Int32(i) for i in min_N:max_N), -Int32(1), CuFrame, 8*1024, Int)
    # NB: the true condition for a branching function cannot intersect null detection.
    # Here, our default value (for null objects) is -1, so 2 < x.value is fine, but
    # 2 >= x.value isn't. Any condition can be negated and the destinations swapped.
    add!(program, :root, (:branch, :increment), value_function = x -> 2 < x.value)
    add!(program, :branch, :root, value_function = (expand_helper, collect(1:2)))
    add!(program, :increment, :end, dest_function = x -> 1)

    @time run(program)
    return program.dest
end
max_N = 18
expected = @time collect(fib(i) for i in 1:max_N)
batched = ragamuffin_fib(1, max_N)
@info batched
@assert expected == batched
