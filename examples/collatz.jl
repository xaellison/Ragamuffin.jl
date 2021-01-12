"""
Compute the stopping times of numbers subjected to the Collatz sequence.
https://en.wikipedia.org/wiki/Collatz_conjecture
"""

include("../ragamuffin.jl")

function collatz_length(x)
    if x == 1
        return 0
    elseif x % 2 == 0
        return 1 + collatz_length(x / 2)
    else
        return 1 + collatz_length(3 * x + 1)
    end
end

N = 1000
range = 1:N
expected = @time collect(map(collatz_length, range))


function ragamuffin_collatz()
    T = Int64
    program = initialize(map(T, range) |> collect, T(-1), Frame, 32, T)
    add!(program, :root, (:end, :increment), value_function=x -> x.value == 1)
    add!(program, :increment, :parity, dest_function = x -> 1)
    add!(program, :parity, (:even, :odd), value_function = x -> x.value % 2 == 0)
    add!(program, :even, value_function = x -> x.value > 0 ? Cell{T}(T(x.value ÷ 2), x.depth, x.output_index) : x)
    add!(program, :odd, value_function = x -> Cell{T}(T(x.value * 3 + 1), x.depth, x.output_index))
    @time run(program)
    return program.dest
end

batched = ragamuffin_collatz()
@info batched
@assert expected == batched
