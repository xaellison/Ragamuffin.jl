using Test

include("../ragamuffin.jl")

@testset "Ragamuffin" begin

"""
This verifies that Cell, Frame, and FramePool play nicely
"""
function test_frame_pool()
    # create a pool of ints whose default value is -1, and frame length = 1024
    p = FramePool{Int, Frame{Int, 1024}}(-1)
    frame = fetch!(p)
    # assert the sum of values for a fresh array makes sense
    @test sum(x.value for x in frame.cells) == -1024
    frame.cells[1] = Cell{Int}(0, -1, -1)
    # verify we can mutate elements
    @test sum(x.value for x in frame.cells) == -1023
    # verify that the pool is empty
    @test length(p) == 0
    reallocate!(p, frame)
    # verify the pool is not empty
    @test length(p) == 1
    frame = fetch!(p)
    # verify the pool is empty
    @test length(p) == 0
    # verify that all values in array were reset
    @test sum(x.value for x in frame.cells) == -1024
end

test_frame_pool()

function get_all_queued(program)
    out = counter(Int)
    for vstack in values(program.stacks)
        while !empty(vstack)
            frame = pop!(vstack, program.pool)

            for i in 1:frame.count
                push!(out, frame.cells[i].value)
            end
        end
    end
    return out
end

"""
Make sure we can initialize a program with frames whose union is the input
problem, for many frame sizes.
This test also validates VirtualStack behavior.
"""
function test_init()
    for f in 2:2:32
        input = collect(1:16)
        expected = counter(input)
        program = initialize(input, -1, Frame, f, Int)
        output = get_all_queued(program)
        @test expected == output
    end
end

test_init()

function test_split(T, decider, frame_type, frame_size, n)
    @assert n <= frame_size
    F = frame_type{T, frame_size}
    frame_pool = FramePool{T, F}(T(0))
    frame = fetch!(frame_pool)
    t_frame = fetch!(frame_pool)
    for i in 1:n
        push!(frame, Cell{T}(rand(T), 1, 1))
    end
    get_counter(frame) = counter(map(cell -> cell.value, filter(is_valid, frame.cells)))
    original_counter = get_counter(frame)

    split!(frame :: Frame{T}, t_frame :: Frame{T}, decider, frame_pool)
    # first ensure the count is right
    @test count(map(is_valid, frame.cells)) == length(frame)
    # then ensure they are all at the beginning
    @test count(map(is_valid, frame.cells[1:length(frame)])) == length(frame)
    # do the same for the other frame
    @test count(map(is_valid, t_frame.cells)) == length(t_frame)
    @test count(map(is_valid, t_frame.cells[1:length(t_frame)])) == length(t_frame)
    # finally make sure the total count of each element was preserved
    @test merge(get_counter(frame), get_counter(t_frame)) == original_counter
end

test_split(Int32, cell -> false, Frame, 32, 18)
test_split(Int32, cell -> false, Frame, 32, 32)
test_split(Int32, cell -> false, Frame, 32, 0)

test_split(Int32, cell -> true, Frame, 32, 18)
test_split(Int32, cell -> true, Frame, 32, 32)
test_split(Int32, cell -> true, Frame, 32, 0)

test_split(Int32, cell -> isodd(cell.value), Frame, 32, 18)
test_split(Int32, cell -> isodd(cell.value), Frame, 32, 32)
test_split(Int32, cell -> isodd(cell.value), Frame, 32, 0)


end
