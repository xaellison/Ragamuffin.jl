using DataStructures, Logging

include("structs.jl")
"""
#Ideally, syntax will eventually look something like this (for Collatz)
@init 1:1024
@branch x -> x == 1 :end
@out_function += 1 :add
@branch x -> x % 2 == 0 :even :odd
@function :even x -> x / 2
@function :odd x -> x * 3 + 1
@run
"""


#--- Program Building
"""
Return a `Program` object to run.
`values` - initial starting problem
`default_value` - effectively a null value for frames which aren't full
`frame_type` - either `Frame` for CPU running or `CuFrame` for GPU
`frame_size` - how many elements in frame
`output_type` - what type output array holds (it'll be same size as `values`)
"""
function initialize(values :: Array{T}, default_value :: T, frame_type, frame_size, output_type :: Type) where T
    @debug "Initializing..."
    F = frame_type{T, frame_size}
    pool = FramePool{T, F}(default_value)
    frame_dict = Dict(:root => VirtualStack{T, F}(Int(round(frame_size * 0.5))))
    output = Array{Int32}(undef, length(values)) .* 0
    program = Program{T, output_type, F}(frame_dict,
                      Dict{Symbol, Function}(),
                      Dict{Symbol, Function}(),
                      Dict{Symbol, Union{Symbol, Tuple{Symbol, Symbol}}}(),
                      pool, output)
    # TODO: implement constructor so scalar access can always be disallowed
    @debug "initializing root"
    for floor in 1:frame_size:length(values)

        ceil = min(frame_size + floor - 1, length(values))
        @debug "Fetching frame"
        frame = fetch!(pool)

        if F == Frame{T, frame_size}
            @debug "cpu frame"
            for (frame_index, value_index) in enumerate(floor:ceil)
                frame[frame_index] = Cell{T}(values[value_index], 0, value_index)
            end
        elseif F == CuFrame{T, frame_size}
            @debug "gpu frame"
            temp = collect(Cell{T}(values[i], 0, i) for i in floor:ceil)
            @debug "temp made"
            @time frame.cells[1:length(floor:ceil)] .= CuArray(temp)
        else
            return error("Unknown type")
        end
        frame.count = length(floor:ceil)
        push!(program, :root, frame)
    end

    @debug "ready!"
    return program
end

"""
Modify program by adding an edge in the process graph between `src` and `dest`
`dest` can be a single symbol or a tuple for branching
`value_function` can be a function or tuple of function and array of aux vars
"""
function add!(program :: Program{T, U, F}, src, dest=:root; value_function=nothing, dest_function=nothing) where {T, U, F}
    if ! isnothing(value_function)
        program.value_functions[src] = value_function
    end
    if ! isnothing(dest_function)
        program.dest_functions[src] = dest_function
    end

    if dest != nothing
        program.flow[src] = dest

        if !isa(dest, Tuple)
            dest = (dest,)
        end
        for d in dest
            if ! (d in keys(program.stacks))
                sz = size(program.pool)
                program.stacks[d] = VirtualStack{T, F}(floor(sz / 2))
            end
        end
    end
end

#---- Program Execution
"""
Get the name of the stack to run a frame from. If the program is done,
return :PROGRAM_TERMINATED
"""
function get_stack_name(program :: Program) :: Symbol
    for i in keys(program.stacks)
        if ready(program.stacks[i])
            return i
        end
    end
    for i in keys(program.stacks)
        if ! empty(program.stacks[i])
            return i
        end
    end
    return :PROGRAM_TERMINATED
end


function value_eval(program, dest, frame :: CuFrame, f_d :: Tuple{Function, Array})
    # break out function and domain
    f, domain = f_d
    for d in domain
        dest_frame = fetch!(program.pool)
        dest_frame.cells .= map(x -> f(x, d), frame.cells)
        dest_frame.count = sum(map(is_valid, frame.cells))
        push!(program, dest, dest_frame)
    end
    reallocate!(program.pool, frame)
end

function value_eval(program, dest, frame :: Frame, f_d :: Tuple{Function, Array})
    # break out function and domain
    f, domain = f_d
    for d in domain
        dest_frame = fetch!(program.pool)
        for i in 1:length(frame)
            push!(dest_frame, f(frame[i], d))
        end
        push!(program, dest, dest_frame)
    end
    reallocate!(program.pool, frame)
end

function value_eval(program, dest, frame :: AFrame, f :: Function)
    # This is a straight function evaluation, so the frame is moved from src
    # to dest without involving the pool
    # It's agnostic to the type of frame
    map!(f, frame.cells, frame.cells)
    push!(program, dest, frame)
end

function dest_eval(program, dest, frame :: AFrame, f)
    # This will copy off device if a CuArray, or do nothing if already Array
    cells = Array(frame.cells)
    for i in 1:length(frame)
        program.dest[cells[i].output_index] += cells[i] |> f
    end
end

function execute_batch(program, src, dest :: Symbol)
    frame = pop!(program, src)
    @debug "Frame $(frame.cells)"
    if src in keys(program.dest_functions)
        dest_eval(program, dest, frame, program.dest_functions[src])
    end
    if src in keys(program.value_functions)
        value_eval(program, dest, frame, program.value_functions[src])
    else
        push!(program, dest, frame)
    end
end

"""
For a cpu frame, compute `decider` for each element and move to `t_frame` if
true and to `f_frame` if false
"""
function split!(frame :: Frame{T}, t_frame :: Frame{T}, decider, frame_pool) where T
    sort!(frame.cells; by=x -> (!is_valid(x), decider(x)))
    #map(x -> x.value, frame.cells)
    partition = findfirst(cell -> is_valid(cell) && decider(cell), frame.cells )
    if isnothing(partition)
        # nothing to move to t_frame
    elseif partition == 1
        # swap
        frame.cells, t_frame.cells = t_frame.cells, frame.cells
        frame.count, t_frame.count = t_frame.count, frame.count
        t_frame.age = frame.age
    else
        n_max = frame.count
        n_false = partition - 1
        n_true = n_max - n_false
        t_frame.cells[1:n_true] .= frame.cells[partition:n_max]
        @simd for i in partition:n_max
            frame.cells[i] = Cell{T}(frame_pool.default_value, 0, -1)
        end
        frame.count = n_false
        t_frame.count = n_true
    end
end

"""
For a GPU frame, compute `decider` for each element and move to `t_frame` if
true and to `f_frame` if false.

Does two boolean sorts: the first for whether a cell is a valid value, the
second `decider`. After those, the frame is all invalid values, followed by all
false, then all true. Once that is done, the false and trues are distributed to
destinations `t_frame` and `f_frame`.

Assumes that every thread of a SM can hold a Cell{T} and an Int32 worth of shmem
"""
function split!(frame :: CuFrame{T, L}, t_frame :: CuFrame, f_frame :: CuFrame, decider) where {T, L}
    N_t = 1024
    N_b = Int(L / N_t)
    flags = map(is_valid, frame.cells)
    indices = CUDA.zeros(Int32, L)
    batch_sums = CUDA.zeros(Int32, N_b)
    @cuda blocks=N_b threads=N_t index_sort_kernel(indices, flags, batch_sums)
    @cuda blocks=N_b threads=N_t index_move_kernel(frame.cells, indices)

    batch_valid_counts = Array(batch_sums)
    flags = map(decider, frame.cells)

    @cuda blocks=N_b threads=N_t index_sort_kernel(indices, flags, batch_sums)
    @cuda blocks=N_b threads=N_t index_move_kernel(frame.cells, indices)

    batch_true_counts = Array(batch_sums)
    t_floor = 0
    f_floor = 0
    for b in 1:N_b
        floor = (b - 1) * N_t
        valid_count = batch_valid_counts[b]
        invalid_count = N_t - valid_count
        true_count = batch_true_counts[b]
        false_count = valid_count - true_count
        if true_count > 0
            t_frame.cells[t_floor+1:t_floor+true_count] .= frame.cells[floor + (N_t - true_count + 1):floor + (N_t)]
            t_frame.count += true_count
        end
        if false_count > 0
            f_frame.cells[f_floor+1:f_floor+false_count] .= frame.cells[floor+invalid_count + 1: floor+invalid_count + false_count]
            f_frame.count += false_count
        end
        t_floor += true_count
        f_floor += false_count
    end
end

function execute_batch(program :: Program{T, U}, src, dest :: Tuple{Symbol, Symbol}) where {T, U}
    t_frame = fetch!(program.pool)
    frame = pop!(program, src)
    #@debug "Frame count: $(length(frame))"
    split!(frame, t_frame, program.value_functions[src], program.pool)
    #reallocate!(program.pool, frame)
    if length(t_frame) > 0
        push!(program, dest[1], t_frame)
    else
        reallocate!(program.pool, t_frame)
    end
    if length(frame) > 0
        push!(program, dest[2], frame)
    else
        reallocate!(program.pool, frame)
    end
end


function Base.run(program :: Program)
    stack_name = get_stack_name(program)
    while stack_name != :PROGRAM_TERMINATED
        execute_batch(program, stack_name, program.flow[stack_name])
        stack_name = get_stack_name(program)
    end
end
