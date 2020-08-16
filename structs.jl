using CUDA, DataStructures, StaticArrays

CUDA.allowscalar(false)

struct Cell{T}
    value :: T
    depth :: Int32
    output_index :: Int32
end

function Cell(cell :: Cell)
    return cell
end

function is_valid(cell :: Cell)
    return cell.output_index >= 1
end

abstract type AFrame{T, L} end

function Base.getindex(f :: AFrame{T, L}, i) :: Cell{T} where {T, L}
    return f.cells[i]
end

function Base.setindex!(f :: AFrame{T, L}, val :: Cell{T}, i) where {T, L}
    f.cells[i] = val
end

mutable struct Frame{T, L} <: AFrame{T, L}
    cells :: MVector{L, Cell{T}}
    count :: Int32
    age :: Int32
end

function Base.push!(f :: Frame{T, L}, val :: Cell{T}) where {T, L}
    i = length(f)
    f[i + 1] = val
    f.count += 1
end

mutable struct CuFrame{T, L} <: AFrame{T, L}
    cells :: CuArray{Cell{T}}
    count :: Int32
    age :: Int32
end

struct FramePool{T, F}
    pool :: Array{F}
    default_value :: T
end

struct VirtualStack{T, F}
    ready :: PriorityQueue{F, Int32}
    partial :: Array{F}
    flags :: Array{Bool}
    threshold :: Int32
end

struct Program{T, U, F}
    stacks :: Dict{Symbol, VirtualStack{T, F}}
    value_functions :: Dict{Symbol, Union{Function, Tuple{Function, Any}}}
    dest_functions :: Dict{Symbol, Function}
    flow :: Dict{Symbol, Union{Symbol, Tuple{Symbol, Symbol}}}
    pool :: FramePool{T, F}
    dest :: Array{U}
end

function FramePool{T, F}(default_value :: T) where {T, F}
    return FramePool{T, F}(Array{F}(undef, 0), default_value)
end

function fetch!(pool :: FramePool{T, F}) :: F where {T, L, F <: AFrame{T, L}}

    if length(pool.pool) > 0
        #@debug "Reusing array"
        return pop!(pool.pool)
    else
        #@debug "Creating new array"
        arr = MVector{L, Cell{T}}(undef)
        @simd for i in 1:length(arr)
            arr[i] = Cell{T}(pool.default_value, 0, -1)
        end

        out = F(arr, 0, 0)
        return out
    end
end

function realloc_kernel(cells, value :: T, N) where T
    i = threadIdx().x
    for iter in 1:N
        cells[i + (iter - 1) * 1024] = Cell{T}(value, 0, -1)
    end
    return nothing
end

function reallocate!(frame_pool :: FramePool{T, F}, frame :: CuFrame{T, L}) where {T, F, L}
    @cuda threads=1024 realloc_kernel(frame.cells, frame_pool.default_value, Int(L / 1024))
    frame.count = 0
    push!(frame_pool.pool, frame)
end

function reallocate!(frame_pool :: FramePool{T, F}, frame :: Frame{T, L}) where {T, F, L}
    #@debug "Reallocating array"
    @simd for i in 1:L
        frame.cells[i] = Cell{T}(frame_pool.default_value, 0, -1)
    end
    frame.count = 0
    push!(frame_pool.pool, frame)
end

function Base.length(frame :: AFrame)
    return frame.count
end

function Base.size(frame :: AFrame{T, L}) where {T, L}
    return L
end

function Base.size(frame_pool :: FramePool{T, F}) where {T, L, F <: AFrame{T, L}}
    return L
end

function Base.length(frame_pool :: FramePool)
    return length(frame_pool.pool)
end

function ready(stack :: VirtualStack)
    return length(stack.ready) > 0
end

function Base.empty(stack :: VirtualStack)
    return length(stack.ready) == length(stack.partial) == 0
end

function VirtualStack{T, F}(threshold) where {T, F}
    return VirtualStack{T, F}(PriorityQueue{F, Int32}(),
                           Array{F}(undef, 0),
                           Array{Bool}(undef, 0), threshold)
end

function collapse!(stack, frame_pool)
    if length(stack.partial) == 0
        return
    end
    frame = pop!(stack.partial)
    cum_sum = frame.count
    new_age = frame.age
    #@debug "attempting collapse"
    while length(stack.partial) > 0 && cum_sum + stack.partial[end].count <= size(frame_pool)
        temp = pop!(stack.partial)
        new_age = min(new_age, temp.age)
        frame.cells[cum_sum + 1: cum_sum + temp.count] .= temp.cells[1:temp.count]
        cum_sum += temp.count
    #    @debug "collapsed 1x"
        reallocate!(frame_pool, temp)
    end

    frame.count = cum_sum
    enqueue!(stack.ready, frame, new_age)
end

function attempt_collapse!(stack :: VirtualStack, frame_pool)
    if length(stack.partial) > 0 && sum(p.count for p in stack.partial) > stack.threshold
        collapse!(stack, frame_pool)
    end
end

function Base.pop!(stack :: VirtualStack{T, F}) :: F where {T, F}
    return dequeue!(stack.ready)
end


function Base.pop!(stack :: VirtualStack{T, F}, frame_pool :: FramePool{T, F}) :: F where {T, F}
    if ready(stack)
        return pop!(stack)
    end
    collapse!(stack, frame_pool)
    return pop!(stack)
end

function Base.pop!(program :: Program{T, U, F}, stack_name :: Symbol) where {T, U, F}
    pop!(program.stacks[stack_name], program.pool)
end


function Base.push!(program :: Program{T, U, F}, key :: Symbol, frame :: F) where {T, U, F}
    if frame.count == 0
        reallocate!(program.pool, frame)
        return
    end
    if key == :end
        reallocate!(program.pool, frame)
        return
    end

    stack = program.stacks[key]
    if frame.count >= stack.threshold
        enqueue!(stack.ready, frame, frame.age)
    else
        push!(stack.partial, frame)
        attempt_collapse!(stack, program.pool)
    end
end
