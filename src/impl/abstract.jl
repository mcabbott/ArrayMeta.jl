
using TiledIteration

### Construction of loop expressions in type domain
### This is the fallback implementation for AbstractArrays
function arrayop_body(name, ::Type{A},
                      op::Type{Assign{L,R,F,E}}) where {A<:AbstractArray, L,R,F,E}

    acc = kernel_expr(:($name.lhs), L) # :() will be ignored
    rhs_inner = kernel_expr(:($name.rhs), R)

    if hasreduceddims(op)
        # we need to wrap the expression in a call to the reducer
        expr = :($acc = $name.reducefn($acc, $rhs_inner))
    else
        expr = :($acc = $rhs_inner)
    end

    checks = :()
    input_ranges = Any[]

    simded = false

    for (sym, spaces) in index_spaces(name, op) # sort this based on # of potential cache misses
        if length(spaces) > 1
            # check dimensions for equality
            equal_dims = [:(axes($(d[3]), $(d[2]))) for d in spaces]
            checks = :($checks; @assert allequal($(equal_dims...),))
        end
        T,dim,nm = first(spaces)
        sym_range = Symbol("$(sym)_range")
        push!(input_ranges, sym_range => :(axes($nm, $dim)))
        expr = :(for $sym in $sym_range
                    $expr
                end)
        if !simded
            # innermost loop
            expr = :(@simd $expr)
            simded = true
        end
    end

    expr = quote
        ranges = ($(map(last, input_ranges)...),)
        @inbounds for tile in TileIterator(ranges, tilesize(ranges))
            ($(map(first, input_ranges)...),) = tile
            $expr
        end
    end

    :($checks; $expr; $name.lhs.array)
end

# constructs Dict
# Keys:  indexing symbol
# Value: a list of (array type, dimension, expr) tuples which correspond to that
#        indexing symbol
# Generate the expression corresponding to a type
function kernel_expr(name, itr)
    kernel_expr(name, arraytype(itr), itr)
end

function kernel_expr(name, ::Type{A},
                     itr::Type{Indexing{X, Idx}}) where {X, Idx, A<:AbstractArray}
    idx = get_subscripts(name, itr)
    :($name.array[$(idx...)])
end

function kernel_expr(name, ::Type{A},
                     itr::Type{Map{F, Ts}}) where {A<:AbstractArray, F, Ts}

    inner_kernels = [kernel_expr(:($name.arrays[$i]), arraytype(T), T)
                        for (i, T) in enumerate(Ts.parameters)]

    :($name.f($(inner_kernels...)))
end

allequal(x) = true
function allequal(x, xs...)
    x == xs[1] && allequal(xs...)
end

function tilesize(ranges)
    map(x->16, ranges)
end

@inline @generated function arrayop!(::Type{A}, t::Assign{L,R}) where {L,R,A<:AbstractArray}
    arrayop_body(:t, arraytype(L), t)
end
