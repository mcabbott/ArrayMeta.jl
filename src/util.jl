import Base: @pure

flatten(xs) = reduce(vcat, vec.(xs))

function indicesinvolved(expr)
    @match expr begin
        A_[idx__] => [(A => idx)]
        f_(ex__)  => reduce(vcat, [indicesinvolved(x) for x in ex])
        _ => []
    end
end


# Duplicate?
# """
# Type Domain `promote_op`
#
# The first argument is the type of the function
# """
# @generated function promote_op_t(::Type{F}, ::Type{T}) where {F,T}
#     :(_promote_op_t(F, $(T.parameters...)))
# end

# Dummy type to imitate a splatted function application
# This is passed into Inference.return_type (--> Compiler.return_type in v0.7)
struct _F{G}
    g::G
end
(f::_F{G})(x) where {G} = f.g(x...)

f(g) = x->g(x...)

# the equivalent to Base._promote_op
function _promote_op_t(F, @nospecialize T)
    G = Tuple{Base.Generator{Tuple{T},F}}
    Core.Compiler.return_type(first, G)
end

Zip2{I1,I2} = Base.Iterators.Zip{Tuple{I1,I2}}

function _promote_op_t(F, @nospecialize(R), @nospecialize(S))
    G = Tuple{Base.Generator{Zip2{Tuple{R},Tuple{S}},_F{F}}}
    Core.Compiler.return_type(first, G)
end

function _promote_op_t(F, @nospecialize(R), @nospecialize(S), @nospecialize T...)
    _promote_op_t(F, _promote_op_t(F, R, S), T...)
end

@pure function eltypes(::Type{T}) where T<:Tuple
    Tuple{map(eltype, T.parameters)...}
end

"""
Type Domain `promote_op`

The first argument is the type of the function
"""
@pure @generated function promote_op_t(::Type{F}, ::Type{T}) where {F,T}
    :(_promote_op_t(F, $(T.parameters...)))
end

"""
`promote_arraytype(F::Type, Ts::Type...)`

Returns an output array type for the result of applying a function of type `F`
on arrays of type `Ts`.
"""
@pure function promote_arraytype(::Type{F}, T...) where F
    length(T) == 1 && return T[1]
    promote_arraytype(F, promote_arraytype(F, T[1]), T[2:end]...)
end

@pure function promote_arraytype(::Type{F}, ::Type{T}, ::Type{S}) where {F, T<:Array, S<:Array}
    Array{_promote_op_t(F, eltype(T), eltype(S)), max(ndims(T), ndims(S))}
end

using SparseArrays

idxtype(::Type{SparseMatrixCSC{X,I}}) where {X,I} = I
@pure function promote_arraytype(::Type{F}, ::Type{T}, ::Type{S}) where {F, T<:SparseMatrixCSC, S<:SparseMatrixCSC}
    SparseMatrixCSC{_promote_op_t(F, eltype(T), eltype(S)),
                    promote_type(idxtype(T), idxtype(S))}
end

@pure function promote_arraytype(::Type{typeof(+)}, ::Type{T}, ::Type{S}) where {T<:Array, S<:SparseMatrixCSC}
    Array{_promote_op_t(F, eltype(T), eltype(S))}
end

@pure function promote_arraytype(::Type{typeof(*)}, ::Type{T}, ::Type{S}) where {T<:Array, S<:SparseMatrixCSC}
    SparseMatrixCSC{_promote_op_t(F, eltype(T), eltype(S)),
                    promote_type(idxtype(T), idxtype(S))}
end

@pure function promote_arraytype(::Type{typeof(+)}, ::Type{T}, ::Type{S}) where {T<:SparseMatrixCSC, S<:Array}
    Array{_promote_op_t(F, eltype(T), eltype(S))}
end

"""
`reduction_identity(f, T::Type)`

Identity value for reducing a collection of `T` with function `f`
"""
reduction_identity(f::Union{typeof(+), typeof(-)}, ::Type{T}) where {T} = zero(T)
reduction_identity(f::typeof(min), ::Type{T}) where {T} = typemax(T)
reduction_identity(f::typeof(max), ::Type{T}) where {T} = typemin(T)
reduction_identity(f::typeof(*), ::Type{T}) where {T} = one(T)
reduction_identity(f::typeof(push!), ::Type{T}) where {T} = T[]

function merge_dictofvecs(dicts...)
    merged_dict = Dict()
    for dict in dicts
        for (k, v) in dict
            if k in keys(merged_dict)
                merged_dict[k] = vcat(merged_dict[k], v)
            else
                merged_dict[k] = v
            end
        end
    end
    merged_dict
end
