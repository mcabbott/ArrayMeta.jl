using Dagger
import Dagger:DomainBlocks

import Dagger: DArray, chunks


function onchunks(X::Indexing)
    # TODO: handle IndexConsts
    idx = X.idx
    Indexing(map(delayed(x -> Indexing(x, idx)), chunks(X.array)), idx)
end

function onchunks(X::Map)
    f = X.f
    Map(delayed((x...) -> Map(f, x)), map(onchunks, X.arrays))
end

function arrayop!(::Type{D}, t::Assign) where D<:DArray
    lhs = onchunks(t.lhs)
    rhs = onchunks(t.rhs)

    idx = t.lhs.idx
    combine_chunks = delayed() do a, b
        Map(t.reducefn, (a, b))
    end
    ct = Assign(lhs, rhs, combine_chunks, delayed(()->nothing))
    cs = arrayop_treereduce(ct)
    f = delayed() do l, r
        arrayop!(Assign(l, r, t.reducefn, t.empty))
    end
    t.lhs.array.chunks = map(f, onchunks(t.lhs).array, cs)
    t.lhs.array
end

@generated function arrayop_treereduce(op::Assign{L,R,F,E}) where {L,R,F,E}

    rspaces = index_spaces(:(op.rhs), R)
    lspaces = index_spaces(:(op.lhs), L)
    expr = kernel_expr(:(op.rhs), R)

    reduceddims = setdiff(keys(rspaces), keys(lspaces))
    for sym in reduceddims
        spaces = rspaces[sym]
        T,dim,nm = first(spaces)
        valx = gensym("valx")
        valy = gensym("valy")
        expr = quote
            Dagger.treereduce(indices($nm, $dim)) do x, y
                x, y
                $sym = x

                $valx = $expr
                $sym = y
                $valy = $expr
                op.reducefn($valx, $valy)
            end
        end
    end

    lhs_expr = kernel_expr(:(op.lhs), L) # :() will be ignored
    expr = :($lhs_expr = $expr)

    checks = :()
    for (sym, spaces) in lspaces
        if length(spaces) > 1
            # check dimensions for equality
            equal_dims = [:(size($(d[3]), $(d[2]))) for d in spaces]
            checks = :($checks; @assert allequal($(equal_dims...),))
        end
        T,dim,nm = first(spaces)
        expr = quote
            for $sym = indices($nm, $dim)
                $expr
            end
        end
    end

    :($checks; $expr; op.lhs.array)
end

function Base.indices(x::DArray)
    Dagger.domainchunks(x)
end

function Base.indices(x::DArray, i)
    idxs = indices(x)
    Dagger.DomainBlocks((idxs.start[i],), (idxs.cumlength[i],))
end

function allocarray(::Type{DArray{T,N}}, default, idxs...) where {T,N}
    dmnchunks = DomainBlocks(map(i->1, idxs),
                             map(i->isa(i, DomainBlocks) ?
                                 i.cumlength[1] : (length(i),), idxs))

    chnks = map(delayed(subd -> allocarray(Array{T,1}, default, size(subd))), dmnchunks)
    sz = map((x,y)->x-y+1, map(last, dmnchunks.cumlength), dmnchunks.start)
    dmn = ArrayDomain(map(x->1:x, sz))
    DArray{T, length(idxs)}(dmn, dmnchunks, chnks)
end
