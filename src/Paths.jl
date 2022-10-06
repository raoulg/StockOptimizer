module Pathlib

export Path, hassuffix

"""
splits at /, gets the last item
"""
getname(p::String) = last(split(p, "/"))

"""
Splits string at points, takes last part
"""
getsuffix(p::String) = occursin(".", p) ? last(split(p, ".")) : ""

#
struct Path
    path::String
    name::String
    suffix::String
    Path(p::String) = new(p, getname(p), getsuffix(p))
end

Base.:/(a::Path, b::Path) = Path(a.path*"/"*b.path)
Base.:/(a::Path, b::String) = Path(a.path*"/"*b)
Base.:(==)(a::Path, b::String) = a.path == b
Base.show(io::IO, p::Path) = print(io, p.path)


function hassuffix(f::String, suffix::String)
    return split(f, ".")[2] == suffix
end

function hassuffix(f::Path, suffix::String)
    return f.suffix == suffix
end

end