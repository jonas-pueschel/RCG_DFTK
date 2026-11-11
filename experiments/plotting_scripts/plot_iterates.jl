using LinearAlgebra
using Random

# Trilinear interpolation of the periodic density ρ on the square base + s*u + t*v, s,t ∈ [0,1]
# (Cartesian coordinates). u and v must be orthogonal and of equal length.
# Default: square in the xy-plane through the origin, ~1.5 x 1.5 unit cells for a cubic lattice.
function get_rho(ρ, lattice; n_pix = 400, base = zeros(3),
                 u = [1.5 * cbrt(abs(det(lattice))), 0, 0],
                 v = [0, 1.5 * cbrt(abs(det(lattice))), 0])
    @assert abs(dot(u, v)) ≤ 1e-8 * norm(u) * norm(v) "u and v must be orthogonal"
    @assert isapprox(norm(u), norm(v); rtol = 1e-8) "u and v must have the same length"

    ρ1 = view(ρ, :, :, :, 1)
    N = size(ρ1)

    # fractional coordinates are affine in (s, t), so only three solves are needed
    f0, fu, fv = lattice \ base, lattice \ u, lattice \ v

    function get_val(s, t)
        x = ntuple(k -> mod((f0[k] + s * fu[k] + t * fv[k]) * N[k], N[k]), 3)
        i0 = ntuple(k -> mod(floor(Int, x[k]), N[k]) + 1, 3)
        i1 = ntuple(k -> mod(i0[k], N[k]) + 1, 3)
        w = ntuple(k -> x[k] - floor(x[k]), 3)

        val = zero(eltype(ρ1))
        for (a, wa) in ((i0[1], 1 - w[1]), (i1[1], w[1])),
            (b, wb) in ((i0[2], 1 - w[2]), (i1[2], w[2])),
            (c, wc) in ((i0[3], 1 - w[3]), (i1[3], w[3]))
            val += wa * wb * wc * ρ1[a, b, c]
        end
        return val
    end

    ts = range(0, 1, length = n_pix)
    return [get_val(s, t) for s in ts, t in ts]
end

function calculate_energy_distribution(ψ, basis)
    amplitudes = []
    e_kins = []

    for (kpt, ψk) = zip(basis.kpoints, ψ)

        Pk = PreconditionerTPA(basis, kpt)
        for i = 1:size(ψk)[1]
            push!(amplitudes, sqrt(sum(abs(ψk[i, j])^2 for j = 1:size(ψk)[2])))
            #probably one can push the whole Pk.kin at once
            push!(e_kins, Pk.kin[i])
        end
    end
    return amplitudes, e_kins
end

# scatter plot of the amplitudes as over |k+G| = √(2 es) (√ scale in the energy), plotting only a
# subset: the plane is divided into cells (nx columns up to the largest |k+G|, rows of dy decades of
# the amplitude) and every point is kept with probability per_cell / (number of points in its cell).
# Sparse regions (low energies, outliers) are thus plotted completely, dense ones (high energies)
# thinned out to about per_cell points per cell, which is what is visible anyway.
# The random numbers are fixed per point and the cells do not depend on the amplitudes: for similar
# data (e.g. subsequent iterates on the same basis) mostly the same points are plotted, such that the
# pattern of the plot only changes where the data does. Points are drawn in a fixed random order (in
# the order of the data, the points of dense regions would be drawn below the others, which shows up
# as streaks). Ecuts sets the x-ticks, other kwargs go to scatter
function smart_scatter(as, es; Ecuts = nothing, nx = 400, dy = 0.04, per_cell = 2, kw...)
    as = Float64.(as)
    xs = sqrt.(2 .* Float64.(es))
    xhi = max(maximum(xs), eps())
    valid = [a > 0 && isfinite(a) for a in as]  # no zeros on the log scale
    cell(i) = (min(floor(Int, xs[i] / xhi * nx), nx - 1), floor(Int, log10(as[i]) / dy))
    counts = Dict{Tuple{Int, Int}, Int}()
    for i in eachindex(as)
        valid[i] && (counts[cell(i)] = get(counts, cell(i), 0) + 1)
    end

    u = rand(Xoshiro(0), length(as))
    keep = [i for i in sortperm(u) if valid[i] && u[i] * counts[cell(i)] < per_cell]

    ticks = isnothing(Ecuts) ? (;) : (; xticks = (sqrt.(2 .* round.(Int, Ecuts)), string.(round.(Int, Ecuts))))
    sc = scatter(xs[keep], as[keep]; yscale = :log10, xlabel = "E_kin (√ scale)", legend = false, ticks..., kw...)
    return (sc, es[keep], xs[keep], as[keep])
end

# function plot_energy_distribution(ψ, basis, Ecuts)
#     as, es = calculate_energy_distribution(ψ, basis)
# end


mutable struct PlotDiffCallback
    name 
    model_name
    ψ_ref
    ρ_ref
    basis_ref
    Ecuts
    default_callback
    n_iter 
    norm_diffs
    plotting
    img_path
    data_path
    heatmap_lims
    function PlotDiffCallback(name, model_name, ψ_ref, ρ_ref, basis_ref, Ecuts; 
            default_callback = (info) -> nothing, 
            plotting = [:density, :distribution],
            img_path = "./experiments/imgs/",
            data_path = "./experiments/data/",
            heatmap_lims = nothing)
        return new(name, model_name, ψ_ref, ρ_ref, basis_ref, Ecuts, default_callback, 0, [], 
            plotting, img_path, data_path, heatmap_lims)
    end
end

# heatmap with a log10 colorbar; cticks sets the colorbar ticks, other kwargs go to the main heatmap
# lims = (min, max) (in the units of A, not log10) fixes the color range, e.g. to share it between
# plots; values outside are drawn in the colors of the limits. Default: extrema of A
function logheatmap(A; cticks = nothing, color = :inferno, dpi = 100, lims = nothing, kw...)
    lo, hi = log10.(something(lims, extrema(A)))
    cticks = something(cticks, round.(exp10.(range(lo, hi, 5)), sigdigits = 2))
    cvals = range(lo, hi, 256)
    main = heatmap(log10.(A); clims = (lo, hi), color, colorbar = false, grid = false, kw...)
    cbar = heatmap([0, 1], cvals, repeat(cvals, 1, 2); clims = (lo, hi), color, colorbar = false,
                   grid = false, xticks = false, ymirror = true,
                   yticks = (log10.(cticks), string.(cticks)), ylims = (lo, hi), framestyle = :box)
    plot(main, cbar; layout = grid(1, 2, widths = [0.93, 0.07]), dpi)
end

function (cb::PlotDiffCallback)(info)
    (info.stage == :finalize) && return cb.default_callback(info)
    cb.n_iter += 1

    title = "$(cb.name)-$(cb.n_iter)"
    # option A: extrapolate ρ
    # option B: recalculate density

    ψ_iter = hasfield(typeof(info), :occupation) ? RCG_DFTK.interpolate_c2f(info.basis, cb.basis_ref, 
        DFTK.select_occupied_orbitals(info.basis, info.ψ, info.occupation).ψ) : RCG_DFTK.interpolate_c2f(info.basis, cb.basis_ref, 
        info.ψ)

    ρ_iter = RCG_DFTK.compute_density(cb.basis_ref, ψ_iter)

    coarse_ψ = RCG_DFTK.interpolate_f2c(info.basis, cb.basis_ref, cb.ψ_ref)
    diff = RCG_DFTK.grassmann_diff_DFTK(cb.ψ_ref, ψ_iter)
    coarse_err = RCG_DFTK.grassmann_diff_DFTK(info.ψ, coarse_ψ)
    push!(cb.norm_diffs, RCG_DFTK.norm_DFTK(cb.basis_ref, diff))

    if :density in cb.plotting
        rho = get_rho(abs.(ρ_iter .- cb.ρ_ref), cb.basis_ref.model.lattice)
        hmp = logheatmap(rho ; color=:magma, dpi = 300, title=title, lims = cb.heatmap_lims,
            xaxis=false, yaxis=false, aspect_ratio = :equal, #colorbar=:none,
            showaxis = false, ticks= false)
        
        save_png(hmp, "$(cb.img_path)$(cb.model_name)-densityerr-$(title).png")
        display(hmp)
    end

    if :distribution in cb.plotting
        plot_diff(diff, cb.basis_ref, cb.Ecuts; 
            name = "$(cb.model_name)-distribution-$(title)",
            img_path = cb.img_path,
            data_path = cb.data_path)
    end

    cb.default_callback(info)
end


# all files are saved via these two functions, which create the directory of the file if necessary
# (file names may also contain subdirectories)
function save_png(plt, file)
    mkpath(dirname(file))
    savefig(plt, file)
end

function save_distribution(file, es, xs, as)
    mkpath(dirname(file))
    open(file, "w") do io
        println(io, "Ekins normGks amplitudes")
        writedlm(io, [es xs as], ' ')
    end
end

function plot_diff(diff, basis_ref, Ecuts; name = nothing, img_path = "./experiments/imgs/", data_path = "./experiments/data/")
    as, es = calculate_energy_distribution(diff, basis_ref)
    (err_distribution, es, xs, as) = smart_scatter(as, es; Ecuts, dpi = 300, ylims = (1e-10, 1.0))
    display(err_distribution)

    if !isnothing(name)
        save_png(err_distribution, "$(img_path)$name.png")
        save_distribution("$(data_path)$name.dat", es, xs, as)
    end

    return err_distribution
end


function plot_ψ(ψ, basis, Ecuts; name = nothing, img_path = "./experiments/imgs/",     data_path = "./experiments/data/") 
    as, es = calculate_energy_distribution(ψ, basis)
    # second palette color (orange), to tell iterates apart from the differences (first color)
    (distribution, es, xs, as) = smart_scatter(as, es; Ecuts, dpi = 300, color = palette(:default)[2])
    
    if !isnothing(name)
        save_png(distribution, "$(img_path)$name.png")
        save_distribution("$(data_path)$name.dat", es, xs, as)
    end

    display(distribution)
    return distribution
end

function plot_diff_hmp(ψ, basis, ρ_ref, basis_ref, title = "difference", rhoargs...) 
    ψ_fine = RCG_DFTK.interpolate_c2f(basis, basis_ref, ψ)
    ρ_iter = RCG_DFTK.compute_density(basis_ref, ψ_fine)

    rho = get_rho(abs.(ρ_iter .- ρ_ref), basis_ref.model.lattice;rhoargs...)
    hmp = logheatmap(rho ; color=:magma, dpi = 300, title=title, 
        xaxis=false, yaxis=false, aspect_ratio = :equal, #colorbar=:none,
        showaxis = false, ticks= false)

    display(hmp)
    return hmp
end
