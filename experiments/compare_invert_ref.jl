using DFTK
using PseudoPotentialData
using LinearAlgebra

function silicon_setup(; Ecut = 20, a = 10.26, kgrid = [2, 2, 2])
    lattice = a / 2 * [
        [0 1 1.0];
        [1 0 1.0];
        [1 1 0.0]
    ]
    Si = ElementPsp(:Si; psp = load_psp(PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"), :Si))
    atoms = [Si, Si]
    positions = [ones(3) / 8, -ones(3) / 8]

    model = model_LDA(lattice, atoms, positions)
    basis = PlaneWaveBasis(model; Ecut = Ecut, kgrid)
    return [model, basis]
end
# Silicon lattice constant in Bohr
model, basis = silicon_setup();

# Convergence tolerance
tol = 1.0e-8;

# Initial value
scfres_start = self_consistent_field(basis; tol = 1e-3, nbandsalg = DFTK.FixedBands(model));
ψ = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;

n_bands = size(ψ[1],2)

Pks = [PreconditionerTPA(basis, kpt) for kpt = basis.kpoints]
for ik = 1:length(Pks)
   DFTK.precondprep!(Pks[ik], ψ[ik])
end

rand1 = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints]
for (rand1k, ψk) = zip(rand1, ψ)
    DFTK.proj_tangent_kpt!(rand1k, ψk)
end
rand2 = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints]
for (rand2k, ψk) = zip(rand2, ψ)
    DFTK.proj_tangent_kpt!(rand2k, ψk)
end

function mul_P(δψ)
    Pδψ = [similar(δψk) for δψk = δψ]
    for ik = 1:length(Pks)
        DFTK.parallel_loop_over_range(1:size(Pδψ[ik], 2)) do n
            Pδψ[ik][:, n] .=  (Pks[ik].mean_kin[n] .+ Pks[ik].kin) .* δψ[ik][:, n]
        end
    end
    return Pδψ
end


function invert_refinement_metric(basis::PlaneWaveBasis{T}, ψ, res) where {T}
    function apply_inv_M(ψk, Pk, resk)
        resk = DFTK.proj_tangent_kpt(resk, ψk)
        Pkinv_resk = similar(ψk)
        Minv_resk = similar(ψk)
        DFTK.parallel_loop_over_range(1:size(ψk, 2)) do n
            Pkinv_resk[:, n] .= resk[:,n] ./ (Pk.mean_kin[n] .+ Pk.kin)
        end
        for n = 1:size(resk, 2)
            Pkninv_ψk = similar(ψk)
            DFTK.parallel_loop_over_range(1:size(ψk, 2)) do m
                Pkninv_ψk[:, m] .= ψk[:,m] ./ (Pk.mean_kin[n] .+ Pk.kin)
            end
            Gn = ψk'Pkninv_ψk
            Minv_resk[:, n] = Pkinv_resk[:, n] - Pkninv_ψk * (Gn \ ψk'Pkinv_resk[:, n])
        end
        return Minv_resk
    end

    map(basis.kpoints, ψ, res) do kpt, ψk, resk
        Pk = PreconditionerTPA(basis, kpt)
        DFTK.precondprep!(Pk, ψk)
        apply_inv_M(ψk, Pk, resk)
    end
end

g1 = DFTK.invert_refinement_metric(basis, ψ, rand1)
g2 = invert_refinement_metric(basis, ψ, rand1)

sum([norm(g2[ik]'ψ[ik]) for ik = 1:length(g1)])

println("ref : $(sum([dot(rand1[ik], rand2[ik]) for ik = 1:length(Pks)]))")
println("DFTK: $(sum([dot(mul_P(g1)[ik], rand2[ik]) for ik = 1:length(Pks)]))")
println("new : $(sum([dot(mul_P(g2)[ik], rand2[ik]) for ik = 1:length(Pks)]))")