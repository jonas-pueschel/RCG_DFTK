mutable struct StepSizeOptions
    step_size::String
    #backtracking
    bt_iter::UInt
    α::Float32
    β::Float32
    δ::Float32
    τ_const::Float32
    τ_0_max::Float32
end

mutable struct  GradientOptions 
    gradient::String
    #Preconditioner
    prec
    #ea_options
    inner_iter::UInt
end

mutable struct RcgOptions
    do_cg::Bool
    retraction::String
    β_cg::String
    μ::Float32
    transport_res::String
    transport_η::String
    step_size_options::StepSizeOptions
    gradient_options::GradientOptions
end



function default_options(basis, ψ)
    Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
    for ik = 1:length(Pks)
        DFTK.precondprep!(Pks[ik], ψ[ik])
    end

    step_size_options = StepSizeOptions("eH", 10, 0.95, 1e-4, 0.5, 0.0, 100.0)
    gradient_options = GradientOptions("H1", Pks, 0)
    return RcgOptions(
        true, 
        "polar",
        "FR-PRP",
        0.0,
        "diff-ret",
        "diff-ret",
        step_size_options,
        gradient_options
    )
end