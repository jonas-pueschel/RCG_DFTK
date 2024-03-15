mutable struct StepSizeOptions
    step_size::String
    #backtracking
    bt_iter::UInt
    α::Float64
    β::Float64
    δ::Float64
    τ_const::Float64
    τ_0_max::Float64
end

mutable struct  GradientOptions 
    gradient::String
    #Preconditioner
    prec
    #ea_options
    inner_iter::UInt
    shift::Float64
end

mutable struct RcgOptions
    do_cg::Bool
    retraction::String
    β_cg::String
    μ::Float64
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

    step_size_options = StepSizeOptions("eH", 10, 0.95, 1e-4, 0.5, 0.0, 5.0)
    gradient_options = GradientOptions("H1", Pks, 10, 0.0)
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