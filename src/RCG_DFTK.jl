module RCG_DFTK

using DFTK
using Krylov
using LinearMaps
using Plots
using Printf
using LaTeXStrings
using LinearAlgebra
using IterativeSolvers
using TimerOutputs

include("rcg.jl")
include("lyap_solvers.jl")
include("inner_solvers.jl")
include("rcg_params.jl")
include("rcg_callbacks.jl")
include("rcg_benchmarking.jl")

export riemannian_conjugate_gradient,
    h1_riemannian_conjugate_gradient,
    energy_adaptive_riemannian_conjugate_gradient,
    energy_adaptive_riemannian_gradient
export AbstractShiftStrategy,
    ConstantShift,
    CorrectedRelativeΛShift
export AbstractGradient,
    RiemannianGradient,
    H1Gradient,
    L2Gradient,
    HessianGradient,
    EAGradient
export AbstractRetraction,
    RetractionPolar,
    RetractionQR
export AbstractTransport,
    DifferentiatedRetractionTransport,
    L2ProjectionTransport,
    RiemannianProjectionTransport,
    EAProjectionTransport,
    InverseRetractionTransport,
    NoTransport,
    default_transport
export AbstractCGParam,
    ParamZero,
    ParamDY,
    ParamFR,
    ParamFR_PRP,
    ParamHS,
    ParamHS_DY,
    ParamPRP,
    ParamRestart
export AbstractStepSize,
    ExactHessianStep,
    ApproxHessianStep,
    ConstantStep,
    AlternatingStep,
    BarzilaiBorweinStep
export AbstractBacktrackingRule,
    NonmonotoneRule,
    ArmijoRule,
    ModifiedSecantRule
export IterationStrategy,
    StandardBacktracking,
    AdaptiveBacktracking,
    NoBacktracking

export AbstractHSolver,
    NaiveHSolver,
    BandwiseHSolver,
    GlobalOptimalHSolver,
    LocalOptimalHSolver

export ResidualEvalCallback,
    ResidualEvalConverged
export AbstractEvalMethod,
    EvalRCG,
    EvalSCF,
    EvalPDCM

export RcgDefaultCallback,
    RcgConvergenceGradient,
    RcgConvergenceResidual


end # module RCG_DFTK
