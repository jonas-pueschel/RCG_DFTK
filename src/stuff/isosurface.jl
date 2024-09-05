using Pkg
Pkg.activate("./src")
Pkg.instantiate()

using DFTK
using LinearAlgebra
using Krylov
#using Plots

include("./rcg.jl")
include("./setups/all_setups.jl")
include("./rcg_benchmarking.jl")

mol = "GaAs"

if mol == "GaAs"
    Ecut = 80
    a = 10.68290949909  # GaAs lattice constant in Bohr
    lattice = a / 2 * [[0 1 1.];
                    [1 0 1.];
                    [1 1 0.]]

    Ga = ElementPsp(:Ga, psp=load_psp("hgh/lda/ga-q3"))
    As = ElementPsp(:As, psp=load_psp("hgh/lda/as-q5"))
    atoms = [Ga, As]
    positions = [ones(3)/8 #+ [0.24, -0.33, 0.12] / 15
                    , -ones(3)/8]

    model = model_LDA(lattice, atoms, positions)
    kgrid = [2, 2, 2]  # k-point grid (Regular Monkhorst-Pack grid)
    ss = 4
    fft_size = compute_fft_size(model, Ecut; supersampling=ss)
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid, fft_size=fft_size)
elseif mol == "TiO2"
    a = 6.0
    Ecut = 50
    Ti = ElementPsp(:Ti, psp=load_psp("hgh/lda/ti-q4.hgh"))
    O  = ElementPsp(:O, psp=load_psp("hgh/lda/o-q6.hgh"))
    atoms     = [Ti, Ti, O, O, O, O]
    positions = [[0.5,     0.5,     0.5],  # Ti
                 [0.0,     0.0,     0.0],  # Ti
                 [0.19542, 0.80458, 0.5],  # O
                 [0.80458, 0.19542, 0.5],  # O
                 [0.30458, 0.30458, 0.0],  # O
                 [0.69542, 0.69542, 0.0]]  # O
    #positions[1] .+= [0.22, -0.28, 0.35] / 5
    lattice   = [[8.79341  0.0      0.0];
                 [0.0      8.79341  0.0];
                 [0.0      0.0      5.61098]];
    
    model = model_LDA(lattice, atoms, positions)
    kgrid = [2, 2, 2]  # k-point grid (Regular Monkhorst-Pack grid)
    ss = 4
    fft_size_ref = compute_fft_size(model, Ecut; supersampling=ss)
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid, fft_size=fft_size_ref)
elseif mol == "graphene"
    Ecut = 100
    L = 20  # height of the simulation box
    kgrid = [9, 9, 1]
    # Define the geometry and pseudopotential
    a = 4.66  # lattice constant
    a1 = a*[1/2,-sqrt(3)/2, 0]
    a2 = a*[1/2, sqrt(3)/2, 0]
    a3 = L*[0  , 0        , 1]
    lattice = [a1 a2 a3]
    C1 = [1/3,-1/3,0.0]  # in reduced coordinates
    C2 = -C1
    positions = [C1, C2]
    C = ElementPsp(:C; psp=load_psp("hgh/pbe/c-q4"))
    atoms = [C, C]

    # Run SCF
    model = model_PBE(lattice, atoms, positions)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
end

DFTK.reset_timer!(DFTK.timer)
        scfres_scf = self_consistent_field(basis; tol = 1e-8,
        maxiter = 100);print("");
println(DFTK.timer)

values_raw = scfres_scf.ρ[:,:,:,1]

using PlotlyJS


dataX = range(-1 * a, stop= 1 * a, length=120)
dataY = range(- 1 * a, stop= 1 * a, length=120)
dataZ = range(- 0.5 * a, stop= 0.5 * a, length=60)
X, Y, Z = mgrid(dataX, dataY, dataZ)



# rotation = [[sqrt(2.0)/2.0 -sqrt(2.0)/2.0 0.];
#             [sqrt(2.0)/2.0 sqrt(2.0)/2.0 0.];
#             [0. 0. 1.]]

rotation  = I(3)

# for i = 1:length(X)
#     X[i], Y[i], Z[i] = lattice * [X[i], Y[i], Z[i]]
# end
# ellipsoid

Xs = X[:]
Ys = Y[:]
Zs = Z[:]

function eval_cube(vals, coords)
    vals1 = coords[1] * vals[2, :, :] + (1- coords[1]) * vals[1, :, :]
    vals2 = coords[2] * vals1[2,  :] + (1- coords[2]) * vals1[1, :]
    return coords[3] * vals2[2] + (1- coords[3]) * vals2[1]
end

function get_val(x,y,z, values_raw)
    #get lattice coordinates

    xl, yl, zl = lattice\(rotation * [x,y,z])

    #unit cell
    coords_fl = xl%1, yl%1, zl%1
    coords_fl = [c >= 0 ? c : c + 1 for c = coords_fl]


    coords_floor = [mod(Int(floor(coords_fl[i] * (size(values_raw)[i]))),size(values_raw)[i]) + 1 for i = 1:3]
    coords_ceil = [mod(Int(ceil(coords_fl[i] * (size(values_raw)[i]))),size(values_raw)[i]) + 1 for i = 1:3]
    coords_offset = [mod(coords_fl[i] * (size(values_raw)[i]),size(values_raw)[i]) + 1 for i = 1:3] - coords_floor

    # 8 -> 4 -> 2 -> 1
    vals = zeros(2,2,2)
    for i=1:2, j = 1:2, k = 1:2
        ijk = [i, j, k]
        coords = [ijk[l] == 0 ? coords_floor[l] : coords_ceil[l] for l = 1:3]
        vals[i,j,k] = values_raw[coords...]
    end

    return eval_cube(vals, coords_offset)    
end

println("calculating values")

vals = [get_val(x,y,z, values_raw) for (x,y,z) = zip(Xs, Ys, Zs)]

println("finished calculating...")



plot(isosurface(
    x=Xs,
    y=Ys,
    z=Zs,
    value=vals,
    isomin=0.025,
    surface_count=2, # number of isosurfaces, 2 by default: only min and max
    colorscale=colors.RdBu_3,
    caps=attr(
        #x_show=false, y_show=false,
         #z_show = false
         )
    ))
