marks = ["diamond", "square", "triangle", "o", "asterisk", "pentagon", "Mercedes star", "|", "|", "|"]

function generate_tikz_plots(molecule, resls, times, cost_hams, method_names, init_res, tol)

    legend_str = replace(method_names[1], "_" => "\\_");
    for method_name = method_names[2:end]
        method_name = replace(method_name, "_" => "\\_");
        legend_str *= ", $method_name"
    end

    st1 = to_points_string([nothing for i = 1:length(resls)], resls, init_res);
    st2 = to_points_string(times, resls, init_res);
    st3 = to_points_string(cost_hams, resls, init_res);

    st = "
    \\begin{figure}
        \\centering
    \\begin{tikzpicture}[trim axis left]
    \\begin{axis}[
    width = {0.95\\textwidth},
    height = 6.5cm,
            legend style={nodes={scale=0.75, transform shape}},
            ymode = log,
        xlabel={iterations},
        ylabel={norm residual},
    ymin=$tol, ymax=1,
        enlargelimits=0.05,
        legend pos=north east,
        grid style=dashed,
    ]
    $st1
    \\legend{$legend_str};
    \\end{axis}
    \\end{tikzpicture} \\\\

    \\begin{tikzpicture}[trim axis left]
    \\begin{axis}[
    width = {0.95\\textwidth},
    height = 6.5cm,
            legend style={nodes={scale=0.75, transform shape}},
            ymode = log,
        xlabel={time (ns)},
        ylabel={norm residual},
    ymin=$tol, ymax=1,
        enlargelimits=0.05,
        legend pos=north east,
        grid style=dashed,
    ]
    $st2
    \\legend{$legend_str};
    \\end{axis}
    \\end{tikzpicture} \\\\

    \\begin{tikzpicture}[trim axis left]
    \\begin{axis}[
    width = {0.95\\textwidth},
    height = 6.5cm,
            legend style={nodes={scale=0.75, transform shape}},
            ymode = log,
        xlabel={estimated cost (Hamiltonian applications)},
        ylabel={norm residual},
    ymin=$tol, ymax=1,
        enlargelimits=0.05,
        legend pos=north east,
        grid style=dashed,
    ]
    $st3
    \\legend{$legend_str};
    \\end{axis}
    \\end{tikzpicture}

        \\caption{Comparison of different methods for $molecule}
        \\label{fig:comp-methods-$molecule}
    \\end{figure}
    ";
    
    path = "./src/performance_tests/plots/$molecule-plot.tex";

    write(path, st);
end

function to_points_string(xss,yss, init_res)
    points_string = ""

    c = 1
    for (xs,ys) = zip(xss, yss) 
        if isnothing(xs)
            xs = 1:length(ys);
        end

        instert_string = ""

        for (x,y) = zip(xs, ys)
            instert_string *= "($x,$y)"
        end

        mark  = marks[c];
        color = "clr$c";
        
        plot_string = "
        \\addplot[ %exact
        color=$color,
        mark=$mark,
        mark repeat=1,mark phase=1
        ]
        coordinates {(0,$init_res)$instert_string
        };";

        points_string *= plot_string;
        c += 1
    end
    return points_string
end

function calculate_init_res(model, ψ1)
    filled_occ = DFTK.filled_occupation(model);
    n_spin = model.n_spin_components;
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp);
    T = Float64
    Nk = length(basis.kpoints)
    ψ = deepcopy(ψ1)
    occupation = [filled_occ * ones(T, n_bands) for ik = 1:Nk]
    ρ = compute_density(basis, ψ, occupation)
    energies, H = energy_hamiltonian(basis, ψ, occupation; ρ)
    Hψ = H * ψ
    Λ = [ψ[ik]'Hψ[ik] for ik = 1:Nk]
    Λ = 0.5 * [(Λ[ik] + Λ[ik]') for ik = 1:Nk]
    res = [Hψ[ik] - ψ[ik] * Λ[ik] for ik = 1:Nk]
    return norm(res)
end