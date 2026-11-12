"""
    multilevel_tikz(Ecuts; algorithm = :rml, Ecut_levels = sort(unique(Ecuts)), digits = 0,
                    dots = false, file = nothing, dx_step = 0.36, dx_level = 0.28, dy = 1.0)

TikZ picture of the level hierarchy visited by a multilevel method. `Ecuts` is the Ecut of the
basis of every gradient calculation, as in the `levels` field of the infos of RML / RFMG.
Every entry is drawn as a smoothing step, consecutive steps on the coarsest level are combined
into one solve. Levels jumped over between two steps are marked as transit (no step).
`algorithm` sets where the picture starts: `:rml` on the finest level (if the first step is
on a coarser level, the picture starts with a transit there), `:rfmg` on the coarsest. If the
last step is not on the finest level, the picture ends with transits up to the finest level.
`Ecut_levels` are the levels drawn (needed if a level of the hierarchy has no step at all),
`digits` the digits of the Ecut labels, `dots = true` adds "⋯" after the last step. The picture
is returned as a string and written to `file` if given. Needs `\\usetikzlibrary{arrows.meta}`.
"""
function multilevel_tikz(Ecuts; algorithm = :rml, Ecut_levels = sort(unique(Ecuts)), digits = 0,
                         dots = false, file = nothing, dx_step = 0.36, dx_level = 0.28, dy = 1.0)
    @assert algorithm in (:rml, :rfmg)
    n_levels = length(Ecut_levels)
    level_of(E) = findfirst(El -> isapprox(El, E), Ecut_levels)
    @assert all(!isnothing ∘ level_of, Ecuts) "every Ecut must be in Ecut_levels"
    y(l) = (l - 1) * dy
    fmt(E) = digits == 0 ? string(round(Int, E)) : string(round(E; digits))

    # steps: level of every node, consecutive steps on the coarsest level are one solve
    steps = Int[]
    for E in Ecuts
        l = level_of(E)
        (l == 1 && !isempty(steps) && last(steps) == 1) || push!(steps, l)
    end

    nodes = String[]
    arrows = String[]
    x = 0.0
    n_nodes = 0
    function add_node!(style, l)
        n_nodes += 1
        push!(nodes, "\\node[$style] (N$n_nodes) at ($(round(x; digits = 3)), $(y(l))) {};")
        return n_nodes
    end
    arrow!(style, i, j) = push!(arrows, "\\draw[$style] (N$i) -- (N$j);")

    l_start = algorithm == :rml ? n_levels : 1
    prev, l = first(steps) == l_start ? (0, l_start) : (add_node!("transit", l_start), l_start)
    for a in steps
        if prev == 0  # first step, no transit before it
            prev = add_node!(a == 1 ? "solve" : "smooth", a)
        elseif a == l
            x += dx_step
            prev = add_node!(a == 1 ? "solve" : "smooth", a)
        else
            style = a < l ? "restrict" : "prolongate"
            for j in (a < l ? ((l - 1):-1:(a + 1)) : ((l + 1):(a - 1)))
                x += dx_level
                next = add_node!("transit", j)
                arrow!(style, prev, next)
                prev = next
            end
            x += dx_level
            next = add_node!(a == 1 ? "solve" : "smooth", a)
            arrow!(style, prev, next)
            prev = next
        end
        l = a
    end
    # end on the finest level: prolongate there without further steps
    for j in (l + 1):n_levels
        x += dx_level
        next = add_node!("transit", j)
        arrow!("prolongate", prev, next)
        prev = next
    end
    l = n_levels
    x_end = x + (dots ? 0.8 : 0.3)

    st = """
    \\begin{tikzpicture}[
        node/.style={circle, draw, inner sep=0pt, line width=0.6pt},
        smooth/.style={node, minimum size=3.2mm, draw=blue!35!black, fill=blue!12},
        solve/.style={node, minimum size=3.2mm, draw=black, fill=black!75},
        transit/.style={node, minimum size=1.5mm, draw=black!45, fill=white},
        edge/.style={-{Stealth[length=5pt, width=5pt, round]}, line width=1.3pt, line cap=round,
                     shorten <=1.5pt, shorten >=1.5pt},
        restrict/.style={edge, red!65!black},
        prolongate/.style={edge, green!50!black, dash pattern=on 3.5pt off 2.5pt},
        label/.style={font=\\small, anchor=east},
        legend/.style={font=\\small, anchor=west}
      ]
    % ---- levels ----
    """
    for k in 1:n_levels
        suffix = k == 1 ? " (coarse)" : (k == n_levels ? " (fine)" : "")
        st *= "\\draw[black!15, line width=0.6pt] (-0.2, $(y(k))) -- ($(round(x_end; digits = 3)), $(y(k)));\n"
        st *= "\\node[label] at (-0.35, $(y(k))) {\$E_\\mathrm{cut} = $(fmt(Ecut_levels[k]))\$$suffix};\n"
    end
    st *= "% ---- steps ----\n" * join(nodes, "\n") * "\n" * join(arrows, "\n") * "\n"
    if dots
        for i in 1:3
            st *= "\\fill[black!60] ($(round(x + 0.3 + 0.12i; digits = 3)), $(y(l))) circle (0.8pt);\n"
        end
    end

    # legend below the coarsest level, arrow heads at the right end
    yl = -0.8
    st *= """
    % ---- legend ----
    \\draw[restrict]   (-2.2, $yl) -- (-1.4, $yl) node[legend, black] {restriction};
    \\draw[prolongate] (-2.2, $(yl - 0.5)) -- (-1.4, $(yl - 0.5)) node[legend, black] {prolongation};
    \\node[smooth]  at (1.0, $yl) {}; \\node[legend] at (1.2, $yl) {smoothing};
    \\node[solve]   at (3.2, $yl) {}; \\node[legend] at (3.4, $yl) {solving};
    \\node[transit] at (5.0, $yl) {}; \\node[legend] at (5.2, $yl) {transit (no step)};
    \\end{tikzpicture}
    """

    isnothing(file) || write(file, st)
    return st
end

# example: the level indices of a 4-level RML run, mapped to Ecuts
# as = [3,3,2,1,3,3,4,4,2,3,3,1,2,3,4,4,3,3,1,2,3,4]
# multilevel_tikz([14.6, 24.9, 42.3, 71.8][as]; algorithm = :rml, file = "multilevel_pattern.tex")
