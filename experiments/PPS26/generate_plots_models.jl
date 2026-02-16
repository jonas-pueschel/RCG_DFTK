using RCG_DFTK
using DFTK
using Printf
using Plots

include("test_model.jl")
include("../precompile_methods.jl")
precompile_methods()


method_names = ["EARCG-St", "EARCG-Gr", "H1RCG", "L2RCG", "SCF"]
model_names =  ["silicon", "GaAs", "TiO2"]
save_mode = "png" #"latex"

function generate_plots(xss,yss, colors, marks)
    st = ""
    for (xs, ys, color,mark) = zip(xss,yss, colors,marks) 
        st *= """ 
        \\addplot[color=$color, draw opacity={1.0}, line width={1}, solid, mark=$mark, mark repeat=2,mark phase=2]
            table[row sep={\\\\}]
            {
                $(generate_table(xs, ys))
            };
        """
    end
    return st
end

function generate_table(xs, ys)
    st = "\\\\"

    for (x,y) = zip(xs,ys)
        st *= "\n$x $y \\\\"
    end
    return st
end

function generate_legend(labels)

    st = ""
    for label = labels
        pf = label == labels[end] ? "" : "\$\\qquad\$"
        st *= "\\addlegendentry {$label$pf}\n"
    end
    return st
end


function fill_template(xss, yss, position, model_name, x_label)
    #position = 1,2,3 : iter, hams, time

    ylabel_st = position == 1 ? "ylabel={\$\\|R^{(m)}\\|_F\$}, " : ""
    ytick = position == 1 ? "yticklabels={\\empty}," : "yticklabels={{\$10^{-8}\$,\$10^{-6}\$,\$10^{-4}\$,\$10^{-2}\$}}, ytick={{1.0e-8,1.0e-6,0.0001,0.01}},"

    at_x = position == 1 ? 0 : (position - 1) * 51.5
    xmin = 0.0
    xmax = sort([xs[end] for xs = xss])[end-1] * 1.1

    marks = ["square", "o", "asterisk", "|", "diamond"]

    colors = ["unia-purple", "cpl1", "cpl4", "cpl2", "cpl3"]

    labels =  ["EARCG-St", "EARCG-Gr", "H1RCG", "L2RCG", "SCF"]

    legend_st1 = position == 3 ? generate_legend(labels) : ""
    
    legend_st2 = position == 3 ? """legend cell align={left}, legend columns={5},
    legend style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, 
    solid, fill={rgb,1:red,1.0;green,1.0;blue,1.0}, fill opacity={1.0}, 
    text opacity={1.0}, font={{\\fontsize{8 pt}{10.4 pt}\\selectfont}}, 
    text={rgb,1:red,0.0;green,0.0;blue,0.0}, cells={anchor={center}}, at={(-2.35, 1.05)}, anchor={south west}},""" : ""

    tex_str = """
    \\begin{axis}[
    name=$model_name$position,
    at={($(at_x)mm,0.0mm)},
    width={\\plotwidth}, height={\\plotheight}, $legend_st2
    xlabel={$x_label}, 
    xticklabel style={font={{\\fontsize{8 pt}{10.4 pt}\\selectfont}}, color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0},
    rotate={0.0}},
    %x tick label style={font={{\\fontsize{8 pt}{11 pt}\\selectfont}}, color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, rotate={0.0}}, 
    xmin={$xmin}, xmax={$xmax}, 
    xtick align={inside}, 
    xmajorgrids={true}, 
    x grid style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={0.1}, line width={0.5}, solid}, 
    %axis x line*={left}, $ylabel_st
    %x axis line style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, line width={1}, solid}, 
    scaled y ticks={false}, 
    ymode={log}, log basis y={10}, ymin={5e-9}, ymax={0.25}, 
    $ytick
    ytick align={inside}, 
    %y tick style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, opacity={1.0}}, 
    yticklabel style={font={{\\fontsize{8 pt}{10.4 pt}\\selectfont}}, color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, rotate={0.0}}, 
    ymajorgrids={true},
    y grid style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={0.1}, line width={0.5}, solid}, 
    %axis y line*={left}, 
    %y axis line style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, line width={1}, solid}, 
    colorbar={false}
    ]

    $(generate_plots(xss,yss,colors,marks))

    $legend_st1
    \\end{axis}
    """

    return tex_str
end


ppercentages = []

for model_name = model_names
    global ppercentages
    callbacks, norm_res_0, percentages = test_model(; model_name, method_names)

    ppercentages = [ppercentages..., percentages]

    yss = []
    xss_its = []
    xss_hams = []
    xss_times = []
    for cb = callbacks
        ys = [norm_res_0, cb.norm_residuals[1:end-1]...]
        xs_its = collect(0:(length(ys)-1))
        xs_hams = [0, cb.calls_DftHamiltonian[1:end-1]...]
        xs_times = [0, cb.times_tot[1:end-1]...] ./ 1e9
        yss = [yss..., ys]
        xss_its = [xss_its..., xs_its]
        xss_hams = [xss_hams..., xs_hams]
        xss_times = [xss_times..., xs_times]
    end

    for (xss, position, x_label) in zip([xss_its, xss_hams, xss_times], [1,2,3], ["Iterations", "Hamiltonians", "CPU time (s)"])

        #re-normalize from matrix-vector to matrix-matrix multiplications
        if x_label == "Hamiltonians" 
            n_els = model_name == "TiO2" ? 16 : 4
            xss ./= n_els
        end

        if save_mode == "png"
            # plot png
            filename = "$model_name-plt$position.png"
            plt = plot(; yscale = :log, ylabel = "norm res", xlabel = x_label, title = model_name)
            for (xs,ys, method_name) in zip(xss,yss,method_names)
                plot!(xs,ys, label = method_name)
            end
            savefig(plt, filename)

        elseif  save_mode == "latex"
            # plot latex code 
            filename = "$model_name-plt$position.tex"
            io = open(filename, "w")
            text =  fill_template(xss, yss, position, model_name, x_label)
            write(io, text)
            close(io)
        end
    end
end

if save_mode == "latex"
    io = open("times_table.tex", "w")
    lines = ""
    headline = prod("& $method_name" for method_name = method_names)
    cs =  prod(" c " for method_name = method_names)
    for (model_name, percentages) = zip(model_names, ppercentages)
        global lines
        line = "$model_name"
        for percentage = percentages
            pstring = @sprintf("%.1f", 100percentage)
            line *= " & $pstring\\%"
        end 
        line *= "\\\\\n"
        lines *= line
    end

    st = """\\begin{table}[H]
    \\begin{center}
    \\begin{tabular}{||c | $cs||} 
    \\hline
    $headline
    \\\\ [0.5ex] 
    \\hline\\hline
    $lines[1ex] 
    \\hline
    \\end{tabular}
    \\end{center}
        \\caption{Percentage of overall runtime cost caused by \\texttt{DftHamiltonian\\_multiplication}.}
        \\label{tab:ham_call_percentage}
    \\end{table}"""
    write(io, st)
    close(io)
end
