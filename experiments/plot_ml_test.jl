function get_x_label(xfield)
    return xfield == :times_tot ? "CPU time in s" : "Iterations"
end

function get_y_label(yfield)
    return yfield == :norm_residuals ? "norm res" : "ΔE"
end

function generate_plot(cbs, ccs_arr, names, xfield, yfield; display_plt = true)
    plt = plot(; yscale = :log, ylabel = get_y_label(yfield), xlabel = get_x_label(xfield))
    for (cb, ccs, name) = zip(cbs, ccs_arr, names)
        plot_result(cb, name, xfield, yfield; ccs)
    end
    if display_plt
        display(plt)
    end
    return plt
end

function plot_result(cb, name, xfield, yfield; ccs = nothing)
    ys = getfield(cb, yfield) 
    xs = xfield == "iter" ? [i for i = 0:(length(ys)-1)] : getfield(cb, xfield)
    scale = xfield == "iter" ? 1 : 1e-9
    plot!(xs[1:end-1] * scale, ys[1:end-1], label = name)
    if !isnothing(ccs)
        ml_ys = [ys[k] for k = 1:length(ccs) if ccs[k]]
        ml_xs = [xs[k] * scale for k = 1:length(ccs) if ccs[k]]
        scatter!(ml_xs, ml_ys, label = "coarse cond $name")
    end
end

# idcs = [1,2,3,7]
# generate_plot(cbs[idcs], ccs_arr[idcs], names[idcs], :times_tot, :Es)
# generate_plot(cbs[idcs], ccs_arr[idcs], names[idcs], :times_tot, :norm_residuals)
# generate_plot(cbs[idcs], ccs_arr[idcs], names[idcs], "iter", :Es)
# generate_plot(cbs[idcs], ccs_arr[idcs], names[idcs], "iter", :norm_residuals);

# idcs = [4,5,6,8]
# generate_plot(cbs[idcs], ccs_arr[idcs], names[idcs], :times_tot, :Es)
# generate_plot(cbs[idcs], ccs_arr[idcs], names[idcs], :times_tot, :norm_residuals)
# generate_plot(cbs[idcs], ccs_arr[idcs], names[idcs], "iter", :Es)
# generate_plot(cbs[idcs], ccs_arr[idcs], names[idcs], "iter", :norm_residuals);



# plot(cbs[1].times_tot, cbs[1].norm_residuals)

colors = ["cpl1", "cpl2", "unia-purple", "cpl3", "cpl4", "cpl6"]

function plot_tikz(cb, ccs, name, xfield, yfield, color, idx, idx_max)
    ys = getfield(cb, yfield) 
    xs = xfield == "iter" ? [i for i = 0:(length(ys)-1)] : getfield(cb, xfield)
    scale = xfield == "iter" ? 1 : 1e-9
    xs *= scale

    st = "\n\\addplot[color=$color, draw opacity={1.0}, line width={1}, solid, mark=x]
    table[row sep={\\\\}]
    { \\\\\n"
    for (x,y) = zip(xs[1:end-1], ys[1:end-1])
        st *= "$x $y \\\\\n"
    end
    st *= "}; \n\\addlegendentry{$name}\n\n"
    


    if !isnothing(ccs)
        ml_ys = [ys[k] for k = 1:length(ccs) if ccs[k]]
        ml_xs = [xs[k] for k = 1:length(ccs) if ccs[k]]
        st *= "\\addplot[only marks, color=$color, draw opacity={1.0}, line width={1}, solid, text mark=\$\\bullet\$, forget plot]
        table[row sep={\\\\}]
        {
            \\\\\n"
        for (x,y) = zip(ml_xs, ml_ys)
            st *= "$x $y \\\\\n"
        end
        st *= "};\n\n"
    end

    return st
end

function generate_plot_tikz(cbs, ccs_arr, names,  xfield, yfield)
    ylabel = get_y_label(yfield)
    xlabel = get_x_label(xfield)
    ylabel = ylabel == "ΔE" ? "\$\\Delta E\$" : ylabel
    st = "\\begin{tikzpicture}
    \\begin{axis}[
    width={\\textwidth}, height={0.8 * \\textheight}, 
    xlabel={$xlabel}, 
    xticklabel style={font={{\\fontsize{8 pt}{10.4 pt}\\selectfont}}, color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0},
    rotate={0.0}},
    %x tick label style={font={{\\fontsize{8 pt}{11 pt}\\selectfont}}, color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, rotate={0.0}}, 
    xtick align={inside}, 
    xmajorgrids={true}, 
    x grid style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={0.1}, line width={0.5}, solid}, 
    ylabel={$ylabel}, 
    %x axis line style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, line width={1}, solid}, 
    scaled y ticks={false}, 
    ymode={log}, log basis y={10},
    ytick align={inside}, 
    %y tick style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, opacity={1.0}}, 
    yticklabel style={font={{\\fontsize{8 pt}{10.4 pt}\\selectfont}}, color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, rotate={0.0}}, 
    ymajorgrids={true},
    y grid style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={0.1}, line width={0.5}, solid}, 
    %axis y line*={left}, 
    %y axis line style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, line width={1}, solid}, 
    colorbar={false}
    ]\n"
    idx_max = length(cbs)
    for (cb, ccs, name, color, idx) = zip(cbs, ccs_arr, names, colors, 1:idx_max)
        st *= plot_tikz(cb, ccs, name, xfield, yfield, color, idx, idx_max)
    end
    st *= "\\end{axis}\n\\end{tikzpicture}"
    return st
end

# idcs = [1,2,3,7]
# st = generate_plot_tikz(cbs[idcs], ccs_arr[idcs], names[idcs], :times_tot, :Es)
# io = open("$model_name-H1-plt1.tex", "w"); write(io, st); close(io)
# st = generate_plot_tikz(cbs[idcs], ccs_arr[idcs], names[idcs], :times_tot, :norm_residuals)
# io = open("$model_name-H1-plt2.tex", "w"); write(io, st); close(io)
# st = generate_plot_tikz(cbs[idcs], ccs_arr[idcs], names[idcs], "iter", :Es)
# io = open("$model_name-H1-plt3.tex", "w"); write(io, st); close(io)
# st = generate_plot_tikz(cbs[idcs], ccs_arr[idcs], names[idcs], "iter", :norm_residuals);
# io = open("$model_name-H1-plt4.tex", "w"); write(io, st); close(io)

# idcs = [4,5,6,8]
# st = generate_plot_tikz(cbs[idcs], ccs_arr[idcs], names[idcs], :times_tot, :Es)
# io = open("$model_name-EA-plt1.tex", "w"); write(io, st); close(io)
# st = generate_plot_tikz(cbs[idcs], ccs_arr[idcs], names[idcs], :times_tot, :norm_residuals)
# io = open("$model_name-EA-plt2.tex", "w"); write(io, st); close(io)
# st = generate_plot_tikz(cbs[idcs], ccs_arr[idcs], names[idcs], "iter", :Es)
# io = open("$model_name-EA-plt3.tex", "w"); write(io, st); close(io)
# st = generate_plot_tikz(cbs[idcs], ccs_arr[idcs], names[idcs], "iter", :norm_residuals);
# io = open("$model_name-EA-plt4.tex", "w"); write(io, st); close(io)
