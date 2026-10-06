using BSON

function get_x_label(xfield)
    return xfield == :times_tot ? "CPU time (sec)" : "iterations"
end

function get_y_label(yfield)
    return yfield == :norm_residuals ? "\$\\|R_k\\|\$" : "\$E(\\Phi_k) - E_{\\rm ref}\$"
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


function inner_plot(xs, ys, ccs, color, mark; vis = nothing)
    vstring = isnothing(vis) ? "" : " visible on=<$vis->,"
    if endswith(mark, "*")
        fill1 = ", mark options = {fill = white}"
        fill2 = ", mark options = {fill = $color}"
    else
        fill1 = ""
        fill2 = ""
    end
    st = "\n\\addplot[color=$color, draw opacity={1.0}, line width={1},$vstring solid, mark=$(mark)$fill1]
    table[row sep={\\\\}]
    { \\\\\n"
    for (x,y) = zip(xs[1:end-1], ys[1:end-1])
        st *= "$x $y \\\\\n"
    end
    st *= "};"
    if !isnothing(ccs)
        ml_ys = [ys[k] for k = 1:length(ccs) if ccs[k]]
        ml_xs = [xs[k] for k = 1:length(ccs) if ccs[k]]
        st *= "\\addplot[only marks, color=$color, draw opacity={1.0}, line width={1},$vstring solid, mark=$(mark)$fill2, forget plot]
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

function plot_tikz(cb, ccs, name, xfield, yfield, color, mark; idx = 0, idx_max = 0, legend = true)
    ys = getfield(cb, yfield) 
    xs = xfield == "iter" ? [i for i = 0:(length(ys)-1)] : getfield(cb, xfield)
    scale = xfield == "iter" ? 1 : 1e-9
    xs *= scale

    st = ""
    if idx == 0 && idx_max == 0
        st *= inner_plot(xs, ys, ccs, color, mark)
    else
        st_i1 = inner_plot(xs, ys, ccs, color, mark; vis = idx)
        st_i2 = inner_plot(xs, ys, ccs, "light-gray", mark; vis = idx)
        st *= "
        \\only<$idx,$(idx_max + 1)>{
            $st_i1
        }
        \\only<$(idx + 1)-$idx_max>{
            $st_i2
        }"
    end

    
    if legend
        st *= "\n\\addlegendentry{$name}\n\n"
    end




    return st
end



function generate_plot_tikz(cbs, ccs_arr, names,  xfield, yfield, pos, pcolors, pmarks; in_beamer = false)
    atx = pos == 1 ? "0" : "0.5 * \\textwidth"
    xmax = 1.1 * get_x_max(cbs, xfield)
    ymin = 0.8 * (yfield == :Es ? 1e-15 : 1e-9)
    legend = (pos == 1)
    height = in_beamer ? "0.75 * \\textheight" : "0.5 * \\textwidth"
    legend_st = legend ? "
    legend cell align={left}, legend columns={-1},
    legend style={/tikz/every even column/.append style={column sep=0.5cm},
    color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, 
    solid, fill={rgb,1:red,1.0;green,1.0;blue,1.0}, fill opacity={1.0}, 
    text opacity={1.0}, font={{\\fontsize{8 pt}{10.4 pt}\\selectfont}}, 
    text={rgb,1:red,0.0;green,0.0;blue,0.0}, cells={anchor={center}}, at={(0.12, 1.05)}, anchor={south west}}," : ""

    ylabel = get_y_label(yfield)
    xlabel = get_x_label(xfield)

    st = "
    \\begin{axis}[at={($atx, 0)},
    width={0.5 * \\textwidth}, height={$height},$legend_st 
    xlabel={$xlabel}, 
    xticklabel style={font={{\\fontsize{8 pt}{10.4 pt}\\selectfont}}, color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0},
    rotate={0.0}},
    xmax = {$xmax},
    xtick align={inside}, 
    xmajorgrids={true}, 
    x grid style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={0.1}, line width={0.5}, solid}, 
    ylabel={$ylabel}, 
    scaled y ticks={false}, 
    ymode={log}, log basis y={10},
    ytick align={inside}, 
    yticklabel style={font={{\\fontsize{8 pt}{10.4 pt}\\selectfont}}, color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, rotate={0.0}}, 
    ymajorgrids={true},
    y grid style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={0.1}, line width={0.5}, solid}, 
    colorbar={false}
    ]\n"
    idx_max = length(cbs)
    for (cb, ccs, name, color, mark, idx) = zip(cbs, ccs_arr, names, pcolors, pmarks, 1:idx_max)
        if in_beamer
            st *= plot_tikz(cb, ccs, name, xfield, yfield, color, mark; idx, idx_max, legend)
        else
            st *= plot_tikz(cb, ccs, name, xfield, yfield, color, mark; idx = 0, idx_max = 0, legend)
        end
    end
    st *= "\\end{axis}\n"
    return st
end

function get_x_max(cbs, xfield)
    mx = 0
    scale = xfield == "iter" ? 1 : 1e-9
    for cb = cbs
        xs = xfield == "iter" ? [i for i = 0:(length(cb.norm_residuals)-1)] : getfield(cb, xfield)
        mx = max(xs..., mx)
    end
    return mx * scale
end


function generate_figs_tikz(plot_name, cbs, ccs_arr, names, xfield; perm = 1:length(cbs))
    st = "\\begin{tikzpicture}"
    st *= generate_plot_tikz(cbs[perm], ccs_arr[perm], names[perm],  xfield, :Es, 1, colors, marks)        
    st *= generate_plot_tikz(cbs[perm], ccs_arr[perm], names[perm],  xfield, :norm_residuals, 2, colors, marks)     
    st*= "\\end{tikzpicture}"
    io = open("$plot_name.tex", "w"); write(io, st); close(io)
end

#permutation 
perm = [7,6,1,2,3,4]

colors = ["cpl6", "cpl3", "cpl1", "unia-purple", "cpl2", "cpl4"]
marks = ["x" ,"asterisk", "pentagon*", "square*", "triangle*" ,"diamond*"]


# cbs_arr is array of named tuples (Es = [...], norm_residuals = [...], times_tot = [...])
# ccs_arr is array of coarse corrections [true, false, ...] corresponding to the results in cbs_arr
# names is array of strings detailing the algorithm names

# BSON.@load "GaAs-H1-result.bson" cbs_arr ccs_arr names

generate_figs_tikz("plot_GaAs_iter", cbs_arr, ccs_arr, names, "iter"; perm)
generate_figs_tikz("plot_GaAs_times", cbs_arr, ccs_arr, names, :times_tot; perm)