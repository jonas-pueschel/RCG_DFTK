using RCG_DFTK
using DFTK
include("calc_gaps.jl")
include("test_gaps.jl")

# In order to make this result independent from precompile time,
# run the precompile_methods function before (otherwise times will be off)
include("../precompile_methods.jl")
precompile_methods()

# range of a
as = 10:0.1:11.4

# how many runs per a
n_examples = 1

function generate_table(xs, ys)
    st = "\\\\"

    for (x,y) = zip(xs,ys)
        st *= "\n$y $x \\\\"
    end
    return st
end

function generate_plots(xss,ys, colors, marks)
    st = ""
    for (xs,color,mark) = zip(xss,colors,marks) 
        st *= """ 
        \\addplot[color=$color, draw opacity={1.0}, line width={1}, solid, mark=$mark] %, mark repeat=2,mark phase=2]
            table[row sep={\\\\}]
            {
                $(generate_table(xs, ys))
            };
        """
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

function fill_template(xss,ys,position)

    fac = 1.1
    #position = 1,2,3 : gaps, hams, rel_hams
    at_x = position == 1 ? 0 : (position - 2) * 51.5 * fac
    ymode = position == 2 ? "ymode = {log}, log basis y = {10}," : ""
    ymins = [0, 200 , 0.8]
    ymaxs = [0.15,4000 , 2.5]

    marks = position == 1 ? ["square", "o", "diamond"] : ["square", "o", "|", "diamond", "asterisk"]

    colors = position == 1 ? ["cpl1", "cpl2", "cpl3"] : ["unia-purple", "cpl1", "cpl2", "cpl3", "cpl4"]

    labels = position == 1 ?  ["HOMO-LUMO gap", "eff. gap", "virt. gap"] :  ["EARCG-St", "EARCG-Gr", "H1RCG", "L2RCG", "SCF"]

    legend_st1 = position == 2 ? "" : generate_legend(labels)
    
    legend_st2 = ""

    if position == 1
        legend_st2 = """legend cell align={left}, legend columns={1},
        legend style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, 
        solid, fill={rgb,1:red,1.0;green,1.0;blue,1.0}, fill opacity={1.0}, 
        text opacity={1.0}, font={{\\fontsize{8 pt}{10.4 pt}\\selectfont}}, 
        text={rgb,1:red,0.0;green,0.0;blue,0.0}, cells={anchor={center}}, at={(0.98, 0.97)}, anchor={north east}},"""
    end

    if position == 3
        legend_st2 = """legend cell align={left}, legend columns={5},
        legend style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, 
        solid, fill={rgb,1:red,1.0;green,1.0;blue,1.0}, fill opacity={1.0}, 
        text opacity={1.0}, font={{\\fontsize{8 pt}{10.4 pt}\\selectfont}}, 
        text={rgb,1:red,0.0;green,0.0;blue,0.0}, cells={anchor={center}}, at={(-1.25, 1.05)}, anchor={south west}},"""
    end

    tex_str = """
    \\begin{axis}[
    at={($(at_x)mm,0.0mm)},
    width={$fac\\plotwidth}, height={$fac\\plotheight}, $legend_st2
    xlabel={\$a\$}, 
    xticklabel style={font={{\\fontsize{8 pt}{10.4 pt}\\selectfont}}, color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0},
    rotate={270.0}},
    xtick = {10.0, 10.2, 10.4, 10.6, 10.8, 11, 11.2, 11.4},
    xticklabels = {10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4},
    %x tick label style={font={{\\fontsize{8 pt}{11 pt}\\selectfont}}, color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, rotate={0.0}}, 
    xmin={$(min(ys...))}, xmax={$(max(ys...))}, 
    xtick align={inside}, 
    xmajorgrids={true}, 
    x grid style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={0.1}, line width={0.5}, solid}, 
    %axis x line*={left}, 
    %x axis line style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, line width={1}, solid}, 
    scaled y ticks={false}, $ymode
    ymin={$(ymins[position])}, ymax={$(ymaxs[position])}, 
    ytick align={inside}, 
    %y tick style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, opacity={1.0}}, 
    yticklabel style={font={{\\fontsize{8 pt}{10.4 pt}\\selectfont}}, color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, rotate={0.0}}, 
    ymajorgrids={true},
    y grid style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={0.1}, line width={0.5}, solid}, 
    %axis y line*={left}, 
    %y axis line style={color={rgb,1:red,0.0;green,0.0;blue,0.0}, draw opacity={1.0}, line width={1}, solid}, 
    colorbar={false}
    ]

    $(generate_plots(xss,ys,colors,marks))

    $legend_st1
    \\end{axis}
    """

    return tex_str
end


using Statistics

(callbacks_l2rcg, callbacks_h1rcg, callbacks_earcg, callbacks_earg, callbacks_earcg0, callbacks_earg0, callbacks_scf, as, gaps, 
    gaps_eff) =  test_gaps(;as, n_examples);
#using BSON
#BSON.@save "temp.bson" gaps gaps_eff callbacks_l2rcg callbacks_h1rcg callbacks_earcg callbacks_earg callbacks_earcg0 callbacks_earg0 callbacks_scf

factor(cb) = 1.0 #(log10(cb.norm_residuals[1]) + 8)/ (log10(cb.norm_residuals[1]) - log10(cb.norm_residuals[end]))

get_means(cbs, fieldname; is_end = false) = is_end ? 
    [mean([getfield(cb, fieldname)[end] * factor(cb) for cb = cblist]) for cblist = cbs[1:end]] : 
    [mean([getfield(cb, fieldname) * factor(cb) for cb = cblist]) for cblist = cbs[1:end]]
get_std(cbs, fieldname; is_end = false) = is_end ? 
    [std([getfield(cb, fieldname)[end] * factor(cb) for cb = cblist]) for cblist = cbs[1:end]] : 
    [std([getfield(cb, fieldname) * factor(cb) for cb = cblist]) for cblist = cbs[1:end]]

xss_means = []
xss_means_norm = []

for cbs = [callbacks_earcg, callbacks_earcg0, callbacks_h1rcg, callbacks_l2rcg,  callbacks_scf]
    global xss_means_norm
    global xss_means
    means = get_means(cbs, :calls_DftHamiltonian; is_end = true)
    stds = get_std(cbs, :calls_DftHamiltonian; is_end = true)
    
    #re-normalize from matrix-vector to matrix-matrix multiplications
    means /= 4

    means_norm = copy(means) 
    means_norm ./= means[1]
    xss_means = [xss_means..., means]
    xss_means_norm = [xss_means_norm..., means_norm]
end

yss = [as for i = 1:5]


virtual_gaps = get_virtual_gaps(; as);

xss_gaps = [gaps, gaps_eff, virtual_gaps]
io = open("plt-gaps-1.tex", "w")
write(io, fill_template(xss_gaps, as, 1))
close(io)

io = open("plt-gaps-2.tex", "w")
write(io, fill_template(xss_means, as, 2))
write(io, fill_template(xss_means_norm, as, 3))
close(io)
