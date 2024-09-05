using PlotlyJS
plot(rand(10))

data = range(-5, stop=5, length=40)

X, Y, Z = mgrid(data, data, data)


values = X .* X .* 0.5 .+ Y .* Y .+ Z .* Z .* 2


plt = plot(isosurface(
    x=X[:],
    y=Y[:],
    z=Z[:],
    value=values[:],
    isomin=10,
    isomax=40,
    caps=attr(x_show=false, y_show=false)
))

display(plt)