using Revise
using Turing
using SciMLBase: VectorOfArray
using SymbolicIndexingInterface
using Random
using Serialization
using StatsPlots
using Plots.Measures: mm

#open the chains
chain_1 = open(string(@__DIR__)*"/reference/rpareal_chain_reference.jls", "r") do io
        deserialize(io)
end

# the right order - mat
# 0 - sigma
# 1:nx2, 2:beta_1, 3:beta_2, 4:alpha_4, 
# 5:nr, 6:r2, 7:r1, 8:nx1, 9:beta_3, 
# 10:kr, 11:kx1, 12:alpha_2, 13:alpha_1, 
# 14:alpha_3, 15:beta_4
# σ, alfa1, kx1, nx1, beta1, alfa2, kx2, nx2, beta2, alfa4, kr, nr, beta4, r1, r2, alfa3, beta3
# same order
symbols_mk = [:nx2, :beta1, :beta2, :alfa4, 
 :nr, :r2, :r1, :nx1, :beta3, 
 :kr, :kx1, :alfa2, :alfa1, 
 :alfa3, :beta4]
symbols = [:nx2, :beta_1, :beta_2, :alpha_4, 
 :nr, :r2, :r1, :nx1, :beta_3, 
 :kr, :kx1, :alpha_2, :alpha_1, 
 :alpha_3, :beta_4]

chain_2 = open(string(@__DIR__)*"/minmtk_c27_jac0_reproduced.jls", "r") do io
        deserialize(io)
end

# chain_3 = open(string(@__DIR__)*"/posterior_samples_large_range_1_c_r5.jls", "r") do io
#         deserialize(io)
# end


# build the overlaid plots
plots = []
for (i, s) in enumerate(symbols)
    show_legend = (i == 1)

    # Plot the first chain
    p = plot(chain_1, s, label="Original", color=:navy, legend=show_legend)
    
    # Overlay the second chain on the SAME axes
    # Note: StatsPlots recipes for 'plot!' are smart enough to overlay 
    # both the trace and the density correctly.
    plot!(p, chain_2, s, label="MinMTK", color=:orangered, alpha=0.7, legend=show_legend)
    
    push!(plots, p)
end

# Assemble the final grid
plot(plots..., 
    layout = (length(symbols), 1), 
    size = (900, 3000), 
    left_margin = 25mm, # Increased slightly for safety
    bottom_margin = 2mm
)

# StatsPlots.plot(chain_2[symbols])
# StatsPlots.plot(chain_2[symbols])
# StatsPlots.plot!(chain_3)

# Increased burnout for better mixing
# StatsPlots.plot(chain[1:1000])
# StatsPlots.plot!(chain_2[1:1000])
# StatsPlots.plot!(chain_3[1000:2000])


# save the plot
savefig(string(@__DIR__)*"/posterior.pdf")
