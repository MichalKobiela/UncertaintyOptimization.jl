# using Revise
# using Turing
# using SciMLBase: VectorOfArray
# using SymbolicIndexingInterface
# using Random
using Serialization
using StatsPlots
using Plots.Measures: mm


symbols = [:nx2, :beta_1, :beta_2, :alpha_4, 
 :nr, :r2, :r1, :nx1, :beta_3, 
 :kr, :kx1, :alpha_2, :alpha_1, 
 :alpha_3, :beta_4]


# recomputed, 3 chains in one go
# "/home/mbieniek/code/uncertainty-circ-opt/RPARealData/Inference/posterior_samples_large_range_1_c_renamedAndCorrectInit.jls"

chain_1 = open(string(@__DIR__)*"/reference/rpareal_chain_reference.jls", "r") do io
        deserialize(io)
end

chain_2 = open(string(@__DIR__)*"/minmtk_c52_nuts0.5_e0.003_DEM_tsitRB23_j12.6updated.jls", "r") do io
        deserialize(io)
end


# build the overlaid plots
plots = []
for (i, s) in enumerate(symbols)
    show_legend = (i == 1)

    # Plot the first chain
    p = plot(chain_1, s, label="Original", color=:navy, legend=show_legend)
    
    # Overlay the second chain on the SAME axes
    # Note: StatsPlots recipes for 'plot!' are smart enough to overlay 
    # both the trace and the density correctly.
    plot!(p, chain_2, s, label="Repr", color=:orangered, alpha=0.7, legend=show_legend)
    
    push!(plots, p)
end

# Assemble the final grid
plot(plots..., 
    layout = (length(symbols), 1), 
    size = (900, 3000), 
    left_margin = 25mm, # Increased slightly for safety
    bottom_margin = 2mm
)

# Increased burnout for better mixing
# StatsPlots.plot(chain[1:1000])
# StatsPlots.plot!(chain_2[1:1000])
# StatsPlots.plot!(chain_3[1000:2000])

# save the plot
savefig(string(@__DIR__)*"/posterior.pdf")

# save text files in case they're needed
# CSV.write("ref.csv", DataFrame(chain_1[1:10:end,:, :]))
# CSV.write("repr.csv", DataFrame(chain_2[1:10:end,:, :]))


# using Statistics
# chain_df = DataFrame(chain_1)
# @show mean(chain_df.tree_depth .== 10)
# @show mean(chain_df.n_steps .== 1023)
# @show median(chain_df.n_steps)
# @show maximum(chain_df.n_steps)