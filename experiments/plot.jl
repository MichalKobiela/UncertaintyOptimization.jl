using Revise
using Turing
using SciMLBase: VectorOfArray
using SymbolicIndexingInterface
using Random
using Serialization
using StatsPlots

#open the chains
chain = open(string(@__DIR__)*"/posterior_samples_large_range_1_c11_r1_u0AD.jls", "r") do io
        deserialize(io)
end

# the right order - mat
# 0 - sigma
# 1:nx2, 2:beta_1, 3:beta_2, 4:alpha_4, 
# 5:nr, 6:r2, 7:r1, 8:nx1, 9:beta_3, 
# 10:kr, 11:kx1, 12:alpha_2, 13:alpha_1, 
# 14:alpha_3, 15:beta_4

chain_2 = open(string(@__DIR__)*"/posterior_try47_auto_max10k_rel1en2_abs1en3.jls", "r") do io
        deserialize(io)
end

# chain_3 = open(string(@__DIR__)*"/posterior_samples_large_range_1_c_r5.jls", "r") do io
#         deserialize(io)
# end

StatsPlots.plot(chain)
StatsPlots.plot!(chain_2)
# StatsPlots.plot!(chain_3)

# Increased burnout for better mixing
# StatsPlots.plot(chain[1:1000])
# StatsPlots.plot!(chain_2[1:1000])
# StatsPlots.plot!(chain_3[1000:2000])


# save the plot
savefig(string(@__DIR__)*"/posterior.pdf")
