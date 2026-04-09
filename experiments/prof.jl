using CSV, Tables
using Serialization
using Plots
using Profile
using PProf

data, lidict = deserialize("profile_results_1thread.jlprof")
# data, lidict = deserialize("/home/mbieniek/code/uncertainty-circ-opt/RPARealData/profile_results_1thread.jlprof")
# tree view
# Profile.print(stdout, data, lidict; format=:tree)
# flat
# Profile.print(stdout, data, lidict; format=:flat)

# Profile.print(stdout, data, lidict; format=:tree, maxdepth=25)
# Profile.print(stdout, data, lidict; format=:tree, C=true)
# Profile.print(stdout, data, lidict; format=:flat)#, sortedby=:count)

# open("profile.txt", "w") do io
#         Profile.print(io, data, lidict; format=:flat, sortedby=:count)
# end

# Profile.print(stdout, data, lidict; format=:tree, C=true, maxdepth=30, sortedby=:count) # groupby=:task



pprof(data, lidict)