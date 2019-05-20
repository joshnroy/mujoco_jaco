#= using Pkg =#
#= Pkg.add("Plots") =#
#= Pkg.add("CSV") =#
#= Pkg.add("DataFrames") =#
#= Pkg.add("NPZ") =#
using Plots
using CSV
using DataFrames
using NPZ

function stretch_to_500k(input_data)
    stretched = Float64[]
    for x in input_data
        for i in range(1, stop=x)
            append!(stretched, x)
        end
    end
    return stretched
end

println("Loading Data")
visualDeepmindLab = npzread("dqn_visual_history.npz")["episode_reward"]
RMSVisualDeepmindLab = npzread("dqn_RMSprop_visual_history.npz")["episode_reward"]
#= stylegan = npzread("stylegan_dqn_training_history_500k_again.npz")["episode_reward"] =#
#= temporalVae = npzread("vae_dqn_training_history_500k_modified.npz")["episode_reward"] =#

smoothedVisualDeepmindLab = float(copy(visualDeepmindLab))
smoothedRMSVisualDeepmindLab = float(copy(RMSVisualDeepmindLab))
#= smoothedstylegan = float(copy(stylegan)) =#
#= smoothedtemporalVae = float(copy(temporalVae)) =#

#= Smooth the array =#
println("Smoothing Data")
alpha = 0.01
for i in 2:length(smoothedVisualDeepmindLab)
    smoothedVisualDeepmindLab[i] = alpha * visualDeepmindLab[i] + (1. - alpha) * smoothedVisualDeepmindLab[i-1]
end
for i in 2:length(smoothedRMSVisualDeepmindLab)
    smoothedVisualDeepmindLab[i] = alpha * visualDeepmindLab[i] + (1. - alpha) * smoothedVisualDeepmindLab[i-1]
end
#= for i in 2:length(smoothedstylegan) =#
#=     smoothedstylegan[i] = alpha * stylegan[i] + (1. - alpha) * smoothedstylegan[i-1] =#
#= end =#
#= for i in 2:length(smoothedtemporalVae) =#
#=     smoothedtemporalVae[i] = alpha * temporalVae[i] + (1. - alpha) * smoothedtemporalVae[i-1] =#
#= end =#

println("Stretching Data")
visualDeepmindLab = stretch_to_500k(visualDeepmindLab)
smoothedVisualDeepmindLab = stretch_to_500k(smoothedVisualDeepmindLab)
#= smoothedstylegan = stretch_to_500k(smoothedstylegan) =#
#= smoothedtemporalVae = stretch_to_500k(smoothedtemporalVae) =#

#= minLength = minimum([length(smoothedvanillaDeepmindLab) length(smoothedVisualDeepmindLab) length(smoothedstylegan) length(smoothedtemporalVae)]) =#
minLength = minimum([length(visualDeepmindLab) length(smoothedVisualDeepmindLab)])
println("minLength is ", minLength)

visualDeepmindLab = visualDeepmindLab[1:minLength]
smoothedVisualDeepmindLab = smoothedVisualDeepmindLab[1:minLength]
#= smoothedstylegan = smoothedstylegan[1:minLength] =#
#= smoothedtemporalVae = smoothedtemporalVae[1:minLength] =#

println("Plotting Data")
x_data = range(1, stop=minLength)
#= plot(x_data, [smoothedvanillaDeepmindLab smoothedVisualDeepmindLab smoothedstylegan smoothedtemporalVae], label=["Smoothed Vanilla DeepmindLab" "Smoothed Visual DeepmindLab" "Smoothed Stylegan" "Smoothed Temporal VAE"], xlabel="Number of Timesteps", ylabel="Reward", title="Reward vs Training Timestep", legend=:bottomright) =#
plot(x_data, [visualDeepmindLab smoothedVisualDeepmindLab], label=["Visual DeepmindLab" "Smoothed Visual DeepmindLab"], linealpha=[0.5 1], xlabel="Number of Timesteps", ylabel="Reward", title="Reward vs Training Timestep", legend=:bottomright)

savefig("losses.png")
