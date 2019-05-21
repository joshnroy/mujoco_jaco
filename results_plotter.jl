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
darlaDqn = npzread("darla_dqn_history.npz")["episode_reward"]
myVaeDqn = npzread("myvae_dqn_history.npz")["episode_reward"]
#= stylegan = npzread("stylegan_dqn_training_history_500k_again.npz")["episode_reward"] =#
#= temporalVae = npzread("vae_dqn_training_history_500k_modified.npz")["episode_reward"] =#

smoothedDarlaDqn = float(copy(darlaDqn))
smoothedmyVaeDqn = float(copy(myVaeDqn))
#= smoothedstylegan = float(copy(stylegan)) =#
#= smoothedtemporalVae = float(copy(temporalVae)) =#

#= Smooth the array =#
println("Smoothing Data")
alpha = 0.01
for i in 2:length(smoothedDarlaDqn)
    smoothedDarlaDqn[i] = alpha * darlaDqn[i] + (1. - alpha) * smoothedDarlaDqn[i-1]
end
for i in 2:length(smoothedmyVaeDqn)
    smoothedDarlaDqn[i] = alpha * darlaDqn[i] + (1. - alpha) * smoothedDarlaDqn[i-1]
end
#= for i in 2:length(smoothedstylegan) =#
#=     smoothedstylegan[i] = alpha * stylegan[i] + (1. - alpha) * smoothedstylegan[i-1] =#
#= end =#
#= for i in 2:length(smoothedtemporalVae) =#
#=     smoothedtemporalVae[i] = alpha * temporalVae[i] + (1. - alpha) * smoothedtemporalVae[i-1] =#
#= end =#

println("Stretching Data")
darlaDqn = stretch_to_500k(darlaDqn)
smoothedDarlaDqn = stretch_to_500k(smoothedDarlaDqn)
#= smoothedstylegan = stretch_to_500k(smoothedstylegan) =#
#= smoothedtemporalVae = stretch_to_500k(smoothedtemporalVae) =#

#= minLength = minimum([length(smoothedvanillaJaco) length(smoothedDarlaDqn) length(smoothedstylegan) length(smoothedtemporalVae)]) =#
minLength = minimum([length(darlaDqn) length(smoothedDarlaDqn)])
println("minLength is ", minLength)

darlaDqn = darlaDqn[1:minLength]
smoothedDarlaDqn = smoothedDarlaDqn[1:minLength]
#= smoothedstylegan = smoothedstylegan[1:minLength] =#
#= smoothedtemporalVae = smoothedtemporalVae[1:minLength] =#

println("Plotting Data")
x_data = range(1, stop=minLength)
#= plot(x_data, [smoothedvanillaJaco smoothedDarlaDqn smoothedstylegan smoothedtemporalVae], label=["Smoothed Vanilla Jaco" "Smoothed Visual Jaco" "Smoothed Stylegan" "Smoothed Temporal VAE"], xlabel="Number of Timesteps", ylabel="Reward", title="Reward vs Training Timestep", legend=:bottomright) =#
plot(x_data, [darlaDqn smoothedDarlaDqn], label=["Visual Jaco" "Smoothed Visual Jaco"], linealpha=[0.5 1], xlabel="Number of Timesteps", ylabel="Reward", title="Reward vs Training Timestep", legend=:bottomright)

savefig("losses.png")
