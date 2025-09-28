using Revise
using Distributions
using Random
using LinearAlgebra
using Plots
using StatsPlots
using Parameters
using StatsBase
using BenchmarkTools

includet("Utils/drawCovarianceEllipse.jl") 
includet("Utils/GBFunctions.jl")
includet("Utils/PFFunctions.jl")
includet("Utils/HelperFunctions.jl")

@with_kw mutable struct POMDPscenario
    F::Array{Float64, 2}   
    Σw::Array{Float64, 2}
    Σv::Array{Float64, 2}
    rng::MersenneTwister
    beacons::Array{Float64, 2}
    d::Float64
    rmin::Float64
end

@with_kw struct ParticleBelief
    particles::Array{Array{Float64, 1}}
    weights::Array{Float64, 1}
end



function main()

    𝒫 = InitiatePOMDPScenario()

    # Initialize Belief
    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0; 0.0 1.0]

    b0P = GetInitialParticleBelief(μ0, Σ0)
    b0G = GetInitialGaussianBelief(μ0, Σ0)

    # Initialize Step count
    T = 6 
    ExecuteQLogic(𝒫, b0G, b0P, T)

    BenchmarkResampling()

end 

main()