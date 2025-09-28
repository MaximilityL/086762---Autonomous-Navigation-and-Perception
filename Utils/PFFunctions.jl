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

function GetInitialParticleBelief(μ0, Σ0)
    n_particles = 10
    b0 = InitParticleBelief(n_particles, μ0, Σ0)
    return b0
end

function InitParticleBelief(nParticles::Int, μ0::Array{Float64,1}, Σ0::Array{Float64,2})::ParticleBelief
    particles = [rand(MvNormal(μ0, Σ0)) for _ in 1:nParticles]
    weights = fill(1.0 / nParticles, nParticles)
    return ParticleBelief(particles, weights)
end

function PropagateBelief(b::ParticleBelief, 𝒫::POMDPscenario, a::Array{Float64, 1})::ParticleBelief
    newParticles = [SampleMotionModel(𝒫, a, p) for p in b.particles]
    newWeights = b.weights 
    return ParticleBelief(newParticles, newWeights)
end

function PropagateUpdateBelief(b::ParticleBelief, 𝒫::POMDPscenario, a::Array{Float64, 1}, o::Array{Float64, 1})::ParticleBelief
    # Step I: Propagate particles using PropagateBelief function
    propagatedBelief = PropagateBelief(b, 𝒫, a)
    newParticles = propagatedBelief.particles
    oldWeights = propagatedBelief.weights

    
    # Step II: Update weights based on observation likelihood
    newWeights = [obsLikelihood(𝒫, o, p) * oldWeights[i] for (i, p) in enumerate(newParticles)]
    
    # Normalize the weights
    totalWeight = sum(newWeights)
    newWeights = totalWeight == 0.0 ? b.weights : newWeights ./ totalWeight
    
    return ParticleBelief(newParticles, newWeights)
end

function ResampleParticles(b::ParticleBelief)::ParticleBelief
    nParticles = length(b.particles)
    
    # Sample indices based on weights
    indices = sample(1:nParticles, Weights(b.weights), nParticles; replace=true)
    
    # Create new particles based on sampled indices
    newParticles = [b.particles[i] for i in indices]
    
    # Reset weights to uniform distribution
    newWeights = fill(1.0 / nParticles, nParticles)
    
    return ParticleBelief(newParticles, newWeights)
end

function LowVarianceResampleParticles(b::ParticleBelief)::ParticleBelief
    nParticles = length(b.particles)
    
    # Initialize
    newParticles = Vector{Array{Float64, 1}}(undef, nParticles)
    newWeights = fill(1.0 / nParticles, nParticles)
    
    # Draw a starting point
    r = rand() / nParticles
    c = b.weights[1]
    i = 1
    
    for m = 1:nParticles
        U = r + (m - 1) / nParticles
        while U > c
            i += 1
            c += b.weights[i]
        end
        newParticles[m] = b.particles[i]
    end
    
    return ParticleBelief(newParticles, newWeights)
end
