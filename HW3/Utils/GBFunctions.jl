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


function GetInitialGaussianBelief(μ0, Σ0)
    return MvNormal(μ0, Σ0)
end

function PropagateBelief(b::FullNormal, 𝒫::POMDPscenario, a::Array{Float64, 1})::FullNormal
    μb, Σb = b.μ, b.Σ
    F  = 𝒫.F
    Σw, Σv = 𝒫.Σw, 𝒫.Σv
    I = [1.0 0.0; 0.0 1.0]
    Ts = F
    Ta = I
    # predict
    μp = Ts * μb + Ta * a
    Σp = Ts * Σb * transpose(Ts) + Σw
    return MvNormal(μp, Σp)
end 

function PropagateUpdateBelief(b::FullNormal, 𝒫::POMDPscenario, a::Array{Float64, 1}, o::Array{Float64, 1})::FullNormal
    μb, Σb = b.μ, b.Σ
    F  = 𝒫.F
    Σw, Σv = 𝒫.Σw, 𝒫.Σv
    I = [1.0 0.0; 0.0 1.0]
    Ts = F
    Ta = I
    Os = I
    # predict
    predictedBelief = PropagateBelief(b, 𝒫, a)
    μp = predictedBelief.μ
    Σp = predictedBelief.Σ
    # update
    K = (Σp * transpose(Os)) / (Os * Σp * transpose(Os) + Σv)
    μb′ = μp + K * (o - (Os * μp))
    Σb′ = (I - (K * Os)) * Σp
    return MvNormal(μb′, Σb′)
end 