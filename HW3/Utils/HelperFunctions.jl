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


function InitiatePOMDPScenario()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1)
    
    d =2.0 
    rmin = 0.1
    # set beacons locations 
    beacons =  [0.0 0.0; 0.0 4.0; 0.0 8.0; 4.0 0.0; 4.0 4.0; 4.0 8.0; 8.0 0.0; 8.0 4.0; 8.0 8.0]

    𝒫 = POMDPscenario(F=[1.0 0.0; 0.0 1.0],
                    Σw=0.1^2*[1.0 0.0; 0.0 1.0],
                    Σv=[1.0 0.0; 0.0 1.0], 
                    rng = rng , beacons=beacons, d=d, rmin=rmin) 

    bplot =  scatter(𝒫.beacons[:, 1], 𝒫.beacons[:, 2], label="beacons", markershape=:utriangle)
    savefig(bplot,"Results/Qf-Beacons.pdf")

    return 𝒫
end

function ScatterPlotParticleBelief(𝒫::POMDPscenario, GroundTruthTrajectory, BeliefTrajectory, T, pdfFileName)
    TrajectoryFigure = scatter([x[1] for x in GroundTruthTrajectory], [x[2] for x in GroundTruthTrajectory], label="Ground Truth")
    scatter!(𝒫.beacons[:, 1], 𝒫.beacons[:, 2], label="beacons", markershape=:utriangle)
    for i in 1:T
        scatterParticles(BeliefTrajectory[i], "$i")
    end
    savefig(TrajectoryFigure,pdfFileName)
end

function ScatterPlotGaussianBelief(𝒫::POMDPscenario, GroundTruthTrajectory, GaussianBelief, T, pdfFileName; useLegend = false)

    BeliefFigure = plot()
    
    BeliefFigure =scatter(𝒫.beacons[:, 1], 𝒫.beacons[:, 2], label="Beacons", markershape=:utriangle, color=:red)

    if useLegend
        BeliefFigure=scatter!([x[1] for x in GroundTruthTrajectory], [x[2] for x in GroundTruthTrajectory])
        for i in 1:T
            covellipse!(GaussianBelief[i].μ, GaussianBelief[i].Σ, showaxes=true, n_std=3, label="step $i")
        end
    else
        BeliefFigure=scatter!([x[1] for x in GroundTruthTrajectory], [x[2] for x in GroundTruthTrajectory], label=false, legend=false)
        for i in 1:T
            covellipse!(GaussianBelief[i].μ, GaussianBelief[i].Σ, showaxes=true, n_std=3, label=false, legend=false)
        end
    end

    savefig(BeliefFigure,pdfFileName)
end

function ScatterPlotCombinedBelief(𝒫::POMDPscenario, GroundTruthTrajectory, ParticleBeliefTrajectory, GaussianBelief, T, pdfFileName; useLegend = false)
    CombinedFigure = plot()

    # Plot beacons
    CombinedFigure = scatter(CombinedFigure, 𝒫.beacons[:, 1], 𝒫.beacons[:, 2], label="Beacons", markershape=:utriangle, color=:red)

    # Plot ground truth trajectory
    scatter!(CombinedFigure, [x[1] for x in GroundTruthTrajectory], [x[2] for x in GroundTruthTrajectory], label="Ground Truth", color=:blue)

    # Plot particle belief trajectory
    for i in 1:T
        scatterParticles(ParticleBeliefTrajectory[i], "$i")
    end

    # Plot Gaussian belief ellipses
    if useLegend
        for i in 1:T
            covellipse!(CombinedFigure, GaussianBelief[i].μ, GaussianBelief[i].Σ, showaxes=true, n_std=3, label="Gaussian Belief $i")
        end
    else
        for i in 1:T
            covellipse!(CombinedFigure, GaussianBelief[i].μ, GaussianBelief[i].Σ, showaxes=true, n_std=3, label=false, legend=false)
        end
    end

    # Save the combined figure
    savefig(CombinedFigure, pdfFileName)
end

function GenerateBeacons(n::Int)
    gridSize = round(Int, sqrt(n))  # Assuming the beacons are placed in a grid

    x = repeat(LinRange(0, n, gridSize), gridSize)  # Repeat the x-coordinates
    y = repeat(LinRange(0, n, gridSize), inner = gridSize)  # Repeat the y-coordinates

    beacons = hcat(x, y)

    return beacons
end

function scatterParticles(belief::ParticleBelief, label::String)
    x = [p[1] for p in belief.particles]
    y = [p[2] for p in belief.particles]
    w = belief.weights
    TrajectoryFigure = scatter!(x, y, markersize=w .*50, markercolor=:auto, markerstrokewidth=0, alpha=0.5, label=label)
    return TrajectoryFigure
end

function obsLikelihood(𝒫::POMDPscenario, o::Array{Float64, 1}, x::Array{Float64, 1})::Float64
    expectedObs = GenerateObservationFromBeacons(𝒫, x)
    if expectedObs === nothing
        return 0.0
    else
        μ_z = expectedObs.obs
        return pdf(MvNormal(μ_z, 𝒫.Σv), o)
    end
end

function SampleMotionModel(𝒫::POMDPscenario, a::Array{Float64, 1}, x::Array{Float64, 1})
    F = 𝒫.F
    Σw = 𝒫.Σw
    # Process noise
    w = rand(MvNormal(zeros(length(x)), Σw))
    # Motion model
    x_next = F * x + a + w
    return x_next
end

function GenerateObservationFromBeacons(𝒫::POMDPscenario, x::Array{Float64, 1}; addNoise::Bool = false)::Union{NamedTuple, Nothing}
    # Step I: Calculate the Euclidean distance from each beacon
    distances = [sqrt(sum((x .- 𝒫.beacons[i, :]).^2)) for i in 1:size(𝒫.beacons, 1)]
    
    # Step II: Find the shortest distance under the threshold
    minDistance = minimum(distances)
    if minDistance <= 𝒫.d
        index = argmin(distances)
        
        vtag = addNoise ? rand(MvNormal(zeros(length(x)), 𝒫.Σv)) : zeros(length(x))

        # Step III: Create and return the observation with or without the noise
        obs = x - 𝒫.beacons[index, :] + vtag   # relative observation

        return (obs=obs, Σ=𝒫.Σv, index=index)
    end
    
    # If no distances are under the threshold, return nothing
    return nothing    
end

function CalculateMotionTrajectory(𝒫::POMDPscenario, ak, T, τ0)
    τ = τ0
    for i in 1:T-1
        push!(τ, SampleMotionModel(𝒫, ak, τ[end]))
    end
    return τ
end

function CalculateObservationTrajectory(𝒫::POMDPscenario, τ, T, τobs0)
        τobs = τobs0
        for i in 1:T
            push!(τobs, GenerateObservationFromBeacons(𝒫, τ[i]; addNoise = true))
        end 
    return τobs
end

function CalculateDeadReckoningBelief(𝒫::POMDPscenario, b0, ak::Array{Float64, 1}, T)
    τbp = [b0]
    for i in 1:T-1
        push!(τbp, PropagateBelief(τbp[end],  𝒫, ak))
    end
    return τbp
end

function CalculatePosteriorBelief(𝒫::POMDPscenario, b0::FullNormal, ak::Array{Float64, 1}, T, τobs)
    τb = [b0]

    for i in 1:T-1
        if τobs[i+1] === nothing
            # Perform an alternative action when τobs[i+1] is nothing
            push!(τb, PropagateBelief(τb[end], 𝒫, ak))
        else
            push!(τb, PropagateUpdateBelief(τb[end], 𝒫, ak, (τobs[i+1].obs .+ 𝒫.beacons[τobs[i+1].index,:])))
        end
    end

    return τb
end

function CalculatePosteriorBelief(𝒫::POMDPscenario, b0::ParticleBelief, ak::Array{Float64, 1}, T, τobs; useResampling::Bool = true, useLowVarianceResampling::Bool = false)
    τbr = [b0]
    
    for i in 1:T-1
        if τobs[i+1] === nothing
            push!(τbr, PropagateBelief(τbr[end],  𝒫, ak))
        else
            b = PropagateUpdateBelief(τbr[end],  𝒫, ak, τobs[i+1].obs)
            if (useResampling)
                if (useLowVarianceResampling)
                    b = LowVarianceResampleParticles(b)
                else
                    b = ResampleParticles(b)  
                end
            end
            push!(τbr, b)
        end
    end

    return τbr
end


function printQfClauseiiiPlots(𝒫::POMDPscenario, τ, τbdrP, τbdrG, T)
    #Plot the Particle Belief Dead Reckoning
    ScatterPlotParticleBelief(𝒫, τ, τbdrP, T, "Results/Qfiii-DeadReckoningParticleBeliefTrajectory.pdf")

    #Plot the Gaussian Belief Dead Reckoning
    ScatterPlotGaussianBelief(𝒫, τ, τbdrG, T ,"Results/Qfiii-DeadReckoningGaussianBeliefTrajectory.pdf")

    #Plot the Guassian and Particle Belief
    ScatterPlotCombinedBelief(𝒫, τ, τbdrP, τbdrG, T, "Results/Qfiii-DeadReckoningGaussianAndParticleBeliefTrajectory.pdf"; useLegend = false)
end

function printQfClauseiVPlots(𝒫::POMDPscenario, τ, τbPwor, τbB, T)
    #Plot the Particle Belief Posterior
    ScatterPlotParticleBelief(𝒫, τ, τbPwor, T, "Results/QfiV-PosteriorParticleBeliefNoResampleTrajectory.pdf")

    #Plot the Gaussian Belief Dead Reckoning
    ScatterPlotGaussianBelief(𝒫, τ, τbB, T ,"Results/QfiV-PosteriorGaussianBeliefTrajectory.pdf")

    #Plot the Guassian and Particle Belief
    ScatterPlotCombinedBelief(𝒫, τ, τbPwor, τbB, T, "Results/QfiV-PosteriorGaussianAndParticleBeliefNoResampleTrajectory.pdf"; useLegend = false)
end

function printQfClauseVPlots(𝒫::POMDPscenario, τ, τbPwr, τbB, T)
    #Plot the Particle Belief Posterior
    ScatterPlotParticleBelief(𝒫, τ, τbPwr, T, "Results/QfV-PosteriorParticleBeliefResampleTrajectory.pdf")

    #Plot the Gaussian Belief Dead Reckoning
    ScatterPlotGaussianBelief(𝒫, τ, τbB, T ,"Results/QfV-PosteriorGaussianBeliefTrajectory.pdf")

    #Plot the Guassian and Particle Belief
    ScatterPlotCombinedBelief(𝒫, τ, τbPwr, τbB, T, "Results/QfV-PosteriorGaussianAndParticleBeliefResampleTrajectory.pdf"; useLegend = false)
end

function ExecuteQLogic(𝒫::POMDPscenario, b0G::FullNormal, b0P::ParticleBelief, b0GMF::GaussianMixtureBelief, T)
    # Create Results directory using absolute path
    resultsPath = joinpath(dirname(@__FILE__), "..", "Results")
    if !isdir(resultsPath)
        mkpath(resultsPath)
    end
    
    ## Initialization of Question f
    # Defining Question Data
    xgt0 = [-0.5, -0.2]           
    ak = [1.5, 1.5]  

    # Initiating Trajectories
    τ0 = [xgt0]      
    τobsbeacons0 = []


    ## Generating Motion trajectory - Clause i
    println("\n[ExecuteQLogic] Calculating Motion Trajectory...")
    τ = CalculateMotionTrajectory(𝒫, ak, T, τ0)
    println("[ExecuteQLogic] Motion Trajectory calculation completed.")

    ## Generating Observation Trajectory - Clause ii
    println("\n[ExecuteQLogic] Calculating Observation Trajectory...")
    τobsbeacons = CalculateObservationTrajectory(𝒫, τ, T, τobsbeacons0)      
    println("[ExecuteQLogic] Observation Trajectory calculation completed.")
    println("[ExecuteQLogic] Observation trajectory:", τobsbeacons)

    # ## Dead Reckoning Belief Calculation - Guassian & Particle Belief - Clause iii
    # # Dead Reckoning Particle Belief Calculation
    # println("\n[ExecuteQLogic] Calculating Dead Reckoning Particle Belief...")
    # τbdrP = CalculateDeadReckoningBelief(𝒫, b0P, ak, T)
    # println("[ExecuteQLogic] Dead Reckoning Particle Belief calculation completed.")

    # # Dead Reckoning Gaussian Belief Calculation
    # println("\n[ExecuteQLogic] Calculating Dead Reckoning Gaussian Belief...")
    # τbdrG = CalculateDeadReckoningBelief(𝒫, b0G, ak, T)
    # println("[ExecuteQLogic] Dead Reckoning Gaussian Belief calculation completed.")

    # # Print The required plots for clause iii
    # printQfClauseiiiPlots(𝒫, τ, τbdrP, τbdrG, T)

    # ## Posterior Calculation (Using Observation Data) - Partile Filter and Guassian Belief - Clause iV and V
    # # Particle Filter Posterior Belief Calculation - Without Resampling
    # println("\n[ExecuteQLogic] Calculating Posterior Belief without resampling...")
    # τbPwor = CalculatePosteriorBelief(𝒫, b0P, ak, T, τobsbeacons; useResampling = false)
    # println("[ExecuteQLogic] Posterior Belief without resampling calculation completed.")

    # # Particle Filter Posterior Belief Calculation - With Resampling
    # println("\n[ExecuteQLogic] Calculating Posterior Belief with resampling...")
    # τbPwr = CalculatePosteriorBelief(𝒫, b0P, ak, T, τobsbeacons; useResampling = true)
    # println("[ExecuteQLogic] Posterior Belief with resampling calculation completed.")
    
    # # Gaussian Posterior Belief Calculation
    # τbB = CalculatePosteriorBelief(𝒫, b0G, ak, T, τobsbeacons)

    # # Print the required plot for clause iV
    # printQfClauseiVPlots(𝒫, τ, τbPwor, τbB, T)

    # # Print the required plot for clause V
    # printQfClauseVPlots(𝒫, τ, τbPwr, τbB, T)

    # ## Posterior Calculation (Using Observation Data) - Particle Filter With LowVarianceResampling
    # println("\n[ExecuteQLogic] Calculating Posterior Belief with low variance resampling...")
    # τbPwrLV = CalculatePosteriorBelief(𝒫, b0P, ak, T, τobsbeacons; useResampling = true, useLowVarianceResampling = true)
    # println("[ExecuteQLogic] Posterior Belief with low variance resampling calculation completed.")

    # #Plot the Guassian and Particle Belief
    # ScatterPlotCombinedBelief(𝒫, τ, τbPwrLV, τbB, T, "Results/Qg-PosteriorGaussianAndParticleBeliefLowVarianceResampleTrajectory.pdf"; useLegend = false)

    ###### GMF UPDATES (NEW SECTION) ######
    println("GMF Dead Reckoning...")
    # FIX: Convert single action to array for multiple time steps
    ak_array = [ak for _ in 1:T-1]  # Create array of actions
    τbdrGMF = CalculateDeadReckoningBeliefGMF(𝒫, b0GMF, ak_array, T)
    
    println("GMF Posterior...")
    τbGMF = CalculatePosteriorBeliefGMF(𝒫, b0GMF, ak_array, T, τobsbeacons)

    ###### PLOTTING ######
    println("Plotting GMF Results...")
    PlotGMFResults(𝒫, τ, τbGMF)
    
    # FIX: Use correct variable names
    # PrintFilterComparison(τ, τbB, τbPwr, τbGMF, T)  # Changed τbG to τbB, τbP to τbPwr
end

function BenchmarkResampling()
    # Sample data for testing
    nParticles = 1000
    particles = [rand(3) for _ in 1:nParticles]
    weights = [rand() for _ in 1:nParticles]
    weights /= sum(weights)
    belief = ParticleBelief(particles, weights)

    # Benchmark the existing resampling function
    println("Benchmarking ResampleParticles:")
    resample_benchmark = @benchmark ResampleParticles($belief)

    # Benchmark the new low variance resampling function
    println("Benchmarking LowVarianceResampleParticles:")
    low_variance_benchmark = @benchmark LowVarianceResampleParticles($belief)

    # Print the results
    println("ResampleParticles Benchmark Results:")
    display(resample_benchmark)

    println("LowVarianceResampleParticles Benchmark Results:")
    display(low_variance_benchmark)

    return resample_benchmark, low_variance_benchmark
end