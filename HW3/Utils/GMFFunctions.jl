@with_kw struct GaussianMixtureBelief
    components::Array{FullNormal}  # Array of Gaussian components
    weights::Array{Float64, 1}     # Component weights
end

"""
    mixtureMoments(gmb::GaussianMixtureBelief)

Returns (μ̄, Σ̄) – the weighted mean and full covariance of the
Gaussian‐mixture belief

    μ̄ = ∑ wᵢ μᵢ
    Σ̄ = ∑ wᵢ [ Σᵢ + (μᵢ-μ̄)(μᵢ-μ̄)ᵀ ]
"""
function mixtureMoments(gmb::GaussianMixtureBelief)
    μ̄ = sum(gmb.weights[i] .* gmb.components[i].μ for i in eachindex(gmb.weights))
    Σ̄ = zeros(length(μ̄), length(μ̄))
    for i in eachindex(gmb.weights)
        μi  = gmb.components[i].μ
        Σi  = gmb.components[i].Σ
        w   = gmb.weights[i]
        Σ̄ .+= w .* (Σi .+ (μi-μ̄)*(μi-μ̄)')
    end
    return μ̄, Σ̄
end

"""
    printMixture(gmb::GaussianMixtureBelief; step::Int)

Prints every component’s weight and mean plus the overall weighted mean.
"""
function printMixture(gmb::GaussianMixtureBelief; step::Int)
    println("── GMF  step $step  ──")
    for (i,(comp,w)) in enumerate(zip(gmb.components, gmb.weights))
        println("  ▸ comp $i :  w=$(round(w,digits=4))  μ=$(round.(comp.μ; digits=3))")
    end
    μ̄, Σ̄ = mixtureMoments(gmb)

    # show overall mean and the two-by-two covariance nicely formatted
    println("  Σw μ  = ", round.(μ̄; digits=3))
    println("  Σw Σ = ")
    for r in 1:size(Σ̄,1)
        println("          ", round.(Σ̄[r,:]; digits=3))
    end
    println()               # blank line for readability
end

# ---------- helpers for adaptive growth -------------------------------

"Mahalanobis innovation – crude non-linearity score"
nonlinScore(comp::FullNormal, o, 𝒫::POMDPscenario) =
    let H = I(2),           # identity for simple range sensors
        S = H*comp.Σ*H' + 𝒫.Σv,
        y = o - H*comp.μ
    y' * inv(S) * y          # χ² distance
    end

"Split a component into two along its strongest axis"
function splitComponent(comp::FullNormal; c=0.6)
    λ, V = eigen(comp.Σ)
    δx   = c*sqrt(λ[1])*V[:,1]  # major axis
    δy   = c*sqrt(λ[2])*V[:,2]  # minor axis

    parts  = [MvNormal(comp.μ + δx, comp.Σ/4),
              MvNormal(comp.μ - δx, comp.Σ/4),
              MvNormal(comp.μ + δy, comp.Σ/4),
              MvNormal(comp.μ - δy, comp.Σ/4)]
    return parts, fill(0.25, 4)
end



function GetInitialGMBelief(μ0, Σ0; num_components::Int = 1, split_strategy::String = "sigma_points")
    if num_components == 1
        # Single component initialization
        component = MvNormal(μ0, Σ0)
        return GaussianMixtureBelief([component], [1.0])
    else
        # Multi-component initialization
        components = FullNormal[]
        weights = Float64[]
        
        if split_strategy == "grid"
            # Split based on grid positions (useful for your 9×9 grid)
            weight_per_component = 1.0 / num_components
            σ_split = sqrt(tr(Σ0) / num_components)  # Reduce variance per component
            
            for i in 1:num_components
                # Distribute means around the original mean
                offset_x = (i % 3 - 1) * 2.0  # -2, 0, 2 grid spacing
                offset_y = (div(i-1, 3) % 3 - 1) * 2.0
                
                μ_split = μ0 + [offset_x, offset_y]
                Σ_split = Σ0 / num_components  # Smaller covariance per component
                
                push!(components, MvNormal(μ_split, Σ_split))
                push!(weights, weight_per_component)
            end
        elseif split_strategy == "sigma_points"
            # Use sigma-point-like initialization for better coverage
            components, weights = CreateSigmaPointGMF(μ0, Σ0, num_components)
        end
        
        return GaussianMixtureBelief(components, weights)
    end
end



function PropagateBelief(gmb::GaussianMixtureBelief, 𝒫::POMDPscenario, a::Array{Float64, 1})
    new_components = FullNormal[]
    new_weights = Float64[]
    
    for (i, component) in enumerate(gmb.components)
        # Propagate each Gaussian component using standard Kalman prediction
        propagated = PropagateBelief(component, 𝒫, a)
        push!(new_components, propagated)
        push!(new_weights, gmb.weights[i])
    end
    
    return GaussianMixtureBelief(new_components, new_weights)
end


function PropagateUpdateBelief(gmb::GaussianMixtureBelief, 𝒫::POMDPscenario, a::Array{Float64, 1}, o::Array{Float64, 1})
    # First propagate
    predicted_gmb = PropagateBelief(gmb, 𝒫, a)
    
    new_components = FullNormal[]
    new_weights = Float64[]
    
    for (i, component) in enumerate(predicted_gmb.components)

        # Calculate likelihood weight for this component
        likelihood_weight = obsLikelihood(𝒫, o, component.μ)
        predicted_weight = predicted_gmb.weights[i]
        
        if likelihood_weight < 1e-10
            push!(new_components, component)  # Keep the component
            push!(new_weights, predicted_weight * 1e-10)  # Avoid zero weight
            continue
        end

        # Update each component with the observation
        updated_component = PropagateUpdateBelief(component, 𝒫, zeros(2), o)
        
        updated_weight = predicted_gmb.weights[i] * likelihood_weight
        
        if updated_weight > 1e-10  # Threshold for numerical stability
            push!(new_components, updated_component)
            push!(new_weights, updated_weight)
        end
    end
    
    # Normalize weights
    if !isempty(new_weights)
        new_weights ./= sum(new_weights)
    else
        # Fallback: return single component with uniform prior
        return GetInitialGMBelief([0.0, 0.0], [4.0 0.0; 0.0 4.0])
    end
    
    return GaussianMixtureBelief(new_components, new_weights)
end


function PruneComponents(gmb::GaussianMixtureBelief; threshold::Float64 = 0.01)
    kept_indices = findall(w -> w >= threshold, gmb.weights)
    
    if isempty(kept_indices)
        # Keep the component with maximum weight
        max_idx = argmax(gmb.weights)
        kept_indices = [max_idx]
    end
    
    new_components = gmb.components[kept_indices]
    new_weights = gmb.weights[kept_indices]
    new_weights ./= sum(new_weights)  # Renormalize
    
    return GaussianMixtureBelief(new_components, new_weights)
end



function MergeComponents(gmb::GaussianMixtureBelief; max_components::Int = 5)
    if length(gmb.components) <= max_components
        return gmb
    end
    
    components = copy(gmb.components)
    weights = copy(gmb.weights)
    
    while length(components) > max_components
        # Find pair with minimum KL divergence
        min_cost = Inf
        merge_i, merge_j = 1, 2
        
        for i in 1:(length(components)-1)
            for j in (i+1):length(components)
                cost = KLDivergence(components[i], components[j], weights[i], weights[j])
                if cost < min_cost
                    min_cost = cost
                    merge_i, merge_j = i, j
                end
            end
        end
        
        # Merge the two closest components
        merged_component, merged_weight = MergeTwoGaussians(
            components[merge_i], components[merge_j], 
            weights[merge_i], weights[merge_j]
        )
        
        # Remove old components and add merged one
        deleteat!(components, max(merge_i, merge_j))
        deleteat!(components, min(merge_i, merge_j))
        deleteat!(weights, max(merge_i, merge_j))
        deleteat!(weights, min(merge_i, merge_j))
        
        push!(components, merged_component)
        push!(weights, merged_weight)
    end
    
    return GaussianMixtureBelief(components, weights)
end

function MergeTwoGaussians(g1::FullNormal, g2::FullNormal, w1::Float64, w2::Float64)
    w_total = w1 + w2
    
    # Merged mean (weighted average)
    μ_merged = (w1 * g1.μ + w2 * g2.μ) / w_total
    
    # Merged covariance (moment matching)
    Σ_merged = (w1 * (g1.Σ + (g1.μ - μ_merged) * (g1.μ - μ_merged)') + 
                w2 * (g2.Σ + (g2.μ - μ_merged) * (g2.μ - μ_merged)')) / w_total
    
    return MvNormal(μ_merged, Σ_merged), w_total
end


function SplitComponent(component::FullNormal, num_splits::Int = 4)
    # Split a single Gaussian into multiple components for better nonlinearity handling
    μ = component.μ
    Σ = component.Σ
    
    components = FullNormal[]
    weights = Float64[]
    
    # Create sigma points around the mean
    n = length(μ)
    λ = 2.0  # Scaling parameter
    
    # Central component
    push!(components, component)
    push!(weights, 0.5)
    
    # Surrounding components
    for i in 1:n
        offset = sqrt((n + λ) * Σ[i,i]) * [j == i ? 1.0 : 0.0 for j in 1:n]
        
        # Positive direction
        μ_pos = μ + offset
        Σ_split = Σ / (num_splits / 2)  # Reduce covariance
        push!(components, MvNormal(μ_pos, Σ_split))
        push!(weights, 0.25 / n)
        
        # Negative direction  
        μ_neg = μ - offset
        push!(components, MvNormal(μ_neg, Σ_split))
        push!(weights, 0.25 / n)
    end
    
    # Normalize weights
    weights ./= sum(weights)
    
    return GaussianMixtureBelief(components, weights)
end


# Add these visualization functions to GMFFunctions.jl

function PlotGMFResults(𝒫::POMDPscenario, τ, τbGMF)
    fig = plot(size=(800,600))

    # beacons
    scatter!(fig, 𝒫.beacons[:,1], 𝒫.beacons[:,2];
             m=:utriangle, c=:red, ms=8, label="Beacons")

    # ground truth as blue dots
    scatter!(fig, getindex.(τ,1), getindex.(τ,2);
             m=:circle, c=:blue, ms=4, label="Ground truth")

    T = length(τbGMF)
    for t in 1:T
        gmb = τbGMF[t]
        fade = 0.15 + 0.8*t/T            # early → very light, final → darker

        # 2.a  all components (faded)
        for (i,comp) in enumerate(gmb.components)
            scatter!(fig, [comp.μ[1]], [comp.μ[2]];
                     m=:circle, ms=max(3,35*gmb.weights[i]),
                     c=:green, alpha=0.30*fade,
                     label = (t==1 && i==1 ? "Component means" : ""))

            ex, ey = drawCovarianceEllipse(comp.μ, comp.Σ, 2)
            plot!(fig, ex, ey; c=:green, alpha=0.15*fade, lw=1, label="")
        end

        # 2.b  weighted posterior (mixture moments)
        μ̄, Σ̄ = mixtureMoments(gmb)
        scatter!(fig, [μ̄[1]], [μ̄[2]];
                 m=:star5, ms=8,  c=:black, alpha=fade,
                 label = (t==1 ? "Posterior mean" : ""))

        exM, eyM = drawCovarianceEllipse(μ̄, Σ̄, 2)
        plot!(fig, exM, eyM; c=:black, alpha=0.15*fade, lw=1.5,
              linestyle=:dash, label=(t==1 ? "Posterior 2σ" : ""))
    end

    title!(fig, "GMF Posterior Evolution")
    xlabel!(fig, "X position")
    ylabel!(fig, "Y position")
    savefig(fig, "Results/GMF_MixtureComponents.pdf")
    return fig
end





function SaveGMFEvolution(τbGMF, T)
    # Save detailed evolution for analysis
    open("Results/GMF_Evolution.txt", "w") do f
        for t in 1:T
            gmb = τbGMF[t]
            println(f, "Time step $t:")
            println(f, "  Number of components: $(length(gmb.components))")
            for (i, (comp, weight)) in enumerate(zip(gmb.components, gmb.weights))
                println(f, "  Component $i: weight=$(round(weight, digits=4)), mean=$(comp.μ)")
            end
            println(f, "")
        end
    end
end



# GMF-specific versions of the calculation functions
function CalculateDeadReckoningBeliefGMF(𝒫::POMDPscenario, b0GMF::GaussianMixtureBelief, ak_array, T)
    τbdr = Array{GaussianMixtureBelief}(undef, T)
    τbdr[1] = b0GMF
    
    for t in 2:T
        # FIX: Use specific action for each time step
        τbdr[t] = PropagateBelief(τbdr[t-1], 𝒫, ak_array[t-1])
        
        # Apply pruning and merging to manage complexity
        τbdr[t] = PruneComponents(τbdr[t])
        τbdr[t] = MergeComponents(τbdr[t], max_components=5)
    end
    
    return τbdr
end

function CalculatePosteriorBeliefGMF(𝒫::POMDPscenario, b0GMF::GaussianMixtureBelief,
                                     ak, T, τobs)

    τb = Vector{GaussianMixtureBelief}(undef, T)
    τb[1] = b0GMF

    εsplit   = 10.0          # non-linearity threshold
    Nmax     = 8           # max components after merge
    wprune   = 1e-3         # prune threshold

    for t in 2:T
        # ---------------- growth (splitting) --------------------------
        gmb_grow = GaussianMixtureBelief([], Float64[])
        for (comp,w) in zip(τb[t-1].components, τb[t-1].weights)
            if (τobs[t] !== nothing) && (nonlinScore(comp, τobs[t].obs, 𝒫) > εsplit)
                parts, loc_w = splitComponent(comp)
                append!(gmb_grow.components, parts)
                append!(gmb_grow.weights,    w .* loc_w)
            else
                push!(gmb_grow.components, comp)
                push!(gmb_grow.weights,    w)
            end
        end
        gmb_grow.weights ./= sum(gmb_grow.weights)   # normalise

        # -------------- predict & (optional) update -------------------
        if τobs[t] === nothing
            gmb_pred = PropagateBelief(gmb_grow, 𝒫, ak[t-1])
            gmb_upd  = gmb_pred                         # no observation
        else
            obs_corr = τobs[t].obs .+ 𝒫.beacons[τobs[t].index,:]
            gmb_upd  = PropagateUpdateBelief(gmb_grow, 𝒫, ak[t-1], obs_corr)
        end

        # -------------- prune & merge (shrink) ------------------------
        gmb_upd  = PruneComponents(gmb_upd;  threshold = wprune)
        gmb_upd  = MergeComponents(gmb_upd; max_components = Nmax)

        τb[t] = gmb_upd
    end
    return τb
end


# Comparison function to analyze filter performance
function PrintFilterComparison(τ_true, τbG, τbP, τbGMF, T)
    println("\n=== FILTER PERFORMANCE COMPARISON ===")
    
    for t in 1:T
        true_pos = τ_true[t]
        
        # Gaussian filter error
        gauss_error = norm(τbG[t].μ - true_pos)
        
        # Particle filter error (using weighted mean)
        pf_mean = sum(τbP[t].weights[i] * τbP[t].particles[i] for i in 1:length(τbP[t].particles))
        pf_error = norm(pf_mean - true_pos)
        
        # GMF error (using weighted mean of mixture)
        gmf_mean = sum(τbGMF[t].weights[i] * τbGMF[t].components[i].μ for i in 1:length(τbGMF[t].components))
        gmf_error = norm(gmf_mean - true_pos)
        
        println("Step $t: G_err=$(round(gauss_error,digits=3)), PF_err=$(round(pf_error,digits=3)), GMF_err=$(round(gmf_error,digits=3))")
    end
end


# Add this function to GMFFunctions.jl
function KLDivergence(g1::FullNormal, g2::FullNormal, w1::Float64, w2::Float64)
    # Simplified KL divergence approximation for merging decision
    # Using Mahalanobis distance as a proxy
    μ_diff = g1.μ - g2.μ
    Σ_avg = (w1 * g1.Σ + w2 * g2.Σ) / (w1 + w2)
    
    try
        distance = sqrt(μ_diff' * inv(Σ_avg) * μ_diff)
        return distance
    catch
        # Fallback if matrix inversion fails
        return norm(μ_diff)
    end
end


function CreateSigmaPointGMF(μ0, Σ0, num_components)
    # Simple implementation - distribute components around the mean
    components = FullNormal[]
    weights = Float64[]
    
    weight_per_component = 1.0 / num_components
    
    for i in 1:num_components
        # Create slight variations around the mean
        angle = 2π * (i-1) / num_components
        radius = 1.0
        offset = radius * [cos(angle), sin(angle)]
        
        μ_comp = μ0 + offset
        Σ_comp = Σ0 / num_components
        
        push!(components, MvNormal(μ_comp, Σ_comp))
        push!(weights, weight_per_component)
    end
    
    return components, weights
end
