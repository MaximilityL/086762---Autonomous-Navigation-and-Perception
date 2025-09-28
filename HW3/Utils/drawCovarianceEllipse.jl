using LinearAlgebra
function drawCovarianceEllipse(x, P, num_of_std=1)
    # covarianceEllipse: plot a Gaussian as an uncertainty ellipse
    # Based on Maybeck Vol 1, page 366
    #
    # num_of_std is an *optional* parameter that specifies how many standard
    # deviations to draw: either 1 or 3. This parameter overrides k.
    #
    # Slightly modifed from covarianceEllipse by Vadim Indelman, July 2014. 
    
    # k=2.296 corresponds to 1 std, 68.26% of all probability
    # k=11.82 corresponds to 3 std, 99.74% of all probability
    
    k = 2.296;
    
    if num_of_std == 1
        k = 2.296
    elseif num_of_std == 3
        k = 11.82
    end
    
    
    s, e = eigen(P)
    s1 = s[1]
    s2 = s[2]
    ex, ey = ellipse(sqrt(s1*k).*e[:,1], sqrt(s2*k).*e[:,2], x[1:2])
    return ex, ey
end


function ellipse(a,b,c)
    # ellipse: return the x and y coordinates for an ellipse
    # a, and b are the axes. c is the center
    q =collect(0:2*pi/25:2*pi)
    ellipse_x = cos.(q)
    ellipse_y = sin.(q)
    
    
    points = a.*transpose(ellipse_x) .+ b.*transpose(ellipse_y)
    x = c[1] .+ points[1,:]
    y = c[2] .+ points[2,:]
    
    return x, y
end