#= Krussel-Smith using sequence-space Jacobian.

Code will have three parts:

(a) Decompose model into blocks. Same as KS example in Auclert et al., but with endogenous labour supply.

(b) Finding steady-state values of KS (no calibration). Need to translate ks_ss into julia and add labour supply choice.

(c) Accumulating Jacobians. In principle straightforward once we have done everything else. Need to get 300x300 matrices
Jacobians of each output with respect to each input. For example: in firm block, need matrix of how "w" and "r" change
with respect to "Z", "K", and "L". Store these in dictionary.

Extra: depending on how things go, can add calibration step, estimation step, or non-linear dynamics. For more complicated
models, can supply jacobian of "hetblock" to SHADE code directly in dictionary form and let their code do the accumulation
automatically.

Can use parameters from Krussel-Smith parameters in code. Other parameter would be elasticity "v" and disutility "ϕ" of labour.
Can use something reasonable.

=#

#TODOs

## have interpolation routines in-house (interest rate is sensitive to type of interpolation method used).

#finding the steady-state: household block

#specification is similar

#Parameter struct

module functions_nolabour

using Parameters
using LinearAlgebra
using Dierckx #interpolation routines written in Fortran
using NLsolve
using Roots
using Plots

export hh_ss!, het_jac, Param



@with_kw mutable struct Param
    apoints::Int = 500 #asset grid
    amax::Float64 = 200  #asset max
    beta::Float64 = 0.9819527880123726 #discount factor
    alpha::Float64 = 0.11 #capital share
    deprec::Float64 = 0.025 #depreciation rate
    gamma::Float64 = 1 #elasticity of substitution
    rho::Float64 = 0.966  #autocorr of income process
    num_states::Int = 7 #number of states for income process
    sd::Float64 = 0.5 #stand. dev. of deviation process
    mean::Float64 = 0 #mean of income process
    uncond_sd::Float64 = 0.5 # unconditional sd
    y_grid::Array{Float64} = zeros() # grid for income process
    P_trans::Array{Float64} = zeros() # transition matrix
    bc::Int = 0 #borrowing constraint
    # frisch::Float64 = 2/3 #elasticity of labour wrt wages
    Amat::Array{Float64} = zeros() #asset grid
    Ymat::Array{Float64}= zeros() #income grid
    # phi::Float64 = 1 #constant in front of disutility of labour
end

# param = Param()


#rowenhort's method to approximate AR(1) w Markov chain
#also constructs asset grids and updates param

# function rowenhorst!(param::Param)
#     rho, sd, num_states, mean = param.rho, param.uncond_sd, param.num_states, param.mean
#
#     bc, apoints, amax = param.bc, param.apoints, param.amax
#     #construct grids
#
#     step_r = sd*sqrt(num_states-1)
#     y_grid = -1:2/(num_states-1):1
#     y_grid = mean .+ step_r*y_grid
#
#     #transition matrix
#
#     p = (rho+1)/2
#     q = p
#
#     P_trans = [p 1-p; 1-q q]
#
#     for i = 2:num_states -1
#         a1 = [P_trans zeros(i, 1); zeros(1, i+1)]
#         a2 = [zeros(i,1) P_trans; zeros(1, i+1)]
#         a3 = [zeros(1, i+1); P_trans zeros(i,1)]
#         a4 = [zeros(1, i+1); zeros(i,1) P_trans]
#
#         P_trans = p*a1 + (1-p)*a2 + (1-q)*a3 + q*a4
#         P_trans[2:i, :] = P_trans[2:i, :]/2
#     end
#
#
#     for i=1:num_states
#        P_trans[i,:] = P_trans[i,:]/sum(P_trans[i,:])
#     end
#
#     #get stationary distribution to normalize
#     #effective labour to L=1
#
#     pi = eigvecs(P_trans')[:,num_states]
#
#     #normalize pi
#
#     pi = pi./sum(pi)
#
#     #exponentiate
#
#     y_grid = exp.(y_grid)
#
#     #normalize
#
#     y_grid = y_grid/sum(pi.*y_grid)
#
#     param.y_grid = y_grid
#     param.P_trans = P_trans
#
#     # construct asset grids
#
#     Amat = [i for i in range(bc, length = apoints, stop= amax), j in 1:length(y_grid)]
#     Ymat = [j for i=1:apoints, j in y_grid]
#
#     param.Amat = Amat
#     param.Ymat = Ymat
#
#     return param
# end

# param = rowenhorst!(param)





#define various functions that are useful for household block

#NOTE: param variable needs to be in global scope
#keywords really needs to be cleaned up

#marginal utilities and their inverses
up(c) = c.^(-param.gamma)
invup(x) = x.^(-1/param.gamma)

#vp(h) = param.phi.*h.^(1/param.frisch)
#invvp(x) = (x./param.phi).^(param.frisch)

#define optimal labour supply

# function h_opt(c_curr,Ymat,w)
#    u_c = up(c_curr)
#    return invvp(u_c.*Ymat.*w)
# end

#define euler equation iteration

function c_curr(;r_future,c_next)
    return invup(param.beta.*(1+r_future).*up(c_next)*param.P_trans')
end

#obtain current assets given consumption today defined on asset grid tomorrow

function a_curr(;r_curr,w,c_curr,Amat,Ymat)
    return 1/(1+r_curr).*(c_curr.+Amat.-w.*Ymat)
end

#find c at borrowing constraint

# function c_binding(r,w,Amat,Ymat,bc)

#     #initial guess

#     c_guess = Amat.+Ymat.*w

#     #variable to be returned

#     c_bind = similar(Amat)

#     for i=1: length(Amat[1,:])
#         for j = 1:length(Amat[:,1])

#          g(c) = (1+r)*Amat[j,i] .+ Ymat[j,i].*w .- c .+bc

#         c_bind[j,i] = find_zero(g, c_guess[j,i])

#         end
#     end

#     return c_bind
# end

# function get_capitaldis(;apoints::Int,num_states::Int,dist::Array{Float64})
#     #input is stationary distribution of dimension apoints*num_states
#     #first apoints elements correspond to mass of
#     #returns capital distribution (vector of size apoints)
#     capital_dist = zeros(num_states*apoints)
#     for i= 1:param.apoints
#     capital_dist[i]=sum(dist[i:apoints:end]);
#     end
#     return capital_dist
# end

#get total EFFECTIVE labor

# function get_L(;n_policy, apoints,num_states,Ymat,dist)
#     #inputs are apoints x num_states policy matrix (policy function for labour) and dist - a stationary distribution
#     # of dimension apoints*num_states.
#     #output is total effective labour.
#     L = reshape(n_policy, (apoints*num_states,1))
#     L = L.*dist.*reshape(Ymat,(apoints*num_states,1))
#     L = sum(L)
#     return L
# end

#get capital of future period i.e. get assets saved in CURRENT period given distribution TODAY
function get_K(;a_policy, dist)
    K= reshape(a_policy, (length(dist),1))
    K= sum(K.*dist)
    return K
end

function get_C(;c_policy, dist)
    C = reshape(c_policy, (length(dist),1))
    C = sum(C.*dist)
    return C
end


#!!!!!NOTE: function as is takes both FUTURE and CURRENT
# real interest rates as parameters, this is a bit of a pain
# best to reformulate only in terms of CURRENT interest rate

function egm_iterate(param::Param;w::Float64,c_next::Array{Float64},c_bind::Array{Float64},r_curr::Float64,r_future::Float64)

#function takes in parameters, CURRENT wage rate and FUTURE real interest rate. Also takes in c_next
# (future guess for consumption) and c_bind (the consumption policy when the borrowing constraint binds). This
# is to reduce number of computations that arise from non-linear solver.

#returns next periods policy functions

#defining some parameters



bc = param.bc
apoints = param.apoints
amax = param.amax
y_grid = param.y_grid
gamma = param.gamma
beta = param.beta
P_trans = param.P_trans
#frisch = param.frisch
Amat = param.Amat
Ymat = param.Ymat
#phi = param.phi

#initial guess c_0 = (r-deprec).*Amat .+ Ymat.*w

c_it = c_curr(r_future=r_future,c_next=c_next)

a_it = a_curr(r_curr=r_curr,w=w,c_curr=c_it,Amat=Amat,Ymat=Ymat)

#update elements for which borrowing constraint does not bind - interpolation

c_nonbinding = similar(Amat)

#get consumption policy function for current grid

for i=1:length(y_grid)

    c_nonbinding[:,i] = Spline1D(a_it[:,i],c_it[:,i],bc="extrapolate",k=1)(Amat[:,i])

end

#update elements for which borrowing constraint does bind

for j = 1:length(y_grid)
   c_it[:,j] = (Amat[:,j].>a_it[1,j]).*c_nonbinding[:,j] .+ (Amat[:,j].<=a_it[1,j]).*c_bind[:,j]
end

# return labour function

# h_it = h_opt(c_it,Ymat,w)

# return asset function

a_it = @. (1+r_curr)*Amat + w*Ymat - c_it

return c_it, a_it

end

# let

# r = -1.0
# w=1.0

# Amat = param.Amat
# Ymat = param.Ymat
# bc = param.bc

# c_bind = c_binding(r, w, Amat, Ymat, bc)
# c_next = @. r*Amat + w

# _, a = egm_iterate(param,c_bind = c_bind, c_next = c_next, r_future = r, r_curr =r,w=w)

# a

# end



#define transition function using Young's method
#input is a_it from egm_iterate

function get_trans(param::Param; policyfun::Array{Float64})

#some parameters

Amat = param.Amat
num_states = param.num_states
apoints = param.apoints
P_trans = param.P_trans

#construct transition matrix without labour stochasticity

Q = zeros(apoints, apoints, num_states)

#finds which element to interpolate to

findnearest(A::AbstractArray,t) = findmin(abs.(A.-t))[2]

next = similar(Amat)
previous = similar(Amat)

for j=1:num_states
for k = 1:apoints

ind = findnearest(Amat[:,j],policyfun[k,j])

if policyfun[k,j] - Amat[ind,j] >0
    previous[k,j] = Amat[ind,j]
    if ind==apoints
        next[k,j] = Amat[ind,j]
    else
     next[k,j] = Amat[ind+1,j]
    end

elseif policyfun[k,j] == Amat[ind,j]
    previous[k,j]=policyfun[k,j]
    next[k,j] = policyfun[k,]

else
    next[k,j] = Amat[ind,j]
    if ind==1
        previous[k,j] = Amat[ind,j]
    else
        previous[k,j] = Amat[ind-1,j]
    end

end
end
end

for k = 1:apoints
    for j = 1:num_states
        if next[k,j] == previous[k,j]
            Q[Amat[:,1].==previous[k,j],k,j] .= 1
        else
            Q[Amat[:,1].==previous[k,j],k,j] .= (next[k,j]-policyfun[k,j])./(next[k,j]-previous[k,j])
            Q[Amat[:,1].==next[k,j],k,j] .= 1 - (next[k,j]-policyfun[k,j])./(next[k,j]-previous[k,j])

        end
    end
end

#construct matrix with labour stochasticity

young_trans = zeros(apoints*num_states,apoints*num_states)

for j=1:num_states
for i=1:num_states
young_trans[(i-1)*apoints+1:i*apoints,(j-1)*apoints+1:j*apoints]=Q[:,:,j].*P_trans[j,i];
end
end

young_trans = young_trans'

return young_trans

end

# Tr = get_trans(param,policyfun=policyfun)






#get stationary distribution given r,w. Returns aggregate capital, effective labour
# and stationary consumption function.

function get_stat(param::Param;r, w, tol=1e-8, max_iter=1000)
    num_states = param.num_states
    apoints = param.apoints
    Amat = param.Amat
    Ymat = param.Ymat
    bc = param.bc

    #initial guess for policy function iteration

    c_next = @. r*param.Amat+param.Ymat*w

    #get binding consumption

    c_bind = @. (1+r)*Amat + w*Ymat + bc

    #initial counters

    dist1=1
    iter=1

    while dist1>tol && iter<max_iter

    c_it,_ = egm_iterate(param, w=w, r_curr=r, r_future = r, c_next=c_next, c_bind=c_bind)
    dist1 = norm(c_it - c_next, Inf)
    c_next = c_it
    iter = iter+1
    end

    #get policy functions after convergence

    c_it,a_it = egm_iterate(param, w=w, r_curr=r,r_future=r, c_next = c_next,c_bind=c_bind)

    #initial guess for transition iteration

    init = ones(apoints*num_states)
    init = init./sum(init)

    #iterate

    trans = get_trans(param,policyfun=a_it)

    #initial counters

    dist2 = 1
    iter=1

    while dist2>tol/100 && iter< max_iter

    next_init = (init'*trans)'
    dist2 = norm(next_init - init,Inf)
    init = next_init
    iter = iter+1
    end

    #construct total effective labour

    # L = get_L(n_policy = h_it, apoints=apoints, num_states=num_states,Ymat=Ymat,dist=init)

    #construct total assets

    K = get_K(a_policy = a_it, dist = init)

    return  K, c_it, init, a_it

end



# let

# alpha = param.alpha
# deprec = param.deprec


# r=0.01

# Z = ((r + deprec)/alpha)^alpha

# w = (1-alpha)*Z*(alpha*Z/(r+deprec))^(alpha/(1-alpha))

# _,c_it,_,a_it = get_stat(param,r=r,w=w)

# c_it[1,:]

# end

#get the same results as auclert et al. if we relax tolerance to
# 1e-6

#note: r=0.0466024 is a good initial guess for algorithm

function hh_ss!(param::Param;r_guess = 0.01,max_iter=100,tol=1e-6)
    alpha = param.alpha
    deprec = param.deprec
    dist = 1
    iter=1

    r_guess

    while dist>tol && iter<max_iter

    if r_guess>=1/(param.beta)-1
        error("r too large for convergence")
    end

    Z = ((r_guess + deprec)/alpha)^alpha #normalize so Y=1

    w = (1-alpha)*Z*(alpha*Z/(r_guess+deprec))^(alpha/(1-alpha))

    K,_ = get_stat(param,r=r_guess,w=w)

    r_supply = Z*alpha*(1/K)^(1-alpha) - deprec

    # w_supply = (1-alpha)*(alpha/(r_supply+deprec))^(alpha/(1-alpha))

    dist = abs(r_guess-r_supply)

    r_guess= 0.9*r_guess+0.1*r_supply

    iter = iter +1

    end

    #get final results

    Z = ((r_guess + deprec)/alpha)^alpha

    w = (1-alpha)*Z*(alpha*Z/(r_guess+deprec))^(alpha/(1-alpha))

    K,c_ss,init,policyfun = get_stat(param,r=r_guess,w=w)


    return r_guess,K,c_ss,init,policyfun

end

#get steady-state for jacobian later

# @time rss, Kss, css, dis, policyfun = hh_ss!(param)


#check walras law
#
# let
# deprec = param.deprec
# alpha = param.alpha
# Z = ((rss + deprec)/alpha)^alpha
#
# Z*Kss^alpha - get_C(c_policy = css, dist = dis) - deprec*Kss
#
# end



function expect_vec(param, policyfun, T; output)
    #inputs are policyfunction (to construct transition matrix) and
    # output vector with which transition matrix is MULTIPLIED. This
    # is the policyfun if one wants assets, css for consumption.

    expv = Array{Array{Float64,1},1}(undef,T-1)

    trans= get_trans(param, policyfun = policyfun)
    vec = reshape(output, length(trans[:,1]))

    expv[1] = vec

    for i=2:T-1
       expv[i] = (trans)*expv[i-1]
    end

    return expv

end

function fake_news(T;expv::Array{Array{Float64,1},1},dD::Array{Array{Float64,1},1}, dY::Array{Float64,1})
    F = zeros(T,T)

    for j=1:T
        for i=1:T
            if i == 1
                F[i,j] = dY[j]
            else
                F[i,j] = expv[i-1]'*dD[j]
            end
        end
    end

    return F
end

function get_jac(F::Array{Float64})
   J = copy(F)
    for j = 2:length(F[:,1])
        for i = 2:length(F[1,:])
            J[i,j] = J[i-1,j-1] + F[i,j]
        end
    end

    return J
end


#getting jacobian with respect to r

function get_dYr(param; T, diff, rss::Float64, css::Array{Float64}, dis::Vector{Float64})
#inputs: time frame for backwards iteration, finite difference step range, equilibrium interest rate
# equilbrium consumption (this is what is iterated), and apoints x num_states distribution (vector)
# returns dY_0,s of chosen outputs (here, consumption and capital).

alpha = param.alpha
deprec = param.deprec
Amat = param.Amat
Ymat = param.Ymat
bc = param.bc
apoints = param.apoints
num_states = param.num_states

r_up = rss + diff
r_down = rss - diff

Z = ((rss + deprec)/alpha)^alpha
w = (1-alpha)*Z*(alpha*Z/(rss+deprec))^(alpha/(1-alpha)) #keep w constant (partial Jacobian)

#vector that holds aggregate differences

dK = zeros(T)
dC = zeros(T)

# vector of one-dimensional vectors (must be column vectors!) to hold change in distributions

dD = Array{Array{Float64,1},1}(undef,T)

#first upward iteration
c_bind = @. (1+r_up)*Amat + w*Ymat + bc
cup,aup = egm_iterate(param, w = w, c_next = css, c_bind = c_bind, r_curr = r_up, r_future = rss)

Λ_up = get_trans(param, policyfun = aup) #get transition function (up)

#first downward iteration
c_bind = @. (1+r_down)*Amat + w*Ymat + bc
cdown,adown = egm_iterate(param, w = w, c_next = css, c_bind = c_bind, r_curr = r_down, r_future = rss)

Λ_down = get_trans(param, policyfun = adown) #get transition function (down)

#get capital and consumption
dK[1] = get_K(a_policy=(aup.-adown)./2, dist=dis)
dC[1] = get_C(c_policy = (cup.-cdown)./2, dist = dis)

#get change in distribution
dD[1] = (dis'*((Λ_up .- Λ_down)./2))'

#second upward iteration
c_bind = @. (1+rss)*Amat + w*Ymat + bc
cup,aup = egm_iterate(param, w = w, c_next = cup, c_bind = c_bind, r_curr = rss, r_future = r_up)

Λ_up = get_trans(param, policyfun = aup) #get transition function (up)

#second downward iteration
c_bind = @. (1+rss)*Amat + w*Ymat + bc
cdown,adown = egm_iterate(param, w = w, c_next = cdown, c_bind = c_bind, r_curr = rss, r_future = r_down)

Λ_down = get_trans(param, policyfun = adown) #get transition function (down)

#get capital and consumption
dK[2] = get_K(a_policy=(aup.-adown)./2, dist=dis)
dC[2] = get_C(c_policy = (cup.-cdown)./2, dist = dis)

#change in distribution
dD[2] = (dis'*((Λ_up .- Λ_down)./2))'

for i=1:T-2
    #upward iteration
    cup,aup = egm_iterate(param, w = w, c_next = cup, c_bind = c_bind, r_curr = rss, r_future = rss)
    Λ_up = get_trans(param, policyfun = aup)

    #downward iteration
    cdown,adown = egm_iterate(param, w = w, c_next = cdown, c_bind = c_bind, r_curr = rss, r_future = rss)
    Λ_down = get_trans(param, policyfun = adown) #get transition function (down)

    #get capital, consumption, and distribution
    dK[i+2] = get_K(a_policy=(aup.-adown)./2, dist=dis)
    dC[i+2] = get_C(c_policy = (cup.-cdown)./2, dist = dis)
    dD[i+2] = (dis'*((Λ_up .- Λ_down)./2))'
end

    return dK./diff, dC./diff, dD./diff

end


#getting jacobian with respect to w

function get_dYw(param; T, diff, rss::Float64, css::Array{Float64}, dis::Vector{Float64})
#inputs: time frame for backwards iteration, finite difference step range, equilibrium interest rate
# equilbrium consumption (this is what is iterated), and apoints x num_states distribution (vector)
# returns dY_0,s of chosen outputs (here, consumption and capital).

alpha = param.alpha
deprec = param.deprec
Amat = param.Amat
Ymat = param.Ymat
bc = param.bc
apoints = param.apoints
num_states = param.num_states

r = rss

Z = ((rss + deprec)/alpha)^alpha
w = (1-alpha)*Z*(alpha*Z/(rss+deprec))^(alpha/(1-alpha)) #keep w constant (partial Jacobian)

wup = w + diff
wdown = w - diff

#vector that holds aggregate differences

dK = zeros(T)
dC = zeros(T)

# vector of one-dimensional vectors (must be column vectors!) to hold change in distributions

dD = Array{Array{Float64,1},1}(undef,T)

#first upward iteration
c_bind = @. (1+r)*Amat + wup*Ymat + bc
cup,aup = egm_iterate(param, w = wup, c_next = css, c_bind = c_bind, r_curr = r, r_future = r)

Λ_up = get_trans(param, policyfun = aup) #get transition function (up)

#first downward iteration
c_bind = @. (1+r)*Amat + wdown*Ymat + bc
cdown,adown = egm_iterate(param, w = wdown, c_next = css, c_bind = c_bind, r_curr = r, r_future = r)

Λ_down = get_trans(param, policyfun = adown) #get transition function (down)

#get capital and consumption
dK[1] = get_K(a_policy=(aup.-adown)./2, dist=dis)
dC[1] = get_C(c_policy = (cup.-cdown)./2, dist = dis)

#get change in distribution
dD[1] = (dis'*((Λ_up .- Λ_down)./2))'

#second upward iteration
c_bind = @. (1+r)*Amat + w*Ymat + bc
cup,aup = egm_iterate(param, w = w, c_next = cup, c_bind = c_bind, r_curr = r, r_future = r)

Λ_up = get_trans(param, policyfun = aup) #get transition function (up)

#second downward iteration
c_bind = @. (1+r)*Amat + w*Ymat + bc
cdown,adown = egm_iterate(param, w = w, c_next = cdown, c_bind = c_bind, r_curr = r, r_future = r)

Λ_down = get_trans(param, policyfun = adown) #get transition function (down)

#get capital and consumption
dK[2] = get_K(a_policy=(aup.-adown)./2, dist=dis)
dC[2] = get_C(c_policy = (cup.-cdown)./2, dist = dis)

#change in distribution
dD[2] = (dis'*((Λ_up .- Λ_down)./2))'

for i=1:T-2
    #upward iteration
    cup,aup = egm_iterate(param, w = w, c_next = cup, c_bind = c_bind, r_curr = r, r_future = r)
    Λ_up = get_trans(param, policyfun = aup)

    #downward iteration
    cdown,adown = egm_iterate(param, w = w, c_next = cdown, c_bind = c_bind, r_curr = rss, r_future = rss)
    Λ_down = get_trans(param, policyfun = adown) #get transition function (down)

    #get capital, consumption, and distribution
    dK[i+2] = get_K(a_policy=(aup.-adown)./2, dist=dis)
    dC[i+2] = get_C(c_policy = (cup.-cdown)./2, dist = dis)
    dD[i+2] = (dis'*((Λ_up .- Λ_down)./2))'
end

    return dK./diff, dC./diff, dD./diff

end

function het_jac(param; rss::Float64, css::Array{Float64}, policyfun::Array{Float64}, dis::Vector{Float64}, T=5, diff = 1e-5)

    #get dY_0^s and dD_1^s
    dKr, dCr, dDr = get_dYr(param; rss=rss, css=css, dis=dis, T=T, diff=diff)
    dKw, dCw, dDw = get_dYw(param; rss=rss, css=css, dis=dis, T=T, diff=diff)

    #get vector of expectation vectors (size T-1)
    expv_a = expect_vec(param, policyfun, T; output = policyfun)
    expv_c = expect_vec(param, policyfun, T; output = css)

    #construct fake news matrix
    F_ar = fake_news(T; expv = expv_a, dY = dKr, dD = dDr)
    F_cr = fake_news(T; expv = expv_c, dY = dCr, dD = dDr)

    F_aw = fake_news(T; expv = expv_a, dY = dKw, dD = dDw)
    F_cw = fake_news(T; expv = expv_c, dY = dCw, dD = dDw)
    #get jacobian
    J_ar = get_jac(F_ar)
    J_cr = get_jac(F_cr)

    J_aw = get_jac(F_aw)
    J_cw = get_jac(F_cw)

    #construct dictionaries
    J = Dict()

    J_a = Dict("r" => J_ar, "w" => J_aw)
    J_c = Dict("r" => J_cr, "w" => J_cw)

    J["A"] = J_a
    J["C"] = J_c

    return J
end

end
