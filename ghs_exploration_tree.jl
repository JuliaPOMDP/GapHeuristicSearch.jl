using POMDPs
using POMDPPolicies
using POMDPModelTools
using POMDPSimulators
using POMDPModels
using BeliefUpdaters
using FIB
using SARSOP
using StaticArrays
using Random
using Statistics
using DataFrames
using ParticleFilters
using RockSample 
using POMDPGifs
import Cairo, Fontconfig
using Parameters
using StatsBase
using DataStructures
using LinearAlgebra

# Developing GapHeuristicSearch

# TODO: Specify requirements with POMDPLinter

struct BAO{B,A,O}
    b::B
    a::Union{A,Nothing}
    o::Union{O,Nothing}
end

@with_kw mutable struct BNode{B,A,O}
    b::B
    a_to_orb::Dict{A,Tuple{Vector{O},Vector{Float64},Vector{Int64},Vector{BNode{B,A,O}}}} = Dict{A,Tuple{Vector{O},Vector{Float64},Vector{Int64},Vector{BNode{B,A,O}}}}()
    initialized::Bool = false
    is_root::Bool = false
    Uhi::Float64 = Inf
    Ulo::Float64 = -Inf
end

struct GapHeuristicSearchSolver <: Solver
    π::Policy           # rollout policy, TBD if more flexibility for lower bound
    up::Updater         # updater
    uhi_func            # upper bound on belief value function
    ulo_func            # lower bound on belief value function
    Rmax::Float64       # max reward
    δ::Float64          # gap threshold
    k_max::Int64        # maximum # simulations
    d_max::Int64        # maximum depth
    nsamps::Int64       # number of montecarlo observation samples
    max_steps::Int64    # number of rollout steps 
    keep_bounds::Bool   # do not re-initialize the upper/lower bounds on repeated calls to the same planner
    verbose::Bool       # verbose operation mode
end

function GapHeuristicSearchSolver(π::Policy,up::Updater, Rmax::Float64; uhi_func=nothing,ulo_func=nothing,δ::Float64=1e-2,k_max::Int64=200, d_max::Int64=10,nsamps::Int64=20,max_steps=100,keep_bounds=false,verbose=false)
    return GapHeuristicSearchSolver(π,up,uhi_func,ulo_func,Rmax,δ,k_max,d_max,nsamps,max_steps,keep_bounds,verbose)
end

@with_kw mutable struct GapHeuristicSearchPlanner{B,A,O} <: Policy
    pomdp::POMDP                            # underlying pomdp
    solver::GapHeuristicSearchSolver    # contains solver parameters 
    root::BNode{B,A,O}
    Uhi::Dict{B,Float64} = Dict{B,Float64}()
    Ulo::Dict{B,Float64} = Dict{B,Float64}()
end

get_type(planner::GapHeuristicSearchPlanner{B,A,O}) where {B,A,O} = (B,A,O)

function BAO_type(pomdp::POMDP,up::Updater)
    b0 = initialize_belief(up, initialstate(pomdp))
    B = typeof(b0)
    A = actiontype(pomdp)
    O = obstype(pomdp)
    return B,A,O
end

function POMDPs.solve(solver::GapHeuristicSearchSolver, pomdp::POMDP) 
    B,A,O = BAO_type(pomdp,solver.up)
    root = BNode{B,A,O}(b = initialize_belief(up, initialstate(pomdp)),is_root = true)
    return GapHeuristicSearchPlanner{B,A,O}(pomdp=pomdp,solver=solver,root=root)
end

POMDPs.updater(planner::GapHeuristicSearchPlanner) = planner.solver.up

function printB(B;long=false)
    # println("inside printB")
    if !long
        bs = unique([BAO.b for BAO in keys(B)])
        for b in bs
            print(b,":: "),
            for (k,v) in filter(x->x[1].b==b,B)
                print("(",k.a,", ",k.o,") ") #,", bp ",v) 
            end
            print(" | ")
        end
        println()
    else
        for key in keys(B)
            println(key)
        end
    end
end

function montecarlo_ors(pomdp::POMDP,b,a,nsamps::Int64)
    outputs = [@gen(:o,:r)(pomdp, rand(b), a) for _ in 1:nsamps]
    obs = [out[1] for out in outputs]
    rs = [out[2] for out in outputs]

    out = (obs=obs,rs=rs)
    return out
end

function initialize_bounds!(planner,bn)
    pomdp, π, up, max_steps,Rmax = planner.pomdp, planner.solver.π, planner.solver.up, planner.solver.max_steps, planner.solver.Rmax
    if isinf(bn.Uhi)
        bn.Uhi = planner.solver.uhi_func === nothing ? Rmax/(1.0-discount(pomdp)) : planner.solver.uhi_func(bn.b)
        bn.Ulo = planner.solver.ulo_func === nothing ? rollout(pomdp,π,up,bn.b,max_steps) : planner.solver.ulo_func(bn.b)
    end
end

function initialize_search_space!(planner::GapHeuristicSearchPlanner, bnode::BNode)
    pomdp, π, up, nsamps, max_steps = planner.pomdp, planner.solver.π, planner.solver.up, planner.solver.nsamps, planner.solver.max_steps
    
    B,A,O = get_type(planner)
    a_to_orb = bnode.a_to_orb
    b = bnode.b
    for a in  actions(pomdp)
        out = montecarlo_ors(pomdp,b,a,nsamps)
        os = out.obs
        rs = out.rs
    
        #todo: add weighting for repeated observation types so we dont waste search
        w = counter(O)
        rcum  = Accumulator{O, Float64}()
        for (o,r) in zip(os,rs)
            inc!(w,o)
            inc!(rcum,o,r)
        end
        
        os = collect(keys(w))
        rs = [rcum[o] for o in os]
        ws = [w[o] for o in os]

        bps = [update(up,b,a,o) for o in os]
        # next_bnodes = [BNode{B,A,O}(b=bp) for (o,r,bp) in zip(os,rs,bps)] #old, not weighted
        next_bnodes = [BNode{B,A,O}(b=bp) for bp in bps]

        #initializing bounds
        [initialize_bounds!(planner,bn) for bn in next_bnodes]
        
        # a_to_orb[a] = (os,rs,next_bnodes) #old, no weighting
        a_to_orb[a] = (os,rs,ws,next_bnodes)

    end
    initialize_bounds!(planner,bnode) # also check for current belief
    bnode.initialized = true

    return a_to_orb
end

function best_ob(planner::GapHeuristicSearchPlanner,bnode::BNode,a)
    obs,rs,ws,bns = bnode.a_to_orb[a]
    # scores = [planner.Uhi[bnp.b]-planner.Ulo[bnp.b] for bnp in bns]
    scores = [bnp.Uhi-bnp.Ulo for bnp in bns]
    ind = argmax(scores)
    return obs[ind],bns[ind]
end

function heuristic_search!(planner::GapHeuristicSearchPlanner,bnode::BNode,d::Int64)
    Uhi, Ulo, δ, pomdp, π, up, nsamps, max_steps, Rmax = planner.Uhi, planner.Ulo, planner.solver.δ, planner.pomdp, planner.solver.π, planner.solver.up, planner.solver.nsamps, planner.solver.max_steps, planner.solver.Rmax
    A = actions(pomdp)

    # check if we have already explored from this belief: if so, reuse sampled aors. otherwise, generate new
    a_to_orb = bnode.initialized ? bnode.a_to_orb : initialize_search_space!(planner,bnode) 

    b = bnode.b
    planner.solver.verbose && println("Current belief is: ",b)

    if d == 0 || bnode.Uhi - bnode.Ulo ≤ δ
    # if d == 0 || Uhi[b] - Ulo[b] ≤ δ
        planner.solver.verbose && (d==0 ? println("Pruned: d==0") : println("Pruned: gap ≤ δ"))
        return
    end

    # a = argmax(a -> lookahead(pomdp,bp -> Uhi[bp],bnode,a), A)
    a = argmax(a -> lookahead(pomdp,bn -> bn.Uhi,bnode,a), A)
    obs,rs,ws,bns = bnode.a_to_orb[a]
    
    # bnp = argmax(bnp->planner.Uhi[bnp.b]-planner.Ulo[bnp.b],bns)
    o,bnp = best_ob(planner,bnode,a)
    # o = argmax(o -> Uhi[B[BAO(b,a,o)]]-Ulo[B[BAO(b,a,o)]], as_to_ors[a].obs)
    
    planner.solver.verbose && print("(level ",d," to go) exploring from belief ",b," \nwith action: ",a,", observation: ",o)
    planner.solver.verbose && println("\nresulting in belief: ",bnp.b)

    heuristic_search!(planner,bnp,d-1)

    # upper = maximum(lookahead(pomdp,bp -> Uhi[bp],B,as_to_ors[a],up,b,a) for a in A)
    upper = maximum(lookahead(pomdp,bn -> bn.Uhi,bnode,a) for a in A)
    # lower = maximum(lookahead(pomdp,bp -> Ulo[bp],B,as_to_ors[a],up,b,a) for a in A)
    lower = maximum(lookahead(pomdp,bn -> bn.Ulo,bnode,a) for a in A)
    
    bnode.Uhi = upper
    bnode.Ulo = lower
    # Uhi[b] = upper
    # Ulo[b] = lower
    planner.solver.verbose && println("After recursion, updating belief",b," to Upper: ",upper,", lower: ",lower,". completed ",d," levels")

end

function rollout(pomdp::POMDP,π::Policy,up::Updater,b,max_steps)
    sim = RolloutSimulator(max_steps=max_steps)
    r = simulate(sim, pomdp, π,up,b)
end

function lookahead(pomdp::POMDP,U,bnode::BNode,a)
    # currently approximating via montecarlo, using the obervsation set defined in heuristic_search
    obs,rs,ws,bns = bnode.a_to_orb[a]
    ws = ws/sum(ws)
    # r = mean(rs) before weights
    r = dot(ws,rs)
    # return r + discount(pomdp)*mean(U(bn) for bn in bns) #before weights
    return r + discount(pomdp)*dot([U(bn) for bn in bns],ws)
end

function POMDPs.action(planner::GapHeuristicSearchPlanner, b)
    if !planner.solver.keep_bounds # Re initialize dictionaries. An error currently occurs if we dont reinitialize the mct. Do we need trash collection?
        Bt,At,Ot = BAO_type(planner.pomdp,planner.solver.up)
        planner.Uhi = Dict{Bt,Float64}()
        planner.Ulo = Dict{Bt,Float64}()
    end
    Uhi, Ulo, k_max, d_max, δ, pomdp, up = planner.Uhi, planner.Ulo, planner.solver.k_max, planner.solver.d_max, planner.solver.δ, planner.pomdp, planner.solver.up
    
    B,A,O = BAO_type(pomdp,planner.solver.up)
    root = BNode{B,A,O}(b = b,is_root = true)
    planner.root = root
    
    # Generate first level of explorations here first
    initialize_search_space!(planner,planner.root)

    for i in 1:k_max
        planner.solver.verbose && println("beginning search ",i)
        heuristic_search!(planner,root,d_max)
        # if Uhi[b] - Ulo[b] < δ
        if root.Uhi - root.Ulo < δ
            planner.solver.verbose && println("Breaking due to gap ",Uhi[b] - Ulo[b],"≤",δ)
            break
        end
    end
    
    a =  argmax(a -> lookahead(pomdp,bn -> bn.Ulo,root,a),actions(pomdp)) # return the best action accoridng to 1 step lookahead with the lower bound on belief state value
    # println("Bnode is: ")
    # println(root)
    # probably need a way to print the tree

    return a
end

# stuff to varify on paratrial for crying baby

# cryingbaby = BabyPOMDP()
# roller = RandomPolicy(cryingbaby)
# up = DiscreteUpdater(cryingbaby)
# Rmax = 0.0  # best possible reward is baby not hungry, didnt feed
# solver = GapHeuristicSearchSolver(roller,up,Rmax,δ=.1,k_max=250,d_max=12,nsamps=10,max_steps=20,verbose=false)
# ghs_policy = solve(solver, cryingbaby)
# Base.show(io::IO,x::DiscreteBelief) = print(x.b)

# b_hungry = DiscreteBelief(cryingbaby,[.1,.9])
# b_nothungry = DiscreteBelief(cryingbaby,[.9,0.1])

# for i in 1:10
#     println(action(ghs_policy, b_hungry))
# end

# sarsop_solver = SARSOPSolver()
# sarsop_policy = solve(sarsop_solver, cryingbaby)
# runs = 100
# q_para = [] # vector of the simulations to be run
# [push!(q_para, Sim(cryingbaby, sarsop_policy, max_steps=50, rng=MersenneTwister(), metadata=Dict(:policy=>"sarsop"))) for i in 1:runs]
# [push!(q_para, Sim(cryingbaby, ghs_policy,ghs_policy.solver.up, max_steps=50, rng=MersenneTwister(), metadata=Dict(:policy=>"ghs"))) for i in 1:runs]

################################################################################################################

# LightDark experimenting

# struct LDNormalStateDist
#     mean::Float64
#     std::Float64
# end
# Base.rand(rng::AbstractRNG, d::LDNormalStateDist) = LightDark1DState(0, d.mean + randn(rng)*d.std)

# Base.show(io::IO,x::ParticleCollection{LightDark1DState}) = print([(s.status,s.y) for s in particles(x)])

# pomdp = LightDark1D()


# roller = RandomPolicy(pomdp)
# up = SIRParticleFilter(pomdp, 5)
# # up = PreviousObservationUpdater()

# Rmax = pomdp.correct_r  # 

# ghs_solver = GapHeuristicSearchSolver(roller,
#                                     up,
#                                     Rmax, # best possible reward is exit or good rock
#                                     δ=1e-2,
#                                     k_max=1, #100
#                                     d_max=2,   #7
#                                     nsamps=1,  #5
#                                     max_steps=15,
#                                     verbose=true)



# ghs_policy = solve(ghs_solver, pomdp)

# b0 = initialize_belief(up, initialstate(pomdp))
# rng = MersenneTwister(11) #11 originally
# s0 = rand(rng,b0)
# println("s: ",s0)
# println("b: ",b0)
# bp = update(up,b0,0,-1)
# println("bp: ",bp)
# isterminal(bp)

# action(ghs_policy,b0)

# rollout_sim = RolloutSimulator(max_steps=1);
# r_pomcp = simulate(rollout_sim, pomdp, ghs_policy, up);

################################################################################################################

# Rock world testing

Base.show(io::IO,b::DiscreteBelief{RockSamplePOMDP{1}, RSState{1}}) = print(b.b[b.b.>0])
pomdp = RockSamplePOMDP(rocks_positions=[(2,3), (4,4), (4,2)], 
                        sensor_efficiency=10.0, 
                        discount_factor=0.95, 
                        good_rock_reward = 10.0);

# pomdp = RockSamplePOMDP(map_size = (1,3),
#                         rocks_positions=[(1,1)], 
#                         sensor_efficiency=10.0, 
#                         discount_factor=0.95, 
#                         good_rock_reward = 10.0);

solver = SARSOPSolver(precision=1e-3)
# policy = solve(solver, pomdp)
policy = load_policy(pomdp, "policy.out");

up = DiscreteUpdater(pomdp)
b0 = initialize_belief(up, initialstate(pomdp))
rng = MersenneTwister(11) #11 originally
s0 = rand(rng,b0)
# println(s0)
# sim = GifSimulator(filename="out_sarsop.gif", max_steps=30)
# simulate(sim, pomdp, policy,up,b0,s0);
rs_exit = solve(RSExitSolver(),pomdp)
POMDPs.updater(rs_exit::RSExit) = NothingUpdater()

function lower(b)
    s = first(b.state_list)
    return s.pos[1] == -1 ? 0.0 : rs_exit.exit_return[s.pos[1]]
end
roller = RandomPolicy(pomdp)

# # known settings k_max=40,d_max=4,nsamps=10,max_steps=15,
# # with tree + weights: 0.016795
# # with tree: 0.090699
# # after changing to IdDicts: 0.242559 second


ghs_solver = GapHeuristicSearchSolver(roller,
                                    DiscreteUpdater(pomdp),
                                    1.8549375000000015, # best possible reward is exit or good rock
                                    ulo_func=lower,
                                    δ=1e-2,
                                    k_max=1000, #100
                                    d_max=7,   #7
                                    nsamps=10,  #5
                                    max_steps=15,
                                    verbose=false)

ghs_policy = solve(ghs_solver, pomdp)
# println(b0)
action(ghs_policy,b0)
@time action(ghs_policy,b0)

# ghs_sim = GifSimulator(filename="out_ghs.gif", max_steps=30)
# simulate(ghs_sim, pomdp, ghs_policy,up,b0,s0)

# #POMCPOW
# using POMCPOW
# solver_pomcpow = POMCPOWSolver(criterion=MaxUCB(20.0),
#                                 max_depth=7,
#                                 tree_queries = 1000,
#                                 estimate_value=POMCPOW.RolloutEstimator(rs_exit),
#                                 enable_action_pw = false,
#                                 check_repeat_obs = true,
#                                 check_repeat_act = true)
# pomcpow_policy = solve(solver_pomcpow, pomdp)

# using BasicPOMCP
# pomcp_solver = POMCPSolver(max_depth=7,
#                             tree_queries=1000,
#                             estimate_value=POMCPOW.RolloutEstimator(rs_exit)
# )
# pomcp_policy = solve(pomcp_solver, pomdp)

# # action(pomcp_policy,b0)
# # @time action(pomcp_policy,b0)

# runs = 200
# q_para = [] # vector of the simulations to be run
# [push!(q_para, Sim(pomdp, policy, updater(policy),b0,s0, max_steps=30, rng=MersenneTwister(), metadata=Dict(:policy=>"sarsop"))) for i in 1:runs]
# [push!(q_para, Sim(pomdp, ghs_policy, ghs_policy.solver.up,b0,s0, max_steps=30, rng=MersenneTwister(), metadata=Dict(:policy=>"ghs"))) for i in 1:runs]
# [push!(q_para, Sim(pomdp, pomcpow_policy, updater(pomcpow_policy),b0,s0, max_steps=30, rng=MersenneTwister(), metadata=Dict(:policy=>"pomcpow"))) for i in 1:runs]
# [push!(q_para, Sim(pomdp, pomcp_policy, updater(pomcp_policy),b0,s0, max_steps=30, rng=MersenneTwister(), metadata=Dict(:policy=>"pomcp"))) for i in 1:runs]


# hr = HistoryRecorder(max_steps=30)
# history = simulate(hr, pomdp, ghs_policy, updater(ghs_policy), b0, s0);
# println("\n\n\nBeginning\n\n")
# for (s,b,a,o,r) in eachstep(history, "s,b,a,o,r")
#     print("s: ",s)
#     print(", b: ",b)
#     print(", a: ",a)
#     print(", r: ",r)
#     println(", o: ",o)
#     println()
# end

# action(ghs_policy,b0)
# @time action(ghs_policy,b0)

# using StatProfilerHTML
# @profilehtml action(ghs_policy,b0)