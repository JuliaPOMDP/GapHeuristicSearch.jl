using POMDPs
using Random
using Parameters
using DataStructures
using LinearAlgebra

# TODO: Specify requirements with POMDPLinter
@with_kw mutable struct BNode{B,A,O}
    b::B
    as::Union{Nothing,Vector{A}} = nothing
    a_to_orb::Dict{A,Tuple{Vector{O},Vector{Float64},Vector{Int64},Vector{BNode{B,A,O}}}} = Dict{A,Tuple{Vector{O},Vector{Float64},Vector{Int64},Vector{BNode{B,A,O}}}}()
    initialized::Bool = false
    is_root::Bool = false
    Uhi::Float64 = Inf
    Ulo::Float64 = -Inf
end

struct GapHeuristicSearchSolver <: Solver
    π::Union{Policy,Nothing}    # rollout policy, TBD if more flexibility for lower bound
    up::Updater                 # updater
    uhi_func                    # upper bound on belief value function
    ulo_func                    # lower bound on belief value function
    Rmax::Float64               # max reward
    delta::Float64              # gap threshold
    k_max::Int64                # maximum # simulations
    d_max::Int64                # maximum depth
    nsamps::Int64               # number of montecarlo observation samples
    max_steps::Int64            # number of rollout steps 
    keep_bounds::Bool           # do not re-initialize the upper/lower bounds on repeated calls to the same planner
    verbose::Bool               # verbose operation mode
end

function GapHeuristicSearchSolver(π::Union{Policy,Nothing},up::Updater, Rmax::Float64; uhi_func=nothing,ulo_func=nothing,delta::Float64=1e-2,k_max::Int64=200, d_max::Int64=10,nsamps::Int64=20,max_steps=100,keep_bounds=false,verbose=false)
    return GapHeuristicSearchSolver(π,up,uhi_func,ulo_func,Rmax,delta,k_max,d_max,nsamps,max_steps,keep_bounds,verbose)
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
    root = BNode{B,A,O}(b = initialize_belief(solver.up, initialstate(pomdp)),is_root = true)
    return GapHeuristicSearchPlanner{B,A,O}(pomdp=pomdp,solver=solver,root=root)
end

POMDPs.updater(planner::GapHeuristicSearchPlanner) = planner.solver.up

function printB(B;long=false)
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
        bn.Uhi = planner.solver.uhi_func === nothing ? Rmax/(1.0-discount(pomdp)) : planner.solver.uhi_func(pomdp,bn.b)
        bn.Ulo = planner.solver.ulo_func === nothing ? rollout(pomdp,π,up,bn.b,max_steps) : planner.solver.ulo_func(pomdp,bn.b)
        # print(bn.b,"::")
        # println(planner.solver.ulo_func(pomdp,bn.b))

    end
end

function initialize_search_space!(planner::GapHeuristicSearchPlanner, bnode::BNode)
    pomdp, π, up, nsamps, max_steps = planner.pomdp, planner.solver.π, planner.solver.up, planner.solver.nsamps, planner.solver.max_steps
    
    B,A,O = get_type(planner)
    a_to_orb = bnode.a_to_orb
    b = bnode.b
    bnode.as = actions(pomdp,b)
    for a in bnode.as
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

        bps = [POMDPs.update(up,b,a,o) for o in os]
        # next_bnodes = [BNode{B,A,O}(b=bp) for (o,r,bp) in zip(os,rs,bps)] #old, not weighted
        next_bnodes = [BNode{B,A,O}(b=bp) for bp in bps]

        #initializing bounds
        [initialize_bounds!(planner,bn) for bn in next_bnodes]
        
        # a_to_orb[a] = (os,rs,next_bnodes) #old, no weighting
        a_to_orb[a] = (os,rs,ws,next_bnodes)

    end
    initialize_bounds!(planner,bnode)# also check for current belief
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
    Uhi, Ulo, delta, pomdp, π, up, nsamps, max_steps, Rmax = planner.Uhi, planner.Ulo, planner.solver.delta, planner.pomdp, planner.solver.π, planner.solver.up, planner.solver.nsamps, planner.solver.max_steps, planner.solver.Rmax
    b = bnode.b

    # check if we have already explored from this belief: if so, reuse sampled aors. otherwise, generate new
    a_to_orb = bnode.initialized ? bnode.a_to_orb : initialize_search_space!(planner,bnode) 

    A = bnode.as
    A == nothing && println("well fuck off")
  
    planner.solver.verbose && println("Current belief is: ",b)

    if d == 0 || bnode.Uhi - bnode.Ulo ≤ delta
        planner.solver.verbose && (d==0 ? println("Pruned: d==0") : println("Pruned: gap ≤ delta"))
        return
    end

    a = argmax(a -> lookahead(pomdp,bn -> bn.Uhi,bnode,a), A)
    obs,rs,ws,bns = bnode.a_to_orb[a]
    
    o,bnp = best_ob(planner,bnode,a)
    # o = argmax(o -> Uhi[B[BAO(b,a,o)]]-Ulo[B[BAO(b,a,o)]], as_to_ors[a].obs)
    
    planner.solver.verbose && print("(level ",d," to go) exploring from belief ",b," \nwith action: ",a,", observation: ",o)
    planner.solver.verbose && println("\nresulting in belief: ",bnp.b)

    heuristic_search!(planner,bnp,d-1)

    upper = maximum(lookahead(pomdp,bn -> bn.Uhi,bnode,a) for a in A)
    lower = maximum(lookahead(pomdp,bn -> bn.Ulo,bnode,a) for a in A)
    
    bnode.Uhi = upper
    bnode.Ulo = lower

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
    Uhi, Ulo, k_max, d_max, delta, pomdp, up = planner.Uhi, planner.Ulo, planner.solver.k_max, planner.solver.d_max, planner.solver.delta, planner.pomdp, planner.solver.up
    
    B,A,O = BAO_type(pomdp,planner.solver.up)
    root = BNode{B,A,O}(b = b,is_root = true)
    planner.root = root
    
    # Generate first level of explorations here first
    initialize_search_space!(planner,planner.root)

    for i in 1:k_max
        planner.solver.verbose && println("beginning search ",i)
        heuristic_search!(planner,root,d_max)
        # if Uhi[b] - Ulo[b] < delta
        if root.Uhi - root.Ulo < delta
            planner.solver.verbose && println("Breaking due to gap ",root.Uhi - root.Ulo,"<=",delta)
            break
        end
    end
    
    a =  argmax(a -> lookahead(pomdp,bn -> bn.Ulo,root,a),root.as) # return the best action accoridng to 1 step lookahead with the lower bound on belief state value
    
    return a
end
