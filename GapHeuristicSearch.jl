"""
Belief node type for the belief search tree. 
"""
@with_kw mutable struct BNode{B,A,O}
    b::B
    as::Any = nothing
    a_to_orb::Dict{A,Tuple{Vector{O},Vector{Float64},Vector{Int64},Vector{BNode{B,A,O}}}} = Dict{A,Tuple{Vector{O},Vector{Float64},Vector{Int64},Vector{BNode{B,A,O}}}}()
    initialized::Bool = false
    is_root::Bool = false
    Uhi::Float64 = Inf
    Ulo::Float64 = -Inf
end

"""
GHS solver type
Fields:
    π::Union{Policy,Nothing}
        Rollout policy, if nothing must implement ulo_func.
    up::Updater
        Updater of type POMDPs.Updater
    uhi_func                    
        Upper bound on belief value function. Function takes in the POMDP and the current belief.
        default: nothing
    ulo_func
        Lower bound on belief value function. Function takes in the POMDP and the current belief.
        default: nothing
    Rmax::Float64               
        Max reward, for the best action best state upper bound.
    delta::Float64
        Gap threshold. Exploration stops once the gap in bounds at a belief is below the threshold.
        default: 1e-2
    k_max::Int64
        Maximum number of searches.
        default: 200
    d_max::Int64                
        Maximum depth
        default: 10
    nsamps::Int64               
        Number of montecarlo observation samples. If the observations are discrete, weights are calculated to group identical observations. 
        default: 20
    max_steps::Int64
        Number of rollout steps.
        default: 100
    verbose::Bool
        Verbose operation mode.
        default: false
"""
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
    verbose::Bool               # verbose operation mode
end

"""
    GapHeuristicSearchSolver()

Use keyword arguments to specify values for the fields.
"""
function GapHeuristicSearchSolver(π::Union{Policy,Nothing},
                                up::Updater, 
                                Rmax::Float64; 
                                uhi_func=nothing,
                                ulo_func=nothing,
                                delta::Float64=1e-2,
                                k_max::Int64=200,
                                d_max::Int64=10,
                                nsamps::Int64=20,
                                max_steps=100,
                                verbose=false)
    return GapHeuristicSearchSolver(π,up,uhi_func,ulo_func,Rmax,delta,k_max,d_max,nsamps,max_steps,verbose)
end

"""
GHS planner type. Stores the POMDP, solver, and the root node in the search tree. 
"""
@with_kw mutable struct GapHeuristicSearchPlanner{B,A,O} <: Policy
    pomdp::POMDP                            # underlying pomdp
    solver::GapHeuristicSearchSolver    # contains solver parameters 
    root::BNode{B,A,O}
end

get_type(planner::GapHeuristicSearchPlanner{B,A,O}) where {B,A,O} = (B,A,O)

function BAO_type(pomdp::POMDP,up::Updater)
    b0 = initialize_belief(up, initialstate(pomdp))
    B = typeof(b0)
    A = actiontype(pomdp)
    O = obstype(pomdp)
    return B,A,O
end

"""
Return the constructed planner object given a GHS solver and POMDP.
"""
function POMDPs.solve(solver::GapHeuristicSearchSolver, pomdp::POMDP) 
    B,A,O = BAO_type(pomdp,solver.up)
    root = BNode{B,A,O}(b = initialize_belief(solver.up, initialstate(pomdp)),is_root = true)
    return GapHeuristicSearchPlanner{B,A,O}(pomdp=pomdp,solver=solver,root=root)
end

@POMDP_require solve(solver::GapHeuristicSearchSolver,pomdp::POMDP) begin
    M = typeof(pomdp)
    S = statetype(pomdp)
    A = actiontype(pomdp)
    U = typeof(solver.up)
    @req initialize_belief(::U,::S)
    @req initialstate(::M)
end

POMDPs.updater(planner::GapHeuristicSearchPlanner) = planner.solver.up

"""
Generate montecaro obervation, rewards pairs from an initial state b. No weighting/combining done in this method.
"""
function montecarlo_ors(pomdp::POMDP,b,a,nsamps::Int64)
    outputs = [@gen(:o,:r)(pomdp, rand(b), a) for _ in 1:nsamps]
    obs = [out[1] for out in outputs]
    rs = [out[2] for out in outputs]

    out = (obs=obs,rs=rs)
    return out
end

"""
Initialize the upper and lower bounds at a belief node. 
"""
function initialize_bounds!(planner,bn)
    pomdp, π, up, max_steps,Rmax = planner.pomdp, planner.solver.π, planner.solver.up, planner.solver.max_steps, planner.solver.Rmax
    if isinf(bn.Uhi)
        bn.Uhi = planner.solver.uhi_func === nothing ? Rmax/(1.0-discount(pomdp)) : planner.solver.uhi_func(pomdp,bn.b)
        bn.Ulo = planner.solver.ulo_func === nothing ? rollout(pomdp,π,up,bn.b,max_steps) : planner.solver.ulo_func(pomdp,bn.b)
    end
end

"""
Initializes search space from a belief node: generates a,o,bp tuples, linked belief nodes, and initializes bounds.
"""
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
    
        w = counter(O) # Weighting for repeated observation types so we dont waste search
        rcum  = Accumulator{O, Float64}()
        for (o,r) in zip(os,rs)
            inc!(w,o)
            inc!(rcum,o,r)
        end
        
        os = collect(keys(w))
        rs = [rcum[o] for o in os]
        ws = [w[o] for o in os]

        bps = [POMDPs.update(up,b,a,o) for o in os]
        next_bnodes = [BNode{B,A,O}(b=bp) for bp in bps]
        
        [initialize_bounds!(planner,bn) for bn in next_bnodes]
        a_to_orb[a] = (os,rs,ws,next_bnodes)
    end
    
    initialize_bounds!(planner,bnode)# also check for current belief
    bnode.initialized = true
    return a_to_orb
end

"""
Return the gap-maximizing observation and corresponding next belief node given a current belief node and an action.
"""
function best_ob(planner::GapHeuristicSearchPlanner,bnode::BNode,a)
    obs,rs,ws,bns = bnode.a_to_orb[a]
    scores = [bnp.Uhi-bnp.Ulo for bnp in bns]
    ind = argmax(scores)
    return obs[ind],bns[ind]
end

"""
The heuristic search itself, called recursively. The main work of `action` is done here. 
"""
function heuristic_search!(planner::GapHeuristicSearchPlanner,bnode::BNode,d::Int64)
    delta, pomdp, π, up, nsamps, max_steps, Rmax = planner.solver.delta, planner.pomdp, planner.solver.π, planner.solver.up, planner.solver.nsamps, planner.solver.max_steps, planner.solver.Rmax
    b = bnode.b

    # check if we have already explored from this belief: if so, reuse sampled aors. otherwise, generate new
    a_to_orb = bnode.initialized ? bnode.a_to_orb : initialize_search_space!(planner,bnode) 

    A = bnode.as
    planner.solver.verbose && println("Current belief is: ",b)

    if d == 0 || bnode.Uhi - bnode.Ulo ≤ delta
        planner.solver.verbose && (d==0 ? println("Pruned: d==0") : println("Pruned: gap ≤ delta"))
        return
    end

    a = argmax(a -> lookahead(pomdp,bn -> bn.Uhi,bnode,a), A)
    obs,rs,ws,bns = bnode.a_to_orb[a]    
    o,bnp = best_ob(planner,bnode,a)

    planner.solver.verbose && print("(level ",d," to go) exploring from belief ",b," \nwith action: ",a,", observation: ",o)
    planner.solver.verbose && println("\nresulting in belief: ",bnp.b)

    heuristic_search!(planner,bnp,d-1)
    upper = maximum(lookahead(pomdp,bn -> bn.Uhi,bnode,a) for a in A)
    lower = maximum(lookahead(pomdp,bn -> bn.Ulo,bnode,a) for a in A)
    bnode.Uhi = upper
    bnode.Ulo = lower

    planner.solver.verbose && println("After recursion, updating belief",b," to Upper: ",upper,", lower: ",lower,". completed ",d," levels")
end

"""
Rollout the provided rollout policy for at most max_steps, starting from b. 
"""
function rollout(pomdp::POMDP,π::Policy,up::Updater,b,max_steps)
    sim = RolloutSimulator(max_steps=max_steps)
    r = simulate(sim, pomdp, π,up,b)
end

"""
One step lookahead update of value estimates. Expecations are approximated using the montecarlo samples.
"""
function lookahead(pomdp::POMDP,U,bnode::BNode,a)
    obs,rs,ws,bns = bnode.a_to_orb[a]
    ws = ws/sum(ws)
    r = dot(ws,rs)
    return r + discount(pomdp)*dot([U(bn) for bn in bns],ws)
end

"""
Implements the POMDPs action method. Initializes the first root node of the belief search tree and then begins the up to k_max searches. 
"""
function POMDPs.action(planner::GapHeuristicSearchPlanner, b)
    k_max, d_max, delta, pomdp, up = planner.solver.k_max, planner.solver.d_max, planner.solver.delta, planner.pomdp, planner.solver.up
    B,A,O = get_type(planner)
    root = BNode{B,A,O}(b = b,is_root = true)
    planner.root = root
    
    initialize_search_space!(planner,planner.root) # Generate first level of explorations here first

    for i in 1:k_max
        planner.solver.verbose && println("beginning search ",i)
        heuristic_search!(planner,root,d_max)
        if root.Uhi - root.Ulo < delta
            planner.solver.verbose && println("Breaking due to gap ",root.Uhi - root.Ulo,"<=",delta)
            break
        end
    end
    
    # return the best action accoridng to 1-step lookahead with the lower bound on belief state value
    a =  argmax(a -> lookahead(pomdp,bn -> bn.Ulo,root,a),root.as) 
    return a
end

@POMDP_require POMDPs.action(planner::GapHeuristicSearchPlanner, b) begin
    M = typeof(planner.pomdp)
    S = statetype(planner.pomdp)
    (B,A,O) = get_type(planner)
    U = typeof(planner.solver.up)

    @req actions(::M,::B)
    @req update(::U,::B,::A,::O)
    # @req @gen(:o,:r)(::M, ::S, ::A)
    @req gen(::M, ::S, ::A, ::AbstractRNG)
end