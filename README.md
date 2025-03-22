# GHS.jl

[![CI](https://github.com/Aero-Spec/GapHeuristicSearch.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Aero-Spec/GapHeuristicSearch.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/Aero-Spec/GapHeuristicSearch.jl/graph/badge.svg)](https://codecov.io/github/Aero-Spec/GapHeuristicSearch.jl)

---

A Julia implementation of the Gap Heuristic Search online planning algorithm, for use with the POMDPs.jl ecosystem. 

## Installation
In Julia, type `]add https://github.com/sisl/GapHeuristicSearch.jl`

## Documentation
The difference between the gap at a belief b is defined as the difference between the upper and lower bound values: U<sub>upper</sub>(b)-U<sub>lower</sub>(b).
The upper bound will be initalized by the best-action best-state upper bound if not specificed, and the lower bound will be initialized by the reward obtained by a random rollout policy if not specified.
The Gap Heuristic Search algorithm seeks to select the obeservation that maximizes the gap of belief b becaause they are more likely to benefit from a belief backup.
Actions will be selected with a lookahead using an approximate value function.
```julia
a = argmax(a -> lookahead(𝒫,b′->Uhi[b′],b,a),𝒜) # find the action that maximizes the lookahead function
o = argmax(o -> Uhi[B[(a,o)]] - Ulo[B[(a,o)]], 𝒪) # find the observation that maximizes the gap between the upper and lower bound
```
The exploration stops when the gap is smaller than the threshold $\delta$ or the maximum depth
is reached. 
## Usage

If `pomdp` is a POMDP defined with the [POMDPs.jl](https://github.com/sisl/POMDPs.jl) interface, the GHS solver can be used to find an optimized action, `a`, for the POMDP in belief state `b` as follows:
```julia
using POMDPs
using POMDPModels # for the CryingBaby problem
using POMDPPolicies
using BeliefUpdaters
using GapHeuristicSearch

pomdp = BabyPOMDP()
roller = RandomPolicy(pomdp)
up = DiscreteUpdater(pomdp)
Rmax = 0.0  # best possible reward is baby not hungry, didnt feed
solver = GapHeuristicSearchSolver(roller,
                                up,
                                Rmax,
                                delta=0.1,
                                k_max=100,
                                d_max=10,
                                nsamps=10,
                                max_steps=20)
                                
policy = solve(solver, cryingbaby)
b = DiscreteBelief(cryingbaby,[0.1,0.9])
a = action(policy, b)
```
