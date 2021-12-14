# GHS.jl
A Julia implementation of the Gap Heuristic Search online planning algorithm, for use with the POMDPs.jl ecosystem. 

## Installation
In Julia, type `]add MCTS`

## Documentation
TODO

## Usage
```julia
If `pomdp` is a POMDP defined with the [POMDPs.jl](https://github.com/sisl/POMDPs.jl) interface, the GHS solver can be used to find an optimized action, `a`, for the POMDP in belief state `b` as follows:

using POMDPs
using POMDPModels # for the CryingBaby problem
using POMDPPolicies
using BeliefUpdaters
using GHS

pomdp = BabyPOMDP()
roller = RandomPolicy(pomdp)
up = DiscreteUpdater(pomdp)
Rmax = 0.0  # best possible reward is baby not hungry, didnt feed
solver = GapHeuristicSearchSolver(roller,up,Rmax,delta=.1,k_max=100,d_max=10,nsamps=10,max_steps=20,verbose=false)
ghs_policy = solve(solver, cryingbaby)

b_hungry = DiscreteBelief(cryingbaby,[.1,.9])

a = action(ghs_policy, b_hungry)
```
