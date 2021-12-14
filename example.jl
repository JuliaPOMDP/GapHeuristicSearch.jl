using POMDPs
using POMDPPolicies
# using POMDPModelTools
using POMDPSimulators
using POMDPModels
using BeliefUpdaters
# using FIB
# using SARSOP
# using StaticArrays
# using Random
# using Statistics
# using DataFrames
# using ParticleFilters
# using RockSample 
# using POMDPGifs
# import Cairo, Fontconfig
# using Parameters
# using StatsBase
# using DataStructures
# using LinearAlgebra

push!(LOAD_PATH,"/Users/ericlux/Desk/GHS.jl")
# include("GapHeuristicSearch.jl")
using GHS

# stuff to varify on paratrial for crying baby

cryingbaby = BabyPOMDP()
roller = RandomPolicy(cryingbaby)
up = DiscreteUpdater(cryingbaby)
Rmax = 0.0  # best possible reward is baby not hungry, didnt feed
solver = GapHeuristicSearchSolver(roller,
                                up,
                                Rmax,
                                delta=.1,
                                k_max=100,
                                d_max=10,
                                nsamps=10,
                                max_steps=20)

ghs_policy = solve(solver, cryingbaby)
# Base.show(io::IO,x::DiscreteBelief) = print(x.b)

b_hungry = DiscreteBelief(cryingbaby,[.1,.9])
b_nothungry = DiscreteBelief(cryingbaby,[.9,0.1])

for i in 1:2
    println(action(ghs_policy, b_hungry))
end

println(get_type(ghs_policy))

# sarsop_solver = SARSOPSolver()
# sarsop_policy = solve(sarsop_solver, cryingbaby)
# runs = 100
# q_para = [] # vector of the simulations to be run
# [push!(q_para, Sim(cryingbaby, sarsop_policy, max_steps=50, rng=MersenneTwister(), metadata=Dict(:policy=>"sarsop"))) for i in 1:runs]
# [push!(q_para, Sim(cryingbaby, ghs_policy,ghs_policy.solver.up, max_steps=50, rng=MersenneTwister(), metadata=Dict(:policy=>"ghs"))) for i in 1:runs]
