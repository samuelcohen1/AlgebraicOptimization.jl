using Test
using AlgebraicOptimization
using Catlab

seed_random()

d = @relation (x,y,z) begin
    f(w,x)
    g(u,w,y)
    h(u,w,z)
end


# Test flow graph composition
g1 = random_open_flowgraph(10, .2, 2)
g2 = random_open_flowgraph(10, .2, 3)
g3 = random_open_flowgraph(10, .2, 3)

g_comp = oapply(d, [g1, g2, g3])

# Test naturality of MCNF
γ = 0.1
iters = 2000
p1 = to_problem(g1)
p2 = to_problem(g2)
p3 = to_problem(g3)
p_comp1 = oapply(d, [p1, p2, p3])

p_comp2 = to_problem(g_comp)

opt1 = Euler(gradient_flow(p_comp1), γ)
opt2 = Euler(gradient_flow(p_comp2), γ)

@time begin r11 = simulate(opt1, zeros(length(opt1.S)), iters) end
@time begin r22 = simulate(opt2, zeros(length(opt2.S)), iters) end


o1 = dual_decomposition(g1, 0.1)             # Why the redefinition?
o2 = dual_decomposition(g2, 0.1)
o3 = dual_decomposition(g3, 0.1)

# o1 = Euler(gradient_flow(p1),γ)
# o2 = Euler(gradient_flow(p2),γ)
# o3 = Euler(gradient_flow(p3),γ)

comp_opt1 = oapply(OpenDiscreteOpt(), d, [o1,o2,o3])
comp_opt2 = dual_decomposition(g_comp, γ)

@time begin res1 = simulate(comp_opt1, zeros(length(comp_opt1.S)), iters) end
@time begin res2 = simulate(comp_opt2, zeros(length(comp_opt2.S)), iters) end


@test r11 ≅ r22
@test res1 ≅ res2
@test r11 ≅ res1   # Why are we checking this?

# Test: did it satisfy the flow constraints?
small_solution = primal_solution(data(p_comp2), res1)



println(node_incidence_matrix(data(g_comp)) * small_solution - data(g_comp).flows)

@test node_incidence_matrix(data(g_comp)) * small_solution ≅ data(g_comp).flows





# Small testable flowgraph situation


γ = 0.1
iters = 4000

small_graph = Graph()
add_vertices!(small_graph, 2)
add_edge!(small_graph, 1, 2)
# show(small_graph)

small_flowgraph = FlowGraph(small_graph, [x ->  1e9 + .01 * x^2], [10, -10])
small_openflowgraph = Open{FlowGraph}(FinSet(2), small_flowgraph, id(FinSet(2)))
small_problem = to_problem(small_openflowgraph)
small_opt = Euler(gradient_flow(small_problem), γ)
small_sim = simulate(small_opt, [0, 0], iters)                 # Why are there 2 values here? Is it one for the x and one for the dual? Are they only duals?
small_solution = primal_solution(data(small_problem), small_sim)

# Test: did it satisfy the flow constraints?
node_incidence_matrix(small_flowgraph) * small_solution - small_flowgraph.flows
@test node_incidence_matrix(small_flowgraph) * small_solution ≅ small_flowgraph.flows



small_graph = Graph()
add_vertices!(small_graph, 4)
add_edge!(small_graph, 1, 2)
add_edge!(small_graph, 2, 4)
add_edge!(small_graph, 1, 3)
add_edge!(small_graph, 3, 4)

# show(small_graph)

small_flowgraph = FlowGraph(small_graph, [x -> 5*x^2 - 4, x -> 5*x^2 - 4, x -> 1*x^2 + 3, x -> 1*x^2], [-10, 0, 0, 10])
small_openflowgraph = Open{FlowGraph}(FinSet(4), small_flowgraph, id(FinSet(4)))
small_problem = to_problem(small_openflowgraph)
small_opt = Euler(gradient_flow(small_problem), γ)
small_sim = simulate(small_opt, zeros(length(small_opt.S)), iters)
small_solution = primal_solution(data(small_problem), small_sim)

# Test: did it satisfy the flow constraints?
node_incidence_matrix(small_flowgraph) * small_solution - small_flowgraph.flows
@test node_incidence_matrix(small_flowgraph) * small_solution ≅ small_flowgraph.flows









