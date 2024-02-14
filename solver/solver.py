import cplex

def solve_MILP(
        item_sizes=[0.42, 0.58, 0.28, 0.57],
        savings=[0.47, 1.92, 4.63, 5.12],
        dataset_sizes=[78848, 108544, 53248, 107520],
        M=102400,
        C=188000,
        tight=False
    ):
    # Create an instance of CPLEX
    problem = cplex.Cplex()
    # Maximize the time savings
    problem.objective.set_sense(problem.objective.sense.maximize)
    
    # Time limit of 5 minutes
    problem.parameters.timelimit.set(300)
    # Set the memory limit to 4GB
    problem.parameters.workmem.set(8192)

    # Define the decision variables
    c = [f"c{i}" for i in range(len(savings))]
    problem.variables.add(obj=savings, lb=[0.0 for _ in range(len(savings))], names=c)

    # Define the constraints

    # Constraint 1: c1*s1 + c2*s2 + c3*s3 + c4*s4 <= M
    constraint_1 = [[c, item_sizes]]
    problem.linear_constraints.add(lin_expr=constraint_1, senses=["L"], rhs=[M], names=["constraint_1"])

    # Constraint 2: c1 + c2 + c3 + c4 <= C
    # Can change it to be equal if I want to cache EVERYTHING
    constraint_2 = [[c, [1 for _ in range(len(savings))]]]
    problem.linear_constraints.add(lin_expr=constraint_2, senses=["L"] if not tight else ["E"], rhs=[C], names=["constraint_2"])

    # Constraint 3: c1*s1 <= d1 ; c2*s2 <= d2 ; c3*s3  <= d3 ; c4*s4 <= d4
    constraint_3_lhs = []
    for i in range(len(item_sizes)):
        index_map = [0 for _ in range(len(item_sizes)-1)]
        index_map.insert(i, item_sizes[i])
        constraint_3_lhs.append([c] + [index_map])

    constraint_3_senses = ["L" for _ in range(len(constraint_3_lhs))]
    problem.linear_constraints.add(lin_expr=constraint_3_lhs, senses=constraint_3_senses, rhs=dataset_sizes, names=[f"constraint_3_{i}" for i in range(1, len(dataset_sizes)+1)])

    # Set the problem type to an integer linear program
    problem.set_problem_type(problem.problem_type.MILP)

    # Solve the problem
    problem.solve()
    # Print the solution
    return [problem.solution.get_values(c[i]) for i in range(len(c))]

def get_cache_solution(n_files, profile_data, cache_size, tight=False):
    steps = sorted(profile_data.keys())
    item_sizes = [profile_data[step][0] for step in steps]
    item_savings = [profile_data[step][1] for step in steps]
    dataset_sizes = [i*n_files for i in item_sizes]
    solution = solve_MILP(
        item_sizes, item_savings, dataset_sizes, cache_size * (1 << 10), n_files, tight=tight)
    cache_steps = []
    cache_sizes = []
    cached_tensors = sum(solution)
    for i in range(len(solution)):
        if solution[i] > 0:
            cache_steps.append(steps[i])
            cache_sizes.append(round((solution[i]*item_sizes[i]) / (1 << 10), 2))
    print(cache_sizes, cache_steps, cached_tensors)
    return cache_sizes, cache_steps, cached_tensors
