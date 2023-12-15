'''
Group Members
    Ujjwal Kumar (CS23BTNSK11002)
    Anvitha (CS23BTNSK11001)
    Ritvik Sai C (CS21BTECH11054)
    Nishanth Bhoomi (CS21BTECH11040)
  
'''
import numpy as np
import pandas as pd

MVAL = 1e-6

def is_solution_degenerate(P1, P2, P3):
    X = get_feasible_point(P1, P2, P3)
    equ_ind = np.where(np.abs(np.dot(P1, X) - P2) < MVAL)[0]
    if len(equ_ind) == P1.shape[1]:
        return False
    return True

def make_non_degenerate(P1, P2, P3):
    rows_to_be_modified = P1.shape[0] - P1.shape[1]
    num_iter = 0
    while True:
        if num_iter < 1000:
            num_iter += 1
            temp_B = P2.copy()
            temp_B[:rows_to_be_modified] += np.random.uniform(MVAL, MVAL * 10, size=rows_to_be_modified)
        else:
            temp_B = P2.copy()
            temp_B[:rows_to_be_modified] += np.random.uniform(0.1, 10, size=rows_to_be_modified)

        if not is_solution_degenerate(P1, temp_B, P3):
            print('Degeneracy has been removed\n')
            break
    return P1, temp_B, P3

def get_feasible_point(P1, P2, P3):
    if np.all((P2 >= 0)):
        return np.zeros(P3.shape)
    else:
        for _ in range(50):
            m = P1.shape[0]
            n = P1.shape[1]
            random_ind = np.random.choice(m, n)
            random_A = P1[random_ind]
            random_B = P2[random_ind]
            try:
                possible_X = np.dot(np.linalg.inv(random_A), random_B)

                if np.all((np.dot(P1, possible_X) - P2 <= 0)):
                    return possible_X
                else:
                    continue

            except:
                pass

def get_neighbour(P1, P2, P3, X):
    Z = get_direction(P1, P2, P3, X)
    costs = np.dot(Z, P3)

    positive_cost_directions = np.where(costs > 0)[0]

    if len(positive_cost_directions) == 0:
        return None
    else:
        v = Z[positive_cost_directions[0]]

        # If there is no bound in the present direction
        if len(np.where(np.dot(P1, v) > 0)[0]) == 0:
            print('Given LP is Unbounded\n')
            exit()

        equ_ind = np.where(np.abs(np.dot(P1, X) - P2) < MVAL)[0]
        not_equ_ind = ~np.isin(np.arange(len(P1)), equ_ind)
        not_equal_A = P1[not_equ_ind]
        not_equal_B = P2[not_equ_ind]

        n = not_equal_B - np.dot(not_equal_A, X)
        d = np.dot(not_equal_A, v)
        n = n[np.where(d > 0)[0]]
        d = d[np.where(d > 0)[0]]
        s = n / d
        t = np.min(s[s >= 0])

        return X + t * v

def get_direction(P1, P2, P3, X):
    equ_ind = np.where(np.abs(np.dot(P1, X) - P2) < MVAL)[0]
    A_bar = P1[equ_ind]

    Z = -np.linalg.inv(np.transpose(A_bar))
    return Z


def SimplexAlgorithm(P1, P2, P3, X, visited_vertices, cost_values):
    while True:
        # Check if the current vertex is already visited
        if any(np.all(np.isclose(X, v, atol=MVAL)) for v in visited_vertices):
            print('Cycle detected. Exiting.\n')
            break

        visited_vertices.append(np.round(X.copy(), 1))
        cost_values.append(round(np.dot(P3, X), 1))

        V = get_neighbour(P1, P2, P3, X)

        if V is None:
            break
        else:
            X = V
    return X

print("\nThe Simplex Algorithm for non-degenerate bounded polytope.")

data = pd.read_csv('input3.csv', header=None)
data = data.fillna('')
data = data.astype(str)

data = data.apply(lambda x: pd.to_numeric(x, errors='coerce')).astype(float)
data = data.values.tolist()

z = np.array(data[0][:-1])
C = np.array(data[1][:-1])
P1 = np.array([row[:-1] for row in data[2:]])
B = np.array([row[-1] for row in data[2:]])

P1, P2, P3 = P1, B, C

P1, P2, P3 = make_non_degenerate(P1, P2, P3)

visited_vertices = []
cost_values = []

X = get_feasible_point(P1, P2, P3)

X = SimplexAlgorithm(P1, P2, P3, X, visited_vertices, cost_values)

optimal_solution = np.round(X, 1)
optimal_solution_str = "[" + " , ".join(map(str, optimal_solution)) + "]"
print('Optimal Solution is ', optimal_solution_str)
print('The Maximum Objective Value is ', round(np.dot(P3, X), 1))
print('\nSequence of Visited Vertices:')
for i, (vertex, cost) in enumerate(zip(visited_vertices, cost_values)):
            vertex_str = "[" + " , ".join(map(str, np.round(vertex, 1))) + "]"
            print(f"Vertex{i + 1}: Vertex: {vertex_str} , Objective Value: {round(cost, 1)}")

