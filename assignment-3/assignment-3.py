'''
Group Members
    Ujjwal Kumar (CS23BTNSK11002)
    Anvitha (CS23BTNSK11001)

  

'''



import numpy as np

EPS = 1e-6


def is_degenerate(P1, P2, P3):
    # Get any feasible point for the given configuration
    X = get_feasible_point(P1, P2, P3)

    # Find number of rows satisfied by X with equality
    equality_indices = np.where(np.abs(np.dot(P1, X)-P2) < EPS)[0]

    # If number of rows is not equal to number of variables
    # It is degenerate(no unique solution)
    if len(equality_indices) == P1.shape[1]:
        return False
    return True


def make_non_degenerate(P1, P2, P3):
    rows_to_be_modified = P1.shape[0]-P1.shape[1]

    num_iter = 0
    while True:
        if(num_iter < 1000):
            num_iter += 1

            temp_B = P2
            temp_B[:rows_to_be_modified] += np.random.uniform(
                EPS, EPS*10, size=rows_to_be_modified)
        else:
            temp_B = P2
            temp_B[:rows_to_be_modified] += np.random.uniform(
                0.1, 10, size=rows_to_be_modified)

        # If degeneracy is removed, Exit
        if not is_degenerate(P1, temp_B, P3):
            print('Degeneracy removed')
            break
    return P1, temp_B, P3


def get_feasible_point(P1, P2, P3):

    if np.all((P2 >= 0)):
        return np.zeros(P3.shape)
    else:
        for _ in range(50):
            # consider any random n of m constraints
            m = P1.shape[0]
            n = P1.shape[1]
            random_indices = np.random.choice(m, n)
            random_A = P1[random_indices]
            random_B = P2[random_indices]
            try:
                # Find Equality Solution for this n constraints
                possible_X = np.dot(np.linalg.inv(random_A), random_B)

                # If the calculated X is satisfying all of the constraint
                if (np.all((np.dot(P1, possible_X) - P2 <= 0))):
                    return possible_X
                else:
                    continue

            except:
                pass


def get_neighbour(P1, P2, P3, X):
    # Get direction vectors of vertex X
    Z = get_direction(P1, P2, P3, X)

    # Find costs through these directions
    costs = np.dot(Z, P3)

    # Find Directions with positive costs
    positive_cost_directions = np.where(costs > 0)[0]

    if len(positive_cost_directions) == 0:
        return None
    else:
        # Consider positive cost direction vector
        v = Z[positive_cost_directions[0]]

        # If there is no bound in the present direction
        if len(np.where(np.dot(P1, v) > 0)[0]) == 0:
            print('Given LP is Unbounded')
            exit()

        # Find P1'' = Matrix of rows other than satisfied by X with equality
        # P2'' = Corresponding P2 values of above rows
        equality_indices = np.where(np.abs(np.dot(P1, X)-P2) < EPS)[0]
        not_equality_indices = ~np.isin(np.arange(len(P1)), equality_indices)
        not_equal_A = P1[not_equality_indices]
        not_equal_B = P2[not_equality_indices]

        # Find maximum t in feasible neighbour(X + tv)
        n = not_equal_B - np.dot(not_equal_A, X)
        d = np.dot(not_equal_A, v)
        n = n[np.where(d > 0)[0]]
        d = d[np.where(d > 0)[0]]
        s = n/d
        t = np.min(s[s >= 0])

        # Return the maximum feasible neighbour of X
        return X + t*v


def get_direction(P1, P2, P3, X):
    equality_indices = np.where(np.abs(np.dot(P1, X)-P2) < EPS)[0]
    A_bar = P1[equality_indices]

    # Find Z = Matrix having direction vectors as columns
    Z = -np.linalg.inv(np.transpose(A_bar))

    return Z


def SimplexAlgorithm(P1, P2, P3, X):
    while True:
        # Find neighbour with greater cost
        V = get_neighbour(P1, P2, P3, X)

        # If the neighbour isn't available
        # the present vertex is the optimal
        # else move to neighbour
        if V is None:
            break
        else:
            print('Current Vertex: ', np.round(V))
            print(np.dot(P3, V))
            X = V
    return X

print("\nThe Simplex Algorithm for non-degenerate bounded polytope.\n")
def read_csv(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = [list(map(float, line.strip().split(','))) for line in lines]
    
    z = np.array(data[0][:-1])
    c = np.array(data[1][:-1])
    P1 = np.array([row[:-1] for row in data[2:]])
    b = np.array([row[-1] for row in data[2:]])

    return c, P1, b,z



def main():
    # Take input with tab spaces
    file_path = "assignment-3\input3.csv"  # Replace with your input file path
    c, P1, b, z = read_csv(file_path)
    print("c: ", c)
    print("P1: ", P1)
    print("b: ", b)
    print("z: ", z)

    P1,P2,P3 = P1,b,c
    # Change Input to non degenerate
    P1, P2, P3 = make_non_degenerate(P1, P2, P3)

    print('Initial Feasible Point: ', z)
    print('Initial Objective Value : ', np.dot(P3, z))
    print("\n")
    # Find the initial feasible point
    X = get_feasible_point(P1, P2, P3)

    # Run Simplex Algorithm and find optimal solution
    X = SimplexAlgorithm(P1, P2, P3, X)
    print('Solution is :', X)
    print('The Maximum Value : ', np.dot(P3, X))


if __name__ == "__main__":
    main()
