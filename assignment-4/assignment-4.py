'''
Group Members
    Ujjwal Kumar (CS23BTNSK11002)
    Anvitha (CS23BTNSK11001)
    Ritvik Sai C (CS21BTECH11054)
    Nishanth Bhoomi (CS21BTECH11040)
  
'''
import numpy as np
import csv
import pandas as pd
threshold_value = pow(10, -8)

class SimplexAlgorithm:
    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c
        size = A.shape
        self.m = size[0]
        self.n = size[1]
        self.b = self.remove_degeneracy()

        self.visited_vertices = []
        self.cost_values = []  

        self.initial_feasible_point = None
        self.z = self.feasible_point()

        result = self.main_simplex()
        optimal_solution = np.round(result, 1)
        optimal_solution_str = "[" + " , ".join(map(str, optimal_solution)) + "]"
        print(f"Optimal solution is {optimal_solution_str}")
        print(f"The Maximum Objective Value is {round(np.dot(self.c, result), 1)}")

        print("\nSequence of Visited Vertices:")
        for i, (vertex, cost) in enumerate(zip(self.visited_vertices, self.cost_values)):
            vertex_str = "[" + " , ".join(map(str, np.round(vertex, 1))) + "]"
            print(f"Vertex{i + 1}: Vertex: {vertex_str} , Objective Value: {round(cost, 1)}")

    def initialize_random_val(self, a):
        if a == threshold_value:
            self.rand_val = np.random.uniform(threshold_value, threshold_value * 10, size=self.row_mod)
        else:
            self.rand_val = np.random.uniform(0.1, 10, size=self.row_mod)

    def feasible_point(self):
        if np.all(self.b >= 0):
            t = self.c.shape
            return np.zeros(t)
        else:
            j = 0
            while j < 50:
                rand_ind = np.random.choice(self.m, self.n)
                r_A = self.A[rand_ind]
                r_b = self.b[rand_ind]
                try:
                    temp = np.linalg.inv(r_A)
                    temp1 = np.dot(temp, r_b)
                    temp2 = np.dot(self.A, temp1) - self.b
                    if np.all(temp2 <= 0):
                        self.initial_feasible_point = temp1
                        return temp1
                    else:
                        continue
                except:
                    pass
                j += 1

    def remove_degeneracy(self):
        self.row_mod = self.m - self.n
        i = 0
        num = 1000
        while True:
            if i >= num:
                X = self.b
                self.initialize_random_val(0.1)
                X[:self.row_mod] = X[:self.row_mod] + self.rand_val

            else:
                i += 1
                X = self.b
                self.initialize_random_val(threshold_value)
                X[:self.row_mod] = X[:self.row_mod] + self.rand_val

            Y = self.feasible_point()

            # Handle the error for finding no feasible points
            if Y is None:
                print("Feasible point not found.")
                exit()
    
            e_ind = np.where(np.abs(np.dot(self.A, Y) - X) < threshold_value)[0]
            if len(e_ind) == self.n:
                print("Non degenerate..")
                break
        return X

    def calculate_direction(self):
        x = np.dot(self.A, self.z) - self.b
        e_ind = np.where(np.abs(x) < threshold_value)[0]
        A_dash = self.A[e_ind]
        self.direction_vector = -np.linalg.pinv(np.transpose(A_dash))

    def find_maximum_alpha(self):
        n = self.n_b - np.dot(self.n_A, self.z)
        d = np.dot(self.n_A, self.u)
        n = n[np.where(d > 0)[0]]
        d = d[np.where(d > 0)[0]]

        if len(n) == 0 or len(d) == 0:
            print("Error: Empty array")
            exit()

        s = n / d
        self.alpha = np.min(s[s >= 0])


    def find_feasible_neighbour(self):
        self.calculate_direction()
        cost_vector = np.dot(self.direction_vector, self.c)
        positive_cost_indices = np.where(cost_vector > 0)[0]

        if len(positive_cost_indices) == 0:
            return None
        else:
            self.u = self.direction_vector[positive_cost_indices[0]]
            temp1 = np.where(np.dot(self.A, self.u) > 0)
            temp2 = temp1[0]
            if len(temp2) == 0:
                print("Unbounded LP")
                exit()

            x = np.abs(np.dot(self.A, self.z) - self.b)
            e_ind = np.where(x < threshold_value)[0]
            non_e_ind = ~np.isin(np.arange(len(self.A)), e_ind)

            self.n_A = self.A[non_e_ind]
            self.n_b = self.b[non_e_ind]

            self.find_maximum_alpha()

            return self.z + self.alpha * self.u

    def main_simplex(self):
        if np.all(np.dot(self.A, self.z) - self.b <= threshold_value):
            self.visited_vertices.append(np.round(self.z.copy(), 1))
            self.cost_values.append(round(np.dot(self.c, self.z), 1))  

        print(f"Initial Feasible Point is {[val for val in np.round(self.initial_feasible_point, 1)]}")
        while True:
            result = self.find_feasible_neighbour()
            if result is None:
                break
            else:
                self.z = result
                self.visited_vertices.append(np.round(self.z.copy(), 1))
                self.cost_values.append(round(np.dot(self.c, self.z), 1))                 
        return self.z


# Read input from CSV file
data=pd.read_csv("input4.csv",header=None)
data=data.fillna('')
data = data.astype(str)
data= data.values.tolist()

# Extract data from CSV
c = np.array([float(val) for val in data[0][:-1]])
b = np.array([float(row[-1]) if row[-1] != '' else 0.0 for row in data[1:]])
A = np.array([[float(val) if val != '' else 0.0 for val in row[:-1]] for row in data[1:]])

SimplexAlgorithm(A, b, c)
