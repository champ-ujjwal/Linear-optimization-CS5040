'''
Group Members
    Ujjwal Kumar (CS23BTNSK11002)
    Anvitha (CS23BTNSK11001)
    Ritvik Sai C (CS21BTECH11054)
    Nishanth Bhoomi (CS21BTECH11040)
  

'''

import numpy as np
import csv

threshold_value = pow(10, -8)

class SimplexAlgorithm:
    def __init__(self, A, b, c, z):
        self.A = A
        self.b = b
        self.c = c
        self.z = z
        size = A.shape
        self.m = size[0]
        self.n = size[1]

        self.visited_vertices = []

        result = self.main_simplex()
        print(f"Optimal solution: {result}")
        print(f"Maximum Objective Value: {np.dot(self.c, result)}")

        print("\nSequence of Visited Vertices:")
        for vertex in self.visited_vertices:
            print(f"Vertex: {vertex}, Objective Value: {np.dot(self.c, vertex)}")

    def calculate_direction(self):
        x = np.dot(self.A, self.z) - self.b
        e_indices = np.where(np.abs(x) < threshold_value)[0]
        self.A_dash = self.A[e_indices]
        self.direction_vector = -np.linalg.inv(np.transpose(self.A_dash))

    def find_maximum_alpha(self):
        n = self.n_b - np.dot(self.n_A, self.z)
        d = np.dot(self.n_A, self.u)
        n = n[np.where(d > 0)[0]]
        d = d[np.where(d > 0)[0]]
        s = n / d
        self.alpha = np.min(s[s >= 0])

    def find_feasible_neighbour(self):
        self.calculate_direction()
        ct = np.dot(self.direction_vector, self.c)
        temp = np.where(ct > 0)
        positive_cost_indices = temp[0]
        if len(positive_cost_indices) == 0:
            return None
        else:
            self.u = self.direction_vector[positive_cost_indices[0]]

            # Check for unbounded LP based on the u vector
            if np.all(np.dot(self.A, self.u) <= 0):
                print("Unbounded LP")
                exit()

            x = np.abs(np.dot(self.A, self.z) - self.b)
            e_ind = np.where(x < threshold_value)[0]
            n_e_ind = ~np.isin(np.arange(len(self.A)), e_ind)

            self.n_A = self.A[n_e_ind]
            self.n_b = self.b[n_e_ind]

            self.find_maximum_alpha()

            return self.z + self.alpha * self.u


    def main_simplex(self):
        if np.all(np.dot(self.A, self.z) - self.b <= threshold_value):
            self.visited_vertices.append(self.z.copy())

        while True:
            result = self.find_feasible_neighbour()
            if result is None:
                break
            else:
                self.z = result.copy()
                self.visited_vertices.append(result)
        return self.z

def main():
    # Read input from CSV file
    with open('assignment-2\input2.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    # Extract data from CSV
    z = np.array([float(val) for val in data[0][:-1]])
    c = np.array([float(val) for val in data[1][:-1]])
    b = np.array([float(row[-1]) for row in data[2:]])
    A = np.array([[float(val) for val in row[:-1]] for row in data[2:]])

    # Call the constructor and run the functions
    SimplexAlgorithm(A, b, c, z)

if __name__ == '_main_':
    main()
