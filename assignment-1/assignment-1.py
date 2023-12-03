'''
Group Members
    Ujjwal Kumar (CS23BTNSK11002)
    Anvitha (CS23BTNSK11001)
    Member3 (Roll)
    Member4 (Roll)
  
'''

import numpy as np
import csv
import pandas as pd

threshold_val_Th = pow(10, -8)

class Simpex_Algorithm:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        size = w.shape
        self.m = size[0]
        self.n = size[1]

        self.visited_vertices = []

        result = self.main_simplex()
        print(f"Optimal solution : {result}")
        print(f"The Maximum Objective Value : {np.dot(self.y, result)}")


        print("\nThe Sequence of Visited Vertices :")
        for vertex in self.visited_vertices:
            print(f"Vertex: {vertex}, Objective Value: {np.dot(self.y, vertex)}")

    def calculate_direction(self):
        x = np.dot(self.w, self.z) - self.x
        e_indices = np.where(np.abs(x) < threshold_val_Th)[0]
        self.A_dash = self.w[e_indices]
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
        cost_vector = np.dot(self.direction_vector, self.y)
        positive_cost_indices = np.where(cost_vector > 0)[0]

        if len(positive_cost_indices) == 0:
            return None
        else:
            self.u = self.direction_vector[positive_cost_indices[0]]
            x = np.abs(np.dot(self.w, self.z) - self.x)
            e_ind = np.where(x < threshold_val_Th)[0]
            non_e_ind = ~np.isin(np.arange(len(self.w)), e_ind)

            self.n_A = self.w[non_e_ind]
            self.n_b = self.x[non_e_ind]

            self.find_maximum_alpha()

            return self.z + self.alpha * self.u

    def main_simplex(self):
        if np.all(np.dot(self.w, self.z) - self.x <= threshold_val_Th):
            self.visited_vertices.append(self.z.copy())
            
        while True:
            result = self.find_feasible_neighbour()
            if result is None:
                break
            else:
                self.z = result
                self.visited_vertices.append(result)
        return self.z


data=pd.read_csv("assignment-1\input1.csv",header=None)
data=data.fillna('')
data = data.astype(str)
data= data.values.tolist()
print(data)
print("\n")
    # Extract data from CSV
z = np.array([float(value) for value in data[0][:-1]])
y = np.array([float(value) for value in data[1][:-1]])
x = np.array([float(row[-1]) for row in data[2:]])
w = np.array([[float(value) for value in row[:-1]] for row in data[2:]])
    # Call the constructor and run the functions
Simpex_Algorithm(w, x, y, z)