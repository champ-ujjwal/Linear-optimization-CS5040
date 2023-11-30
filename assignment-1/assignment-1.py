import numpy as np
import pandas as pd

epsilon = 1e-8
class simplex_algorithm:
    def __init__(self, A, b, c, z):
        self.A = A
        self.b = b
        self.c = c
        self.z = z
        size = A.shape
        self.m = size[0]
        self.n = size[1]

        result = self.main_simplex()
        print(f"Optimal solution is {result}")
        print(f"The Maximum Value is {np.dot(self.c, result)}")

    # function to get the direction of the z.
    def direction(self):
        x = np.dot(self.A, self.z) - self.b
        e_ind = np.where(np.abs(x) < epsilon)[0]
        self.A_dash = self.A[e_ind]
        self.d_vec = -np.linalg.inv(np.transpose(self.A_dash))

    # function to find alpha
    def maximum_alpha(self):
        n = self.n_b - np.dot(self.n_A, self.z)
        d = np.dot(self.n_A, self.u)
        n = n[np.where(d > 0)[0]]
        d = d[np.where(d > 0)[0]]
        s = n / d
        self.alpha = np.min(s[s >= 0])

    # function to find maximum feasible neighbor of z.
    def neighbor(self):
        self.direction()
        ct = np.dot(self.d_vec, self.c)
        p_cost = np.where(ct > 0)[0]

        if len(p_cost) == 0:
            return None
        else:
            self.u = self.d_vec[p_cost[0]]
            x = np.abs(np.dot(self.A, self.z) - self.b)
            e_ind = np.where(x < epsilon)[0]
            n_e_ind = ~np.isin(np.arange(len(self.A)), e_ind)

            self.n_A = self.A[n_e_ind]
            self.n_b = self.b[n_e_ind]

            self.maximum_alpha()

            return self.z + self.alpha * self.u

    def main_simplex(self):
        while True:
            result = self.neighbor()
            if result is None:
                break
            else:
                self.z = result
        return self.z

def main():
    # Read input from CSV file
  df = pd.read_csv('input1.csv', header=None)

    m, n = df.iloc[0]

    # Extract A, b, c, z from the dataframe
    A = df.iloc[1:m + 1, :n].to_numpy()
    b = df.iloc[m + 1].to_numpy()
    c = df.iloc[m + 2].to_numpy()
    z = df.iloc[m + 3].to_numpy()

    print(f"A : {A} \nB : {b} \nC : {c} \nz : {z}")
    print("-------------------------------------------")

    # Call the constructor and run the functions
    simplex_algorithm(A, b, c, z)

if __name__ == '__main__':
    main()
