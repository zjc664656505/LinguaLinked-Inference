import json
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
import numpy as np

class Optimizer:
    def __init__(self, num_devices, num_modules):
        """
            Initial variables for optimizer
        """
        self.num_devices = num_devices
        self.num_modules = num_modules
        self.num_flops ={}
        self.flop_speed = [0.0]*num_devices
        self.execution_time = np.zeros([num_devices, num_modules])
        self.ping_latency = np.zeros([num_devices, num_devices])
        self.bandwidths = np.zeros([num_devices, num_devices])
        self.m2m = {} # module output size
        self.m2m_size = None
        self.model_size = {}
        self.module_size = []
        self.total_mem = np.zeros([1, num_devices])
        self.ava_mem = np.zeros([1, num_devices])
        self.info_processed = False
        self.Solu = None # initial split optimized result
        self.Strategy = None # optimized result for load-balancing modules considering overlapping memory

    def process_initial_info(self, num_flop:dict, flop_speed:list, ping_latency: np.ndarray,
                             bandwidths: np.ndarray, m2m: dict, model_size:dict,
                             total_mem: np.ndarray, ava_mem: np.ndarray):
        self.info_processed = True
        self.num_flops = num_flop
        self.flop_speed = flop_speed
        # compute submodule execution time based on flops
        for i in range(len(self.flop_speed)):
            for k, val in self.num_flops.items():
                self.execution_time[i, k] = val / self.flop_speed[i]

        self.ping_latency = ping_latency
        self.bandwidths = bandwidths
        self.m2m = m2m
        self.m2m_size = np.zeros([len(self.m2m), len(self.m2m)])

        # compute module output size and convert them into matrix in MB
        for i, res in self.m2m.items():
            print(f'res[]: {res}')
            for j, val in res['seq'].items():
                self.m2m_size[int(i)][int(j)] = sum(val) / 1000_000

            for j, val in res['res'].items():
                self.m2m_size[int(i)][int(j)] = sum(val) / 1000_000

        self.model_size = model_size
        # compute module loading size in MB
        for k, v in self.model_size.items():
            self.module_size.append(v["load"] / 1000_000)

        self.total_mem = total_mem
        self.ava_mem = ava_mem

    def initial_module_arrangement(self):
        assert self.info_processed == True, "Initial Optimization Information Must Be Processed!"
        m = gp.Model()
        x = {}
        for i in range(self.num_devices):
            for j in range(self.num_modules):
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # Auxiliary variables
        z = {}
        for i in range(self.num_devices):
            for j in range(self.num_modules):
                if (j == self.num_modules - 1):
                    if (i == self.num_devices - 1):
                        z[i, j] = m.addVar(vtype=GRB.BINARY, lb=1, name=f"z_{i}_{j}")
                    else:
                        z[i, j] = m.addVar(vtype=GRB.BINARY, ub=0, name=f"z_{i}_{j}")
                else:
                    z[i, j] = m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")

        m.update()

        def objective_function(x, execution_time):
            total = 0
            for i in range(self.num_devices):
                for j in range(self.num_modules):
                    total += execution_time[i][j] * x[i, j]
            return total

        def matrix_mul(data, x):
            interm_mat = {}
            for i in range(self.num_devices):
                for j in range(self.num_modules):
                    interm_mat[i, j] = gp.quicksum(x[i, k] * data[k, j] for k in range(self.num_modules))

            final_mat = {}
            for i in range(self.num_devices):
                for j in range(self.num_devices):
                    final_mat[i, j] = gp.quicksum(interm_mat[i, k] * x[j, k] for k in range(self.num_modules))
            return final_mat

        def objective_function2(x, ping_latency, m2m_size, bandwidths):
            total = 0
            d2d_size = matrix_mul(m2m_size, x)
            for i in range(self.num_devices):
                for j in range(self.num_devices):
                    if i != j:
                        total += ping_latency[i][j]
                        total += d2d_size[i, j] / bandwidths[i][j]

            return total

        m.setObjective(
            objective_function(x, self.execution_time) + objective_function2(x, self.ping_latency,
                                                                             self.m2m_size, self.bandwidths),
            gp.GRB.MINIMIZE)

        m.addConstr(x[0, 0] >= 1)

        # Sum of each column is 1
        for j in range(self.num_modules):
            m.addConstr(quicksum(x[i, j] for i in range(self.num_devices)) == 1, f"col_sum_{j}")

        # Constraint to set z[i][j]
        for i in range(self.num_devices):
            for j in range(self.num_modules - 1):
                m.addConstr(z[i, j] >= x[i, j] - x[i, j + 1], f"set_z1_{i}_{j}")
                m.addConstr(z[i, j] <= x[i, j], f"set_z2_{i}_{j}")
                m.addConstr(z[i, j] <= 1 - x[i, j + 1], f"set_z3_{i}_{j}")

        # Constraint to ensure no non-consecutive 1's after z[i][j] becomes 1
        for i in range(self.num_devices):
            for j in range(self.num_modules - 1):
                for k in range(j + 1, self.num_modules):
                    m.addConstr(x[i, k] <= 1 - z[i, j], f"no_non_consecutive_ones_row_{i}_self.num_modules_{j}_{k}")

        for i in range(self.num_devices):
            m.addConstr(quicksum([self.module_size[j] * x[i, j] for j in range(self.num_modules)]) <= self.ava_mem[i],
                        f"device_{i}_out_of_the_memory")

        m.optimize()

        # Print results
        if m.status == gp.GRB.OPTIMAL:
            print("\nOptimal solution found.")
            for i in range(self.num_devices):
                print()
                for j in range(self.num_modules):
                    print(f"{int(x[i, j].x)}", end=" ")
            print("\n\nAuxiliary Solution found")
            for i in range(self.num_devices):
                print()
                for j in range(self.num_modules - 1):
                    print(f"{int(z[i, j].x)}", end=" ")
            print("\n")
            print("Objective value =", m.objVal)
            for i in range(self.num_devices):
                print(f"Device {i} Memory Usage {quicksum([self.module_size[j] * x[i, j].x for j in range(self.num_modules)])}")
        else:
            print("Optimization did not converge to an optimal solution.")

        self.Solu = [[int(x[i, j].x) for j in range(self.num_modules)] for i in range(self.num_devices)]
        
        return np.array(self.Solu)

    def dynamic_module_arrangement(self):
        assert self.Solu is not None, ("Original Optimized Solution Cannot be None. Check whether"
                                       "initial_module_arrangement() is run successfully?")
        md = gp.Model("Mem issue")
        current_Mem = np.array(self.Solu) @ np.array(self.module_size).T
        print(f"The optimal memory split solution on three devices is: {current_Mem}")
        overlap = {}
        overlap_left = {}
        overlap_right = {}

        for j in range(self.num_modules):
            for i in range(self.num_devices):
                if self.Solu[i][j] == 1:
                    if i - 1 >= 0:
                        overlap[i - 1, j] = 1
                        overlap_right[i - 1, j] = md.addVar(vtype=GRB.BINARY, name=f"overlap_right_{i - 1}_{j}")
                    if i + 1 < self.num_devices:
                        overlap[i + 1, j] = 1
                        overlap_left[i + 1, j] = md.addVar(vtype=GRB.BINARY, name=f"overlap_left_{i + 1}_{j}")

        for j in range(self.num_modules):
            for i in range(self.num_devices):
                if (i, j) not in overlap:
                    overlap[i, j] = 0

        for j in range(self.num_modules):
            for i in range(self.num_devices):
                if (i, j) not in overlap_left:
                    overlap_left[i, j] = 0

        for j in range(self.num_modules):
            for i in range(self.num_devices):
                if (i, j) not in overlap_right:
                    overlap_right[i, j] = 0

        def totalMemoryUsage(i, Solu, overlap_left, overlap_right):

            Merge_Mem = Solu[i]
            if i + 1 < self.num_devices:
                Merge_Mem = [Merge_Mem[j] - overlap_left[i + 1, j] for j in range(self.num_modules)]
            if i - 1 >= 0:
                Merge_Mem = [Merge_Mem[j] - overlap_right[i - 1, j] for j in range(self.num_modules)]

            UnMerge_Mem_Left = [overlap_left[i, j] for j in range(self.num_modules)]
            if i - 1 >= 0:
                UnMerge_Mem_Left = [UnMerge_Mem_Left[j] + overlap_right[i - 1, j] for j in range(self.num_modules)]

            UnMerge_Mem_Right = [overlap_right[i, j] for j in range(self.num_modules)]
            if i + 1 < self.num_devices:
                UnMerge_Mem_Right = [UnMerge_Mem_Right[j] + overlap_left[i + 1, j] for j in range(self.num_modules)]

            total_Mem = [UnMerge_Mem_Left[j] + Merge_Mem[j] + UnMerge_Mem_Right[j] for j in range(self.num_modules)]

            return total_Mem

        u = {}
        for i in range(self.num_devices):
            for j in range(self.num_modules - 1):
                u[i, j] = md.addVar(vtype=GRB.BINARY, name=f"u_{i}_{j}")

        md.update()

        def objective_mem_function(x, y):
            overlap_M = 0
            for j in range(self.num_modules):
                overlap_M += quicksum(self.module_size[j] * x[i, j] + self.module_size[j] * y[i, j] for i in range(self.num_devices))
            return overlap_M

        md.setObjective(objective_mem_function(overlap_left, overlap_right), gp.GRB.MAXIMIZE)

        # Sum of each column is 1
        for j in range(self.num_modules):
            md.addConstr(quicksum([overlap_left[i, j] + overlap_right[i, j] for i in range(self.num_devices)]) <= 1, f"col_sum_{j}")

            # Constraint to set u[i][j]
        for i in range(self.num_devices):
            tmp = totalMemoryUsage(i, self.Solu, overlap_left, overlap_right)
            for j in range(self.num_modules - 1):
                md.addConstr(u[i, j] >= tmp[j] - tmp[j + 1], f"set_u1_{i}_{j}")
                md.addConstr(u[i, j] <= tmp[j], f"set_u2_{i}_{j}")
                md.addConstr(u[i, j] <= 1 - tmp[j + 1], f"set_u3_{i}_{j}")

        # Constraint to ensure no non-consecutive 1's after w[i][j] becomes 1
        for i in range(self.num_devices):
            tmp = totalMemoryUsage(i, self.Solu, overlap_left, overlap_right)
            for j in range(self.num_modules - 1):
                for k in range(j + 1, self.num_modules):
                    md.addConstr(tmp[k] <= 1 - u[i, j], f"no_non_consecutive_ones_row_{i}_self.num_modules_{j}_{k}")

        for i in range(self.num_devices):
            #     md.addConstr(quicksum([S[j] * (Solu[i][j] + overlap_left[i, j] + overlap_right[i, j]) for j in range(self.num_modules)]) <= AvailMem[i],  f"device_{i}_out_of_the_memory")
            md.addConstr(
                quicksum(np.array(self.module_size) * totalMemoryUsage(i, self.Solu, overlap_left, overlap_right)) <= self.ava_mem[i],
                f"device_{i}_out_of_the_memory")

        md.optimize()

        # Print results
        if md.status == gp.GRB.OPTIMAL:
            print("\nOptimal solution found.")
            for i in range(self.num_devices):
                print()
                for j in range(self.num_modules):
                    if isinstance(overlap_left[i, j], gp.Var):
                        print(f"{int(overlap_left[i, j].x)}", end=" ")
                    else:
                        print(f"{int(overlap_left[i, j])}", end=" ")
            print("\n")
            for i in range(self.num_devices):
                print()
                for j in range(self.num_modules):
                    if isinstance(overlap_right[i, j], gp.Var):
                        print(f"{int(overlap_right[i, j].x)}", end=" ")
                    else:
                        print(f"{int(overlap_right[i, j])}", end=" ")
            print("\n")
            print("Objective value =", md.objVal)
        else:
            print("Optimization did not converge to an optimal solution.")

        Strategy = []
        for i in range(self.num_devices):
            print()
            tmp = []
            for j in range(self.num_modules):
                old = self.Solu[i][j]
                if isinstance(overlap_right[i, j], gp.Var):
                    right = int(overlap_right[i, j].x)
                else:
                    right = int(overlap_right[i, j])
                if isinstance(overlap_left[i, j], gp.Var):
                    left = int(overlap_left[i, j].x)
                else:
                    left = int(overlap_left[i, j])
                print(f"{old + right + left}", end=" ")
                tmp.append(old + right + left)
            Strategy.append(tmp)

        print()
        for i in range(self.num_devices):
            print(f"Device {i} Memory Usage {quicksum([self.module_size[j] * Strategy[i][j] for j in range(self.num_modules)])}")

        self.Strategy = Strategy
        return Strategy








