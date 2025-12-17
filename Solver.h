#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include "Instance.h"
#include <vector>
#include <numeric>
#include "Operators.h"

class Solver {
public:
    Instance instance;
    Operators ops;
    std::vector<int> X; // X[i] có nghĩ là việc i được xử lý bằng máy x[i]
    std::vector<int> cost;
    int upper_bound;
    int lower_bound;
    int bound;
    double block_parametter = 0.1, cooling_rate = 0.995, adaptation_rate = 0.15; 
    double phi1_score_factor = 0.8, phi2_score_factor = 0.7, reset_threshold = 300; 
    double penalty_factor = 50, t0 = 1000.0;

    Solver() {
        if (instance.readFromFile("data/T_3.txt")) {
            X.resize(instance.job+1, 1);
            cost.resize(instance.mach+1, 0);
            instance.print();
            upper_bound = calculateUpper();
            lower_bound = calculateLower();
            bound = (1 - instance.ctrl_factor) * upper_bound + instance.ctrl_factor * lower_bound;
            std::cout << "Upper Bound: " << upper_bound << "\n";
            std::cout << "Lower Bound: " << lower_bound << "\n";
            std::cout << "Bound: " << bound << "\n";
            std::cout << "Cost: " << totalEnergyConsumption() << "\n";
            init();
            t0 = 0.2*totalCompletionTime()/log(2);
            printSolution();
        } else {
            std::cerr << "Không thể đọc dữ liệu instance!\n";
        }
    }

    void init() {
        double cost = totalEnergyConsumption();
        std::cout << "X size: " << X.size() << "\n";
        long long theta = instance.mach;
        long long iter = 1;
        while(true) {
            std::cout << iter << "\n";
            theta = instance.mach;
            while(instance.proc_time[iter]*(instance.unit_cost[theta-1]-instance.unit_cost[1] > bound-cost) && theta > 1) {
                --theta;
            }

            if (theta == 1) break;
            else {
                X[iter] = theta;
                cost = totalEnergyConsumption();
                iter++;
            }
            printSolution();
            std::cout << "After" << std::endl;
            redistrubutionBasedOnCost();


            if(iter == instance.job) break;
        }
    }

    void ISA() {

    }

    void repair() {
        if (totalEnergyConsumption() <= bound) return;

        std::vector<std::vector<int>> machine_jobs(instance.mach+1);
        while(totalEnergyConsumption() > bound) {
            int gamma = -1;
        
            for (int i = 0; i <= instance.job; ++i) {
                machine_jobs[X[i]].push_back(i);
            }
            for (int i = instance.mach; i > 0; --i) {
                if (!machine_jobs[i].empty()) {
                    gamma = i;
                    break;
                }
            }

            if (gamma == 0) break;
            int first_job = machine_jobs[gamma][0];
            X[first_job] = gamma - 1;

            for (int m = 1; m <= instance.mach; ++m) {
                std::sort(machine_jobs[m].begin(), machine_jobs[m].end(),
                        [&](int a, int b) { return instance.proc_time[a] < instance.proc_time[b]; });
            }
        }
    }

    void redistrubutionBasedOnCost() {
        std::vector<int> cost_tmp(instance.mach+1, 0);
        std::vector<std::vector<int>> machine_jobs(instance.mach+1);

        for (int i = 0; i <= instance.job; ++i) {
            cost_tmp[X[i]] += instance.proc_time[i];
            machine_jobs[X[i]].push_back(i);
        }

        std::vector<int> idx(instance.mach + 1);
        std::iota(idx.begin(), idx.end(), 0); 

        std::sort(idx.begin(), idx.end(),
            [&](int a, int b) { return cost_tmp[a] > cost_tmp[b]; });

        std::vector<int> new_cost(instance.mach + 1);
        std::vector<std::vector<int>> new_jobs(instance.mach + 1);

        for (int k = 0; k <= instance.mach; ++k) {
            new_cost[k] = cost_tmp[idx[k]];
            new_jobs[k] = machine_jobs[idx[k]];
        }

        cost_tmp.swap(new_cost);
        machine_jobs.swap(new_jobs);

        for(int machine = 0; machine < instance.mach; ++machine) {
            for(int job : machine_jobs[machine]) {
                X[job] = machine+1;
            }
        }
    }

    long long calculateUpper() {
        std::vector<long long> x_tmp(instance.mach, 0);
        for (int job : instance.proc_time) {
            long long min_pos = std::min_element(x_tmp.begin(), x_tmp.end()) - x_tmp.begin();
            x_tmp[min_pos] += job;
        }
        long long U = 0;
        for(long long i = 0; i < instance.mach; ++i) {
            U += x_tmp[i] * instance.unit_cost[i];
        }

        return U;
    }

    long long calculateLower() {
        int U = 0;
        for (int i = 0; i <= instance.job; ++i) {
            U += instance.proc_time[i] * instance.unit_cost[0];
        }

        return U;
    }

    long long totalCompletionTime() {
        std::vector<long long> machineFinish(instance.mach, 0);
        long long total = 0;

        for (int i = 0; i < instance.job; ++i) {
            int m = X[i] - 1; 
            machineFinish[m] += instance.proc_time[i];
            total += machineFinish[m];
        }

        return total;
    }

    long long totalEnergyConsumption() {
        long long total = 0;
        for (int i = 0; i < X.size(); ++i) {
            total += instance.unit_cost[X[i]-1]*instance.proc_time[i];
        }
        return total;
    }

    double fitnessFunction() {
        long long TCT = totalCompletionTime();
        long long TEC = totalEnergyConsumption();
        
        if (TEC <= bound) {
            return TCT;
        } else {
            return TCT + (penalty_factor * (bound - TEC) * (bound - TEC));
        }
    }

    void printSolution() {
        for(int i =1; i <= instance.mach; ++i) {
            std::cout << "Machine " << i << ": ";
            for(int j = 1; j < instance.job+1; ++j) {
                if(X[j] == i) {
                    std::cout << "Job " << j << " (Proc time: " << instance.proc_time[j] << "), ";
                }
            }
            std::cout << "\n";
        }
    }
    
};

#endif 
