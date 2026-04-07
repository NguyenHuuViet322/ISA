#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include "Instance.h"
#include <vector>
#include <numeric>
#include "Operators.h"
#include <ctime>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <deque>

class Solver
{
public:
    std::mt19937 rng{std::random_device{}()};
    Instance instance;
    Operators ops;
    std::vector<int> X;
    std::vector<int> cost;
    int upper_bound;
    int lower_bound;
    int bound;

    double block_parametter = 0.4, cooling_rate = 0.999, adaptation_rate = 0.15;
    double phi1_score_factor = 0.8, phi2_score_factor = 0.7, reset_threshold = 100;
    double penalty_factor = 50, t0 = 1000.0, beta = 10;
    double penalty = penalty_factor;
    bool isPrint = false;
    bool isEscape = false;

    Solver()
    {
        // runAllInstances(10);
        if (instance.readFromFile("data/T_2151.txt"))
        {
            parameterTuning();
        }
        else
        {
            std::cerr << "Không thể đọc dữ liệu instance!\n";
        }
    }

void init()
{
    double cost = totalEnergyConsumption();
    long long iter = 1;
    while (true)
    {
        long long theta = instance.mach;
        while (instance.proc_time[iter] * (instance.unit_cost[theta - 1] - instance.unit_cost[0]) > (bound - cost) && theta > 1)
        {
            --theta;
        }

        if (theta == 1)
            break;

        X[iter] = theta;
        cost = totalEnergyConsumption();
        redistrubutionBasedOnCost();
        iter++;

        if (iter > instance.job)
            break;
    }
    std::cout << "Initial solution cost: " << totalCompletionTime() << "\n";
}

    void runAllInstances(int numRuns)
    {
        std::ofstream outFile("batch_results5.csv");

        if (!outFile.is_open())
        {
            std::cerr << "Cannot open output file!\n";
            return;
        }

        outFile << "Instance,Run,Objective,Runtime\n";

        for (int fileIdx = 1; fileIdx <= 2160; ++fileIdx)
        {
            // if (fileIdx >= 1600 && fileIdx <= 1800)
            //     continue;

            std::stringstream ss;
            ss << "data/T_" << fileIdx << ".txt";
            std::string filename = ss.str();

            std::cout << "\n=== Running " << filename << " ===\n";

            if (!instance.readFromFile(filename))
            {
                std::cout << "Cannot read file " << filename << "\n";
                continue;
            }

            for (int run = 1; run <= numRuns; ++run)
            {
                // reset solution
                X.clear();
                X.resize(instance.job + 1, 1);
                cost.clear();
                cost.resize(instance.mach + 1, 0);

                upper_bound = calculateUpper();
                lower_bound = calculateLower();
                bound = (1 - instance.ctrl_factor) * lower_bound +
                        instance.ctrl_factor * upper_bound;

                ops.resetWeights();
                init();
                t0 = 0.2 * fitnessFunction() / log(2);

                auto start_time = std::chrono::high_resolution_clock::now();

                ISA();

                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end_time - start_time;

                double obj = totalCompletionTime();

                std::cout << "Run " << run
                          << " | Obj = " << obj
                          << " | Time = " << elapsed.count() << " sec\n";

                outFile << fileIdx << ","
                        << run << ","
                        << obj << ","
                        << elapsed.count() << "\n";

                outFile.flush();
            }
        }

        outFile.close();
        std::cout << "\n✅ Batch results saved to batch_results.csv\n";
    }

    void ISA()
    {
        std::vector<int> Xb = X;
        double bestFitness = fitnessFunction();
        const int AR_WINDOW = 300;
        std::deque<int> acceptWindow;

        double T = t0;
        int stagnation = 0;
        int maxInnerIter = std::ceil((double)instance.job / instance.mach);
        int blockSize = std::max(2, (int)std::ceil(block_parametter * instance.job / instance.mach));

        double phi1 = phi1_score_factor;
        double phi2 = phi2_score_factor;
        int iteration_count = 0;
        std::vector<int> bestImprovCount(6, 0);  // thêm vào ISA
        std::vector<int> localImprovCount(6, 0); // cải thiện so với current
        std::vector<int> acceptCount(6, 0);      // accepted (kể cả SA accept)
        std::vector<int> usageCount(6, 0);

        while (T > 1e-5)
        {
            for (int iter = 0; iter < maxInnerIter; ++iter)
            {
                iteration_count++;
                int op = ops.selectOperator();
                std::vector<int> X_old = X;
                double oldFitness = fitnessFunction();
                ops.apply(op, X, instance.mach, blockSize, instance.proc_time);
                redistrubutionBasedOnCost();
                usageCount[op]++;

                double newFitness = fitnessFunction();

                bool accepted = false;
                std::normal_distribution<double> ndist(0.0, 0.05 * T);
                double noise = ndist(rng);

                // double delta_eff = (newFitness - oldFitness) + noise;
                // double prob = std::exp(-delta_eff / T);

                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double prob = std::exp(-(newFitness - oldFitness) / T);
                double r = dist(rng);

                if (newFitness < bestFitness)
                {
                    Xb = X;
                    bestFitness = newFitness;
                    stagnation = 0;
                    accepted = true;
                    ops.reward(op, beta);                    // +β
                    bestImprovCount[op]++;
                }
                else if (newFitness < oldFitness)
                {
                    accepted = true;
                    stagnation++;
                    ops.reward(op, phi1_score_factor * beta); // +φ1×β
                    localImprovCount[op]++;
                }
                else if (r < prob)
                {
                    accepted = true;
                    stagnation++;
                    ops.reward(op, phi2_score_factor * beta); // +φ2×β
                    acceptCount[op]++;
                }
                else
                {
                    X = X_old;
                }

                if (stagnation >= reset_threshold)
                {
                    ops.resetWeights();
                    stagnation = 0;
                }
            }
            if (isPrint)
            {
                std::cout << "Iteration " << iteration_count
                          << ", Current Fitness: " << fitnessFunction()
                          << ", Best Fitness: " << bestFitness
                          << ", Cooling_rate = " << cooling_rate
                          << ", Temperature: " << T << "\n";
            }

            ops.updateWeights(adaptation_rate);

            T *= cooling_rate;
        }

        X = Xb;

        std::cout << "\n=== Operator Analysis ===\n";
        std::cout << std::setw(5) << "Op"
                  << std::setw(10) << "Usage"
                  << std::setw(12) << "BestImprov"
                  << std::setw(12) << "LocalImprov"
                  << std::setw(10) << "Accept"
                  << std::setw(12) << "BestRate%"
                  << std::setw(12) << "LocalRate%"
                  << "\n";
        for (int i = 0; i < 6; ++i)
        {
            double bestRate = 100.0 * bestImprovCount[i] / std::max(1, usageCount[i]);
            double localRate = 100.0 * localImprovCount[i] / std::max(1, usageCount[i]);
            std::cout << std::setw(5) << i
                      << std::setw(10) << usageCount[i]
                      << std::setw(12) << bestImprovCount[i]
                      << std::setw(12) << localImprovCount[i]
                      << std::setw(10) << acceptCount[i]
                      << std::setw(11) << std::fixed << std::setprecision(2) << bestRate << "%"
                      << std::setw(11) << localRate << "%"
                      << "\n";
        }
        repair();
        std::cout << "Total Completion Time: " << totalCompletionTime() << "\n";
        // std::cout << "Total Energy Consumption: " << totalEnergyConsumption() << "\n";
    }

    void repair()
    {
        if (totalEnergyConsumption() <= bound)
            return;

        std::cout << "Repairing solution to meet energy bound...\n";

        std::vector<std::vector<int>> machine_jobs(instance.mach + 1);

        while (totalEnergyConsumption() > bound)
        {
            for (int m = 0; m <= instance.mach; ++m)
                machine_jobs[m].clear();

            for (int i = 0; i <= instance.job; ++i)
                machine_jobs[X[i]].push_back(i);

            int gamma = 0;
            for (int m = instance.mach; m >= 1; --m)
            {
                if (!machine_jobs[m].empty())
                {
                    gamma = m;
                    break;
                }
            }
            if (gamma == 0)
                break; // không còn job nào để di chuyển

            int s = 1;
            while (true)
            {
                int target_machine = gamma - s;
                if (target_machine < 0 || s > gamma)
                    break;

                if (!machine_jobs[gamma].empty())
                {
                    int first_job = machine_jobs[gamma][0]; // job đầu tiên trên máy γ
                    X[first_job] = target_machine;

                    machine_jobs[target_machine].push_back(first_job);
                    machine_jobs[gamma].erase(machine_jobs[gamma].begin());
                }

                s++;
            }

            for (int m = 1; m <= instance.mach; ++m)
            {
                std::sort(machine_jobs[m].begin(), machine_jobs[m].end(),
                          [&](int a, int b)
                          { return instance.proc_time[a] < instance.proc_time[b]; });
            }

            std::cout << "Current Energy Consumption: " << totalEnergyConsumption() << "\n";
            std::cout << "Current Total Completion Time: " << totalCompletionTime() << "\n";
            std::cout << "bound: " << bound << "\n";
        }
    }

    void redistrubutionBasedOnCost()
    {
        // Tính load từng máy
        std::vector<long long> machineLoad(instance.mach + 1, 0);
        for (int j = 1; j <= instance.job; ++j)
            machineLoad[X[j]] += instance.proc_time[j];

        // Swap job giữa 2 máy nếu vi phạm Lemma 4.1
        // lp <= lq => Cp_max >= Cq_max
        // tức là máy rẻ hơn (index nhỏ hơn) phải có load >= máy đắt hơn
        bool swapped = true;
        while (swapped)
        {
            swapped = false;
            for (int p = 1; p <= instance.mach; ++p)
            {
                for (int q = p + 1; q <= instance.mach; ++q)
                {
                    // unit_cost[p-1] <= unit_cost[q-1] nhưng load[p] < load[q] → vi phạm
                    if (machineLoad[p] < machineLoad[q])
                    {
                        // Swap toàn bộ job giữa máy p và q
                        for (int j = 1; j <= instance.job; ++j)
                        {
                            if (X[j] == p) X[j] = q;
                            else if (X[j] == q) X[j] = p;
                        }
                        std::swap(machineLoad[p], machineLoad[q]);
                        swapped = true;
                    }
                }
            }
        }
    }

    long long calculateUpper()
    {
        std::vector<long long> x_tmp(instance.mach, 0);
        for (int job : instance.proc_time)
        {
            long long min_pos = std::min_element(x_tmp.begin(), x_tmp.end()) - x_tmp.begin();
            x_tmp[min_pos] += job;
        }
        long long U = 0;
        for (long long i = 0; i < instance.mach; ++i)
        {
            U += x_tmp[i] * instance.unit_cost[i];
        }

        return U;
    }

    long long calculateLower()
    {
        int U = 0;
        for (int i = 0; i <= instance.job; ++i)
        {
            U += instance.proc_time[i] * instance.unit_cost[0];
        }

        return U;
    }

    long long totalCompletionTime()
    {
        long long total = 0;

        for (int m = 1; m <= instance.mach; ++m)
        {
            long long time = 0;

            for (int j = 1; j <= instance.job; ++j)
            {
                if (X[j] == m)
                {
                    time += instance.proc_time[j];
                    total += time;
                }
            }
        }

        return total;
    }

    long long totalEnergyConsumption()
    {
        std::vector<long long> machineTime(instance.mach + 1, 0);

        for (int j = 1; j <= instance.job; ++j)
        {
            int m = X[j];
            machineTime[m] += instance.proc_time[j];
        }

        long long U = 0;
        for (int m = 1; m <= instance.mach; ++m)
        {
            U += machineTime[m] * instance.unit_cost[m - 1];
        }

        return U;
    }

    double fitnessFunction()
    {
        long long TCT = totalCompletionTime();
        long long TEC = totalEnergyConsumption();
        if (TEC <= bound)
        {
            return TCT;
        }
        else
        {
            return TCT + (penalty_factor * (-bound + TEC) * (-bound + TEC));
        }
    }

    void printSolution()
    {
        for (int i = 1; i <= instance.mach; ++i)
        {
            std::cout << "Machine " << i << ": ";
            for (int j = 1; j < instance.job + 1; ++j)
            {
                if (X[j] == i)
                {
                    std::cout << "Job " << j << " (Proc time: " << instance.proc_time[j] << "), ";
                }
            }
            std::cout << "\n";
        }
    }

    struct ParameterSet
    {
        double cooling_rate;
        double adaptation_rate;
        double block_param;
        int reset_threshold;
        double phi1;
        double phi2;
        double penalty_factor;
        double t0_factor;
        bool isPrint;
        double beta;
    };

    void parameterTuning()
    {
        std::vector<ParameterSet> params = {
            {0.999, 0.15, 0.4, 300, 0.8, 0.7, 50, 0.2, true, 10},

            // {0.999, 0.15, 0.4, 100, 0.8, 0.7, 50, 0.2, false, false, false, false,3.0, 0.2, false},
            // {0.999, 0.15, 0.4, 100, 0.4, 0.3, 50, 0.2, false, false, true, false,3.0, 0.2, false},
            // {0.999, 0.15, 0.4, 100, 0.8, 0.7, 50, 0.2, false, false, false, false,3.0, 0.2, false},

            // {0.999, 0.15, 0.4, 100, 0.8, 0.7, 50, 0.2, true, false, 0, 0, false},
        };

        std::ofstream outFile("parameter_tuning_results.csv");

        if (!outFile.is_open())
        {
            std::cerr << "Cannot open output file!\n";
            return;
        }

        outFile << "cooling_rate,adaptation_rate,block_param,reset_threshold,"
                << "phi1,phi2,penalty_factor,t0_factor,best,worst,avg,stddev\n";

        std::cout << "Starting parameter tuning...\n";
        std::cout << "Progress: ";

        int paramIndex = 0;
        for (const auto &p : params)
        {
            paramIndex++;
            std::cout << paramIndex << "/" << params.size() << "... ";
            std::cout.flush();

            cooling_rate = p.cooling_rate;
            adaptation_rate = p.adaptation_rate;
            block_parametter = p.block_param;
            reset_threshold = p.reset_threshold;
            phi1_score_factor = p.phi1;
            phi2_score_factor = p.phi2;
            penalty_factor = p.penalty_factor;
            isPrint = p.isPrint;
            beta = p.beta;

            std::vector<double> results;

            for (int run = 0; run < 10; ++run)
            {
                X.clear();
                X.resize(instance.job + 1, 1);
                cost.clear();
                cost.resize(instance.mach + 1, 0);

                upper_bound = calculateUpper();
                lower_bound = calculateLower();
                bound = (1 - instance.ctrl_factor) * lower_bound + instance.ctrl_factor * upper_bound;
                init();
                double size_factor = std::log(1.0 + instance.job * instance.mach);
                t0 = p.t0_factor * fitnessFunction() / log(2);

                penalty_factor = p.penalty_factor;

                auto start_time = std::chrono::high_resolution_clock::now();

                ISA();

                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end_time - start_time;
                std::cout << "Run " << run + 1 << " completed in " << elapsed.count() << " seconds.\n";
                results.push_back(totalCompletionTime());
            }

            double best = *std::min_element(results.begin(), results.end());
            double worst = *std::max_element(results.begin(), results.end());
            double avg = std::accumulate(results.begin(), results.end(), 0.0) / results.size();

            double variance = 0.0;
            for (double val : results)
                variance += (val - avg) * (val - avg);
            double stddev = std::sqrt(variance / results.size());

            outFile << std::fixed
                    << p.cooling_rate << ","
                    << p.adaptation_rate << ","
                    << p.block_param << ","
                    << p.reset_threshold << ","
                    << p.phi1 << ","
                    << p.phi2 << ","
                    << p.penalty_factor << ","
                    << p.t0_factor << ","
                    << best << ","
                    << worst << ","
                    << avg << ","
                    << stddev << "\n";

            outFile.flush();
        }

        outFile.close();
        std::cout << "\n✅ Results saved to: parameter_tuning_results.csv\n";
    }
};

#endif