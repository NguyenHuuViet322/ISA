#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include "Instance.h"
#include <vector>
#include <numeric>
#include "Operators.h"
#include <chrono>

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
    double block_parametter = 0.4, cooling_rate = 0.995, adaptation_rate = 0.15;
    double phi1_score_factor = 0.8, phi2_score_factor = 0.7, reset_threshold = 300;
    double penalty_factor = 50, t0 = 1000.0, beta = 15;
    double penalty = penalty_factor; bool isLNS = false;
    double reheat_factor; double T_cap_factor;
    bool isPrint = false; bool isReheat = false;

    Solver()
    {
        if (instance.readFromFile("data/test2.txt"))
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
        long long theta = instance.mach;
        long long iter = 1;
        while (true)
        {
            theta = instance.mach;
            while (instance.proc_time[iter] * (instance.unit_cost[theta - 1] - instance.unit_cost[1] > bound - cost) && theta > 1)
            {
                --theta;
            }

            if (theta == 1)
                break;
            else
            {
                X[iter] = theta;
                cost = totalEnergyConsumption();
                iter++;
            }
            redistrubutionBasedOnCost();

            if (iter == instance.job)
                break;
        }
    }

    void ISA()
    {
        std::vector<int> Xb = X;
        double bestFitness = fitnessFunction();

        double T = t0;
        int stagnation = 0;
        int maxInnerIter = std::ceil((double)instance.job / instance.mach);
        int blockSize = std::max(1, (int)(block_parametter * instance.job));

        double beta = 15.0;
        double phi1 = phi1_score_factor;
        double phi2 = phi2_score_factor;
        int iteration_count = 0;
        int reheat_count = 0;
        int improve_window = 0;
        int window_size = 500;
        int improve_count = 0;
        int move_count = 0;

        double T_low = 0.03 * t0;
        double T_cap = 0.2 * t0;
        int reheat_hold = 6000;
        int reheat_hold_counter = 0;
        double reheat_factor = 3;
        int max_reheat = 5;


        while (T > 1e-4)
        {
            for (int iter = 0; iter < maxInnerIter; ++iter)
            {
                double T_accept = T;
                double T_move = std::min(T, 0.05 * t0);

                iteration_count++;

                int op = ops.selectOperator();
                std::vector<int> X_old = X;
                double oldFitness = fitnessFunction();

                ops.apply(op, X, instance.mach, std::max(1, (int)(block_parametter * instance.job * (T_move  / t0))));
                double newFitness = fitnessFunction();

                bool accepted = false;
                double prob = std::exp(-(newFitness - oldFitness) / T_accept);
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double r = dist(rng);

                if (newFitness < bestFitness)
                {
                    Xb = X;
                    bestFitness = newFitness;
                    improve_count++;
                    stagnation = 0;
                    ops.reward(op, beta);
                }
                else if (newFitness < oldFitness)
                {
                    stagnation++;
                    ops.reward(op, phi1 * beta);
                }
                else if (r < prob)
                {
                    stagnation++;
                    ops.reward(op, phi2 * beta);
                }
                else
                {
                    X = X_old;
                    ops.penalize(op);
                }

                if (stagnation >= reset_threshold)
                {
                    ops.resetWeights();
                    stagnation = 0;
                }

                if (isReheat) {
                    if (iteration_count % window_size == 0)
                    {
                        if (T < T_low &&
                            improve_window >= 2 &&
                            reheat_count < max_reheat)
                        {
                            T = std::min(T * reheat_factor, T_cap);
                            reheat_hold_counter = reheat_hold;
                            reheat_count++;
                        }
                        improve_window = 0;
                    }
                }

                if (isLNS) {
                    if (T < 0.01 * t0 && stagnation > 2 * reset_threshold)
                    {
                        int k = 2 + stagnation / 100;
                        apply_LNS_destroy_repair(k);
                    }

                }
            }
            if (isPrint) {
                std::cout << "Iteration " << iteration_count
                      << ", Current Fitness: " << fitnessFunction()
                      << ", Best Fitness: " << bestFitness
                      << ", Temperature: " << T << "\n";
            }

            ops.updateWeights(adaptation_rate);
            if (reheat_hold_counter > 0)
            {
                reheat_hold_counter--;
            }
            else
            {
                T *= cooling_rate;
            }

            penalty = penalty_factor * (1.0 + T / t0);
        }

        X = Xb;
        finalLocalSearch();

        repair();
        std::cout << "Total Completion Time: " << totalCompletionTime() << "\n";
        // std::cout << "Total Energy Consumption: " << totalEnergyConsumption() << "\n";
    }

    void stochasticTailSearch(int iters = 5000)
    {
        double T = 1e-4;   // rất thấp
        for (int k = 0; k < iters; ++k)
        {
            int op = ops.selectOperator();
            auto X_old = X;
            double oldF = fitnessFunction();

            ops.apply(op, X, instance.mach, 1); // block size = 1

            double newF = fitnessFunction();
            if (newF <= oldF)
                continue;

            // rất hiếm accept xấu
            double prob = exp(-(newF - oldF) / T);
            if (rand() > prob)
                X = X_old;
            std::cout << "Tail Search Iteration " << k
                      << ", Current Fitness: " << fitnessFunction()
                      << ", Previous Fitness: " << oldF << "\n";
        }
    }


    bool local_single_relocate()
    {
        long long bestTCT = totalCompletionTime();

        for (int j = 1; j <= instance.job; ++j)
        {
            int old_m = X[j];

            for (int m = 1; m <= instance.mach; ++m)
            {
                if (m == old_m)
                    continue;

                X[j] = m;

                if (totalEnergyConsumption() <= bound)
                {
                    long long tct = totalCompletionTime();
                    if (tct < bestTCT)
                    {
                        return true; // 🔥 accept FIRST improvement
                    }
                }
            }

            X[j] = old_m; // rollback
        }

        return false;
    }

    bool local_block_relocate(int blockSize)
    {
        long long bestTCT = totalCompletionTime();

        for (int m1 = 1; m1 <= instance.mach; ++m1)
        {
            std::vector<int> idx;
            for (int j = 1; j <= instance.job; ++j)
                if (X[j] == m1)
                    idx.push_back(j);

            if ((int)idx.size() < blockSize)
                continue;

            for (int s = 0; s <= (int)idx.size() - blockSize; ++s)
            {
                for (int m2 = 1; m2 <= instance.mach; ++m2)
                {
                    if (m2 == m1)
                        continue;

                    // apply block relocate
                    for (int k = 0; k < blockSize; ++k)
                        X[idx[s + k]] = m2;

                    if (totalEnergyConsumption() <= bound)
                    {
                        long long tct = totalCompletionTime();
                        if (tct < bestTCT)
                            return true;
                    }

                    // rollback
                    for (int k = 0; k < blockSize; ++k)
                        X[idx[s + k]] = m1;
                }
            }
        }
        return false;
    }

    bool local_block_swap(int blockSize)
    {
        long long bestTCT = totalCompletionTime();

        for (int m1 = 1; m1 <= instance.mach; ++m1)
            for (int m2 = m1 + 1; m2 <= instance.mach; ++m2)
            {
                std::vector<int> a, b;
                for (int j = 1; j <= instance.job; ++j)
                {
                    if (X[j] == m1)
                        a.push_back(j);
                    else if (X[j] == m2)
                        b.push_back(j);
                }

                if ((int)a.size() < blockSize || (int)b.size() < blockSize)
                    continue;

                for (int i = 0; i <= (int)a.size() - blockSize; ++i)
                    for (int j = 0; j <= (int)b.size() - blockSize; ++j)
                    {
                        for (int k = 0; k < blockSize; ++k)
                            std::swap(X[a[i + k]], X[b[j + k]]);

                        if (totalEnergyConsumption() <= bound)
                        {
                            long long tct = totalCompletionTime();
                            if (tct < bestTCT)
                                return true;
                        }

                        for (int k = 0; k < blockSize; ++k)
                            std::swap(X[a[i + k]], X[b[j + k]]);
                    }
            }
        return false;
    }

    int findMaxLoadMachine()
    {
        std::vector<long long> load(instance.mach + 1, 0);

        for (int j = 1; j <= instance.job; ++j)
        {
            int m = X[j];
            load[m] += instance.proc_time[j];
        }

        int maxM = 1;
        for (int m = 2; m <= instance.mach; ++m)
        {
            if (load[m] > load[maxM])
                maxM = m;
        }

        return maxM;
    }

    int findMinLoadMachine()
    {
        std::vector<long long> load(instance.mach + 1, 0);

        for (int j = 1; j <= instance.job; ++j)
            load[X[j]] += instance.proc_time[j];

        int minM = 1;
        for (int m = 2; m <= instance.mach; ++m)
            if (load[m] < load[minM])
                minM = m;

        return minM;
    }

    bool local_heavy_to_light()
    {
        int m_heavy = findMaxLoadMachine();
        int m_light = findMinLoadMachine();

        long long bestTCT = totalCompletionTime();

        for (int j = 1; j <= instance.job; ++j)
        {
            if (X[j] != m_heavy)
                continue;

            X[j] = m_light;

            if (totalEnergyConsumption() <= bound)
            {
                if (totalCompletionTime() < bestTCT)
                    return true;
            }

            X[j] = m_heavy;
        }
        return false;
    }

    bool shallow_local_refine()
    {
        long long baseTCT = totalCompletionTime();

        // tìm job có proc_time lớn
        int target = -1;
        long long maxp = -1;
        for (int j = 1; j <= instance.job; ++j)
        {
            if (instance.proc_time[j] > maxp)
            {
                maxp = instance.proc_time[j];
                target = j;
            }
        }

        int old_m = X[target];

        for (int m = 1; m <= instance.mach; ++m)
        {
            if (m == old_m)
                continue;

            X[target] = m;

            if (totalEnergyConsumption() <= bound &&
                totalCompletionTime() < baseTCT)
            {
                return true; // accept 1 bước là đủ
            }
        }

        X[target] = old_m;
        return false;
    }

    bool local_two_job_swap()
    {
        long long bestTCT = totalCompletionTime();

        for (int j1 = 1; j1 <= instance.job; ++j1)
        {
            for (int j2 = j1 + 1; j2 <= instance.job; ++j2)
            {
                if (X[j1] == X[j2])
                    continue;

                std::swap(X[j1], X[j2]);

                if (totalEnergyConsumption() <= bound)
                {
                    long long tct = totalCompletionTime();
                    if (tct < bestTCT)
                    {
                        return true;
                    }
                }

                std::swap(X[j1], X[j2]); // rollback
            }
        }

        return false;
    }

    void apply_LNS_destroy_repair(int destroy_k)
{
    int n = instance.job;
    int m = instance.mach;

    /* =========================
       1. Compute load per machine
       ========================= */
    std::vector<long long> load(m + 1, 0);
    for (int i = 1; i <= n; ++i)
    {
        int mach = X[i];
        if (mach > 0)
            load[mach] += instance.proc_time[i];
    }

    /* =========================
       2. Select destroy candidates
       job already sorted by proc_time
       → take from tail
       ========================= */
    int candidate_start = std::max(1, n - 3 * destroy_k);
    std::vector<int> candidates;
    for (int i = candidate_start; i <= n; ++i)
        candidates.push_back(i);

    // Prefer jobs on heavily loaded machines
    std::sort(candidates.begin(), candidates.end(),
              [&](int a, int b)
              {
                  return load[X[a]] > load[X[b]];
              });

    destroy_k = std::min(destroy_k, (int)candidates.size());

    /* =========================
       3. Destroy
       ========================= */
    std::vector<int> removed_jobs;
    removed_jobs.reserve(destroy_k);

    for (int i = 0; i < destroy_k; ++i)
    {
        int job = candidates[i];
        removed_jobs.push_back(job);
        X[job] = 0; // remove assignment
    }

    /* =========================
       4. Repair (stochastic greedy)
       ========================= */
    std::uniform_real_distribution<double> rand01(0.0, 1.0);

    for (int job : removed_jobs)
    {
        std::vector<std::pair<long long, int>> options;

        for (int mach = 1; mach <= m; ++mach)
        {
            X[job] = mach;

            // energy constraint
            if (totalEnergyConsumption() > bound)
                continue;

            long long tct = totalCompletionTime();
            options.emplace_back(tct, mach);
        }

        // No feasible machine → fallback
        if (options.empty())
        {
            X[job] = 1;
            continue;
        }

        std::sort(options.begin(), options.end());

        // stochastic selection in top-k
        int k = std::min(3, (int)options.size());
        double r = rand01(rng);
        int chosen_idx = 0;

        if (r < 0.6)
            chosen_idx = 0;
        else if (r < 0.85 && k > 1)
            chosen_idx = 1;
        else if (k > 2)
            chosen_idx = 2;

        X[job] = options[chosen_idx].second;
    }
}


    void repair()
    {
        if (totalEnergyConsumption() <= bound)
            return;

        std::cout << "Repairing solution to meet energy bound...\n";

        // Lưu job theo từng máy
        std::vector<std::vector<int>> machine_jobs(instance.mach + 1);

        while (totalEnergyConsumption() > bound)
        {
            // Clear máy trước khi build lại
            for (int m = 0; m <= instance.mach; ++m)
                machine_jobs[m].clear();

            // Đưa các job vào từng máy theo X
            for (int i = 0; i <= instance.job; ++i)
                machine_jobs[X[i]].push_back(i);

            // Tìm gamma: máy cao nhất còn job
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

            // Xử lý s = 1 và tăng dần theo pseudocode
            int s = 1;
            while (true)
            {
                int target_machine = gamma - s;
                if (target_machine < 0 || s > gamma)
                    break;

                if (!machine_jobs[gamma].empty())
                {
                    int first_job = machine_jobs[gamma][0]; // job đầu tiên trên máy γ
                    // Di chuyển job xuống máy γ - s
                    X[first_job] = target_machine;

                    // Cập nhật machine_jobs
                    machine_jobs[target_machine].push_back(first_job);
                    machine_jobs[gamma].erase(machine_jobs[gamma].begin());
                }

                s++;
            }

            // Sắp xếp các job trên từng máy theo SPT
            for (int m = 1; m <= instance.mach; ++m)
            {
                std::sort(machine_jobs[m].begin(), machine_jobs[m].end(),
                          [&](int a, int b)
                          { return instance.proc_time[a] < instance.proc_time[b]; });
            }

            // Cập nhật chi phí
            std::cout << "Current Energy Consumption: " << totalEnergyConsumption() << "\n";
            std::cout << "Current Total Completion Time: " << totalCompletionTime() << "\n";
            std::cout << "bound: " << bound << "\n";
        }
    }

    void tailSwapLocalSearch(int max_pass = 3)
{
    for (int pass = 0; pass < max_pass; ++pass)
    {
        bool improved = false;

        for (int i = instance.job; i >= 1; --i)
        {
            for (int j = i - 1; j >= std::max(1, i - 10); --j)
            {
                if (X[i] == X[j]) continue;

                int mi = X[i], mj = X[j];

                X[i] = mj;
                X[j] = mi;

                if (totalEnergyConsumption() <= bound &&
                    totalCompletionTime() < fitnessFunction())
                {   
                    std::cout << "Tail Swap Improvement: "
                              << "Jobs " << i << " and " << j
                              << " swapped between machines " << mi << " and " << mj << "\n";
                    improved = true;
                    break;
                }

                X[i] = mi;
                X[j] = mj;
            }
            if (improved) break;
        }

        if (!improved) break;
    }
}



    void redistrubutionBasedOnCost()
    {
        std::vector<int> cost_tmp(instance.mach + 1, 0);
        std::vector<std::vector<int>> machine_jobs(instance.mach + 1);

        for (int i = 0; i <= instance.job; ++i)
        {
            cost_tmp[X[i]] += instance.proc_time[i];
            machine_jobs[X[i]].push_back(i);
        }

        std::vector<int> idx(instance.mach + 1);
        std::iota(idx.begin(), idx.end(), 0);

        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b)
                  { return cost_tmp[a] > cost_tmp[b]; });

        std::vector<int> new_cost(instance.mach + 1);
        std::vector<std::vector<int>> new_jobs(instance.mach + 1);

        for (int k = 0; k <= instance.mach; ++k)
        {
            new_cost[k] = cost_tmp[idx[k]];
            new_jobs[k] = machine_jobs[idx[k]];
        }

        cost_tmp.swap(new_cost);
        machine_jobs.swap(new_jobs);

        for (int machine = 0; machine < instance.mach; ++machine)
        {
            for (int job : machine_jobs[machine])
            {
                X[job] = machine + 1;
            }
        }
    }

    void finalLocalSearch()
    {
        while (true)
        {
            if (local_single_relocate())
            {
                std::cout << "Local search - Current fitness: " << fitnessFunction() << "\n";
                continue;
            }
            if (local_two_job_swap())
            {
                std::cout << "Local search - Current fitness: " << fitnessFunction() << "\n";
                continue;
            }

            if (local_block_relocate(2))
            {
                std::cout << "Local search - Current fitness: " << fitnessFunction() << "\n";
                continue;
            }
            if (local_block_swap(2))
            {
                std::cout << "Local search - Current fitness: " << fitnessFunction() << "\n";
                continue;
            }

            if (local_heavy_to_light())
            {
                std::cout << "Local search - Current fitness: " << fitnessFunction() << "\n";
                continue;
            }
            std::cout << "Local search - Current fitness: " << fitnessFunction() << "\n";
            break;
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

    bool accept_LNS(double fitness_old, double fitness_new, double T)
    {
        double delta = fitness_new - fitness_old;
        if (delta < 0 || rand() < exp(-delta / T))
            return true;

        return false;
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
        bool isLNS;
        bool isReheat;
        double reheat_factor;
        double t_cap_factor;
        bool isPrint;
    };

    void parameterTuning()
    {
        std::vector<ParameterSet> params = {
            {0.9999, 0.15, 0.4, 100, 0.8, 0.7, 50, 0.2, false, false, 3.0, 0.2, true},
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
            isLNS = p.isLNS;
            isReheat = p.isReheat;
            isPrint = p.isPrint;

            std::vector<double> results;

            for (int run = 0; run < 1; ++run)
            {
                X.clear();
                X.resize(instance.job + 1, 1);
                cost.clear();
                cost.resize(instance.mach + 1, 0);

                upper_bound = calculateUpper();
                lower_bound = calculateLower();
                bound = (1 - instance.ctrl_factor) * lower_bound + instance.ctrl_factor * upper_bound;
                init();
                t0 = p.t0_factor * totalCompletionTime() / log(2);

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