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
    double block_parametter = 0.4, cooling_rate = 0.995, adaptation_rate = 0.15;
    double phi1_score_factor = 0.8, phi2_score_factor = 0.7, reset_threshold = 300;
    double penalty_factor = 50, t0 = 1000.0, beta = 15;
    double penalty = penalty_factor; bool isLNS = false; bool isAdaptivePenalty = false; bool isAdaptiveReward = false;
    double reheat_factor; double T_cap_factor;
    bool isPrint = false; bool isEscape = false;
    int escape_fail_count = 0;
    int escape_phase = 0;

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

    std::string makeLogFilename()
    {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);

        std::tm tm{};
    #ifdef _WIN32
        localtime_s(&tm, &t);
    #else
        localtime_r(&t, &tm);
    #endif

        std::ostringstream oss;
        oss << "log/isa_log_"
            << std::put_time(&tm, "%Y%m%d_%H%M%S")
            << ".csv";

        return oss.str();
    }

void adaptiveEscape(double T)
{
    int n = instance.job;
    int m = instance.mach;

    int strength = std::min(
        n / 3,
        3 + escape_fail_count * 2
    );

    std::uniform_int_distribution<int> randJob(1, n);
    std::uniform_int_distribution<int> randMach(1, m);

    if (escape_phase == 0)
    {
        // ===== PHASE 0: RANDOM REASSIGN (PHÁ MẠNH) =====
        for (int i = 0; i < strength; ++i)
        {
            int j = randJob(rng);
            X[j] = randMach(rng);
        }
    }
    else if (escape_phase == 1)
    {
        // ===== PHASE 1: SWAP JOBS BETWEEN MACHINES =====
        for (int i = 0; i < strength; ++i)
        {
            int j1 = randJob(rng);
            int j2 = randJob(rng);
            std::swap(X[j1], X[j2]);
        }
    }
    else
    {
        // ===== PHASE 2: SHUFFLE SUBSET =====
        std::vector<int> subset;
        for (int i = 0; i < strength * 2; ++i)
            subset.push_back(randJob(rng));

        std::shuffle(subset.begin(), subset.end(), rng);

        for (int i = 0; i < (int)subset.size() - 1; ++i)
            X[subset[i]] = X[subset[i + 1]];
    }
}


void midTemperatureEscape(double &T, int maxInnerIter)
{
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    auto uniform01 = [&](std::mt19937 &rng) {
        return dist01(rng);
    };
    double savedT = T;
    T *= 1.5;                 // reheat nhẹ
    int escapeIter = maxInnerIter / 2;

    for (int i = 0; i < escapeIter; ++i)
    {
        int op = ops.selectOperator();
        std::vector<int> X_old = X;
        double oldF = fitnessFunction();

        ops.apply(op, X, instance.mach,
                  std::max(1, (int)(0.2 * instance.job)));

        double newF = fitnessFunction();
        double prob = std::exp(-(newF - oldF) / T);

        if (newF < oldF || uniform01(rng) < prob)
            ops.reward(op, beta);
        else
            X = X_old;
    }

    T = savedT;
}


    void ISA()
    {
        std::ofstream logFile(makeLogFilename());
        logFile << "iter,type,temperature,oldFitness,newFitness,bestFitness,delta,improveTotal,stagnation\n";

        std::vector<int> Xb = X;
        double bestFitness = fitnessFunction();
        const int AR_WINDOW = 300;
        std::deque<int> acceptWindow;

        double T = t0;
        int stagnation = 0;
        int maxInnerIter = std::ceil((double)instance.job / instance.mach);
        int blockSize = std::max(1, (int)(block_parametter * instance.job));

        double beta = 15.0;
        double phi1 = phi1_score_factor;
        double phi2 = phi2_score_factor;
        int iteration_count = 0;
        int reheat_count = 0;
        int improve_total = 0;
        int window_size = 500;
        int acceptedMoves = 0;
        int totalMoves = 0;
        int noImproveIter = 0;
        int maxReheat = 5;
        double tau = T / t0;

        while (T > 1e-4)
        {
            for (int iter = 0; iter < maxInnerIter; ++iter)
            {
                iteration_count++;
                tau = T / t0;
                int op = ops.selectOperator();
                std::vector<int> X_old = X;
                double oldFitness = fitnessFunction();

                //ops.apply(op, X, instance.mach, blockSize);


                ops.apply(op, X, instance.mach, std::max(1, (int)(block_parametter * instance.job * (T  / t0))));
                totalMoves++;

                double newFitness = fitnessFunction();

                bool accepted = false;
                std::normal_distribution<double> ndist(0.0, 0.05 * T);
                double noise = ndist(rng);

                double delta_eff = (newFitness - oldFitness) + noise;
                double prob = std::exp(-delta_eff / T);

                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double r = dist(rng);

                if (newFitness < bestFitness)
                {
                    Xb = X;
                    bestFitness = newFitness;
                    stagnation = 0;
                    noImproveIter = 0;
                    
                    accepted = true;

                    if(isAdaptiveReward) {
                        double delta = oldFitness - newFitness;
                        double norm = delta / (oldFitness + 1e-9);
                        ops.reward(op, beta * std::min(norm, 0.01));
                    } else 
                        ops.reward(op, beta);

                    improve_total++;


                    logFile  << "[IMPROVE] "
                        << "Iter=" << iteration_count
                        << " | T=" << T
                        << " | NewBest=" << bestFitness
                        << " | Δ=" << (oldFitness - newFitness)
                        << " | ImproveTotal=" << improve_total
                        << "\n";
                }
                else if (newFitness < oldFitness)
                {
                    accepted = true;
                    stagnation++;
                    noImproveIter++;
                    
                    ops.reward(op, phi1 * beta);
                }
                else if (r < prob)
                {
                    accepted = true;
                    stagnation++;
                    noImproveIter++;
                    
                    ops.reward(op, phi2 * beta);
                }
                else
                {                    
                    X = X_old;
                    double factor = 1.0;
                    if (T > 0.2 * t0)
                        factor = 1.0 + (1.0 - T / t0);

                    noImproveIter++;
                    if (isAdaptivePenalty) {
                        ops.penalize(op, factor);
                    } else
                        ops.penalize(op, 1.0);
                }

                if (accepted)
                    acceptWindow.push_back(1);
                else
                    acceptWindow.push_back(0);

                if ((int)acceptWindow.size() > AR_WINDOW)
                    acceptWindow.pop_front();


                double accept_ratio = 0.0;
                if (!acceptWindow.empty())
                {
                    int sum = std::accumulate(acceptWindow.begin(), acceptWindow.end(), 0);
                    accept_ratio = (double)sum / acceptWindow.size();
                }


                if (stagnation >= reset_threshold)
                {
                    ops.resetWeights();
                    stagnation = 0;
                }
            }
            if (isPrint) {
                std::cout << "Iteration " << iteration_count
                      << ", Current Fitness: " << fitnessFunction()
                      << ", Best Fitness: " << bestFitness
                      << ", Cooling_rate = " << cooling_rate
                      << ", Temperature: " << T << "\n";
            }

            double accept_ratio = (double)acceptedMoves / totalMoves;
            acceptedMoves = 0;
            totalMoves = 0;


            ops.updateWeights(adaptation_rate);
            
            T *= cooling_rate;
        }

        int exploitNoImprove = 0;
        int EXPLOIT_LIMIT = 3 * instance.job;

        while (exploitNoImprove < EXPLOIT_LIMIT)
        {
            int op = ops.selectOperator();   // hoặc cố định 1–2 op tốt nhất
            std::vector<int> X_old = X;
            double oldFitness = fitnessFunction();

            // block size cực nhỏ
            ops.apply(op, X, instance.mach, 1);

            double newFitness = fitnessFunction();

            if (newFitness < oldFitness)
            {
                exploitNoImprove = 0;

                if (newFitness < bestFitness)
                {
                    Xb = X;
                    bestFitness = newFitness;
                }
            }
            else
            {
                X = X_old;
                exploitNoImprove++;
            }
        }

        X = Xb;

        repair();
        std::cout << "Total Completion Time: " << totalCompletionTime() << "\n";
        // std::cout << "Total Energy Consumption: " << totalEnergyConsumption() << "\n";
    }

    bool lightLocalSearch(int maxSteps = 20)
    {
        double bestF = fitnessFunction();
        bool improved = false;

        for (int step = 0; step < maxSteps; ++step)
        {
            int m_heavy = findMaxLoadMachine();

            std::vector<int> jobs;
            for (int j = 1; j <= instance.job; ++j)
                if (X[j] == m_heavy)
                    jobs.push_back(j);

            if (jobs.empty()) break;

            int job = jobs[rng() % jobs.size()];
            int old_m = X[job];

            int new_m;
            do {
                new_m = rng() % instance.mach + 1;
            } while (new_m == old_m);

            X[job] = new_m;
            double newF = fitnessFunction();

            if (newF < bestF)
            {
                bestF = newF;
                improved = true;
            }
            else
            {
                X[job] = old_m;
            }
        }
        return improved;
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
            // if (local_single_relocate())
            // {
            //     std::cout << "Local search - Current fitness: " << fitnessFunction() << "\n";
            //     continue;
            // }
            // if (local_two_job_swap())
            // {
            //     std::cout << "Local search - Current fitness: " << fitnessFunction() << "\n";
            //     continue;
            // }

            // if (local_block_relocate(2))
            // {
            //     std::cout << "Local search - Current fitness: " << fitnessFunction() << "\n";
            //     continue;
            // }
            // if (local_block_swap(2))
            // {
            //     std::cout << "Local search - Current fitness: " << fitnessFunction() << "\n";
            //     continue;
            // }

            // if (local_heavy_to_light())
            // {
            //     std::cout << "Local search - Current fitness: " << fitnessFunction() << "\n";
            //     continue;
            // }
            // std::cout << "Local search - Current fitness: " << fitnessFunction() << "\n";
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
        bool isEscape;
        bool isAdaptivePenalty;
        bool isAdaptiveReward;
        double reheat_factor;
        double t_cap_factor;
        bool isPrint;
    };

    void parameterTuning()
    {
        std::vector<ParameterSet> params = {
            {0.999, 0.15, 0.4, 100, 0.8, 0.7, 50, 0.2, false, false, false, true,3.0, 0.2, false},

            // {0.999, 0.15, 0.4, 100, 0.4, 0.3, 50, 0.2, false, false, false, true,3.0, 0.2, false},
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
            isLNS = p.isLNS;
            isAdaptivePenalty = p.isAdaptivePenalty;
            isAdaptiveReward = p.isAdaptiveReward;
            isEscape = p.isEscape;
            isPrint = p.isPrint;

            std::vector<double> results;

            for (int run = 0; run < 20; ++run)
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