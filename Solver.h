#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include "Instance.h"
#include <vector>
#include <numeric>
#include "Operators.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cmath>

class Solver
{
public:
    std::mt19937 rng{std::random_device{}()};
    Instance instance;
    Operators ops;

    std::vector<int> X;
    std::vector<long long> machLoad;

    int upper_bound, lower_bound, bound;

    double block_parametter  = 0.4;
    double cooling_rate      = 0.999;
    double adaptation_rate   = 0.15;
    double phi1_score_factor = 0.8;
    double phi2_score_factor = 0.7;
    int    reset_threshold   = 300;
    double penalty_factor    = 50.0;
    double t0                = 1000.0;
    double beta              = 10.0;
    double time_limit        = 5.0;

    std::vector<int> machJobCount;

    // ─────────────────────────────────────────────
    // Cache
    // ─────────────────────────────────────────────
    void rebuildCache()
    {
        int m = instance.mach;
        machLoad.assign(m + 1, 0);
        machJobCount.assign(m + 1, 0);
        for (int j = 1; j <= instance.job; ++j)
        {
            machLoad[X[j]]     += instance.proc_time[j];
            machJobCount[X[j]] += 1;
        }
    }

    long long computeTCT_fromCache()
    {
        long long total = 0;
        std::vector<long long> runTime(instance.mach + 1, 0);
        for (int j = 1; j <= instance.job; ++j)
        {
            int m = X[j];
            runTime[m] += instance.proc_time[j];
            total += runTime[m];
        }
        return total;
    }

    long long computeTEC_fromCache()
    {
        long long U = 0;
        for (int m = 1; m <= instance.mach; ++m)
            U += machLoad[m] * instance.unit_cost[m - 1];
        return U;
    }

    double computeFitness_fromCache()
    {
        long long TCT = computeTCT_fromCache();
        long long TEC = computeTEC_fromCache();
        if (TEC <= bound)
            return (double)TCT;
        double diff = (double)(TEC - bound);
        return (double)TCT + penalty_factor * diff * diff;
    }

    // ─────────────────────────────────────────────
    // Constructor
    // ─────────────────────────────────────────────
    Solver()
    {
        // runSingle("data/T_2160.txt", 20, 4);
        runAllInstances(10);
    }

    // ─────────────────────────────────────────────
    // Init
    // ─────────────────────────────────────────────
    void init()
    {
        rebuildCache();
        long long cost = computeTEC_fromCache();
        long long iter = 1;
        while (true)
        {
            long long theta = instance.mach;
            while (theta > 1 &&
                   (long long)instance.proc_time[iter] *
                   (instance.unit_cost[theta - 1] - instance.unit_cost[0]) > (bound - cost))
                --theta;

            if (theta == 1) break;

            int oldM = X[iter];
            machLoad[oldM]     -= instance.proc_time[iter];
            machJobCount[oldM] -= 1;
            X[iter] = (int)theta;
            machLoad[theta]     += instance.proc_time[iter];
            machJobCount[theta] += 1;

            redistrubutionBasedOnCost();
            cost = computeTEC_fromCache();
            ++iter;
            if (iter > instance.job) break;
        }
    }

    // ─────────────────────────────────────────────
    // Redistribution — O(m log m + n)
    // ─────────────────────────────────────────────
    void redistrubutionBasedOnCost()
    {
        int m = instance.mach;
        std::vector<int> order(m);
        std::iota(order.begin(), order.end(), 1);
        std::sort(order.begin(), order.end(),
            [&](int a, int b){ return machLoad[a] > machLoad[b]; });

        std::vector<int> remap(m + 1);
        for (int i = 0; i < m; ++i)
            remap[order[i]] = i + 1;

        std::vector<long long> newLoad(m + 1, 0);
        std::vector<int>       newCnt (m + 1, 0);
        for (int j = 1; j <= instance.job; ++j)
        {
            X[j] = remap[X[j]];
            newLoad[X[j]] += instance.proc_time[j];
            newCnt [X[j]] += 1;
        }
        machLoad     = std::move(newLoad);
        machJobCount = std::move(newCnt);
    }

    // ─────────────────────────────────────────────
    // ISA
    // ─────────────────────────────────────────────
    int ISA(std::chrono::high_resolution_clock::time_point start_time)
    {
        std::vector<int>       Xb    = X;
        std::vector<long long> loadB = machLoad;
        std::vector<int>       cntB  = machJobCount;

        double currentFitness = computeFitness_fromCache();
        double bestFitness    = currentFitness;

        double T         = t0;
        int stagnation   = 0;
        int maxInnerIter = (instance.job + instance.mach - 1) / instance.mach;
        int blockSize    = std::max(2, (int)std::ceil(block_parametter * instance.job / instance.mach));
        int totalIter    = 0;

        std::uniform_real_distribution<double> dist(0.0, 1.0);

        std::vector<int>       X_snap;
        std::vector<long long> load_snap;
        std::vector<int>       cnt_snap;
        X_snap.reserve(instance.job + 1);
        load_snap.reserve(instance.mach + 1);
        cnt_snap.reserve(instance.mach + 1);

        bool timeUp = false;
        while (!timeUp)
        {
            for (int iter = 0; iter < maxInnerIter && !timeUp; ++iter)
            {
                // Check time inside inner loop
                auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration<double>(now - start_time).count() >= time_limit)
                {
                    timeUp = true;
                    break;
                }

                ++totalIter;
                int op = ops.selectOperator();

                X_snap    = X;
                load_snap = machLoad;
                cnt_snap  = machJobCount;

                ops.apply(op, X, instance.mach, blockSize, instance.proc_time);

                rebuildCache();
                redistrubutionBasedOnCost();

                double newFitness = computeFitness_fromCache();
                double prob = std::exp(-(newFitness - currentFitness) / T);
                double r = dist(rng);

                if (newFitness < bestFitness)
                {
                    Xb = X; loadB = machLoad; cntB = machJobCount;
                    bestFitness = currentFitness = newFitness;
                    stagnation  = 0;
                    ops.reward(op, beta);
                }
                else if (newFitness < currentFitness)
                {
                    currentFitness = newFitness;
                    ++stagnation;
                    ops.reward(op, phi1_score_factor * beta);
                }
                else if (r < prob)
                {
                    currentFitness = newFitness;
                    ++stagnation;
                    ops.reward(op, phi2_score_factor * beta);
                }
                else
                {
                    X            = std::move(X_snap);
                    machLoad     = std::move(load_snap);
                    machJobCount = std::move(cnt_snap);
                    X_snap.resize(instance.job + 1);
                    load_snap.resize(instance.mach + 1);
                    cnt_snap.resize(instance.mach + 1);
                }

                if (stagnation >= reset_threshold)
                {
                    ops.resetWeights();
                    stagnation = 0;
                }
            }

            if (!timeUp)
            {
                ops.updateWeights(adaptation_rate);
                T *= cooling_rate;
            }
        }

        X = Xb; machLoad = loadB; machJobCount = cntB;
        repair();
        return totalIter;
    }

    // ─────────────────────────────────────────────
    // runSingle
    // ─────────────────────────────────────────────
    void runSingle(const std::string& filename, int numRuns, double timeLimitSec)
    {
        if (!instance.readFromFile(filename))
        {
            std::cerr << "Cannot read file: " << filename << "\n";
            return;
        }
        time_limit = timeLimitSec;

        upper_bound = (int)calculateUpper();
        lower_bound = (int)calculateLower();
        bound = (int)((1.0 - instance.ctrl_factor) * lower_bound +
                       instance.ctrl_factor * upper_bound);

        std::cout << "=== " << filename << " ===\n";
        std::cout << "Jobs=" << instance.job << " Machines=" << instance.mach << "\n";
        std::cout << "Lower=" << lower_bound << " Upper=" << upper_bound
                  << " Bound=" << bound << "\n\n";
        std::cout.flush();

        std::vector<double> results, runtimes;

        for (int run = 1; run <= numRuns; ++run)
        {
            X.assign(instance.job + 1, 1);
            ops.resetWeights();
            init();
            t0 = 0.2 * computeFitness_fromCache() / std::log(2.0);

            auto start_time = std::chrono::high_resolution_clock::now();
            int iterCount   = ISA(start_time);
            auto end_time   = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end_time - start_time;
            double obj = computeTCT_fromCache();

            results.push_back(obj);
            runtimes.push_back(elapsed.count());

            std::cout << "Run " << run
                      << " | TCT="  << obj
                      << " | Iter=" << iterCount
                      << " | Time=" << std::fixed << std::setprecision(3)
                      << elapsed.count() << "s\n";
            std::cout.flush();
        }

        double best  = *std::min_element(results.begin(), results.end());
        double worst = *std::max_element(results.begin(), results.end());
        double avg   = std::accumulate(results.begin(), results.end(), 0.0) / results.size();
        double var   = 0.0;
        for (double v : results) var += (v - avg) * (v - avg);
        double stddev  = std::sqrt(var / results.size());
        double avgTime = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();

        std::cout << "\n--- Summary ---\n";
        std::cout << "Best="  << best
                  << " Worst=" << worst
                  << " Avg="   << std::fixed << std::setprecision(2) << avg
                  << " Std="   << stddev
                  << " AvgTime=" << std::setprecision(3) << avgTime << "s\n";
        std::cout.flush();
    }

    // ─────────────────────────────────────────────
    // runAllInstances
    // ─────────────────────────────────────────────
    void runAllInstances(int numRuns)
    {
        std::ofstream outFile("batch_results6.csv");
        if (!outFile.is_open())
        {
            std::cerr << "Cannot open output file!\n";
            return;
        }
        outFile << "Instance,Run,Objective,Runtime,Iterations\n";
        outFile.flush();

        for (int fileIdx = 1; fileIdx <= 2160; ++fileIdx)
        {
            std::stringstream ss;
            ss << "data/T_" << fileIdx << ".txt";
            if (!instance.readFromFile(ss.str())) continue;

            int n = instance.job, m = instance.mach;
            time_limit = (n <= 20 && m <= 4) ? 1.0 :
                         (n <= 50 && m <= 6) ? 2.0 : 4.0;

            upper_bound = (int)calculateUpper();
            lower_bound = (int)calculateLower();
            bound = (int)((1.0 - instance.ctrl_factor) * lower_bound +
                           instance.ctrl_factor * upper_bound);

            for (int run = 1; run <= numRuns; ++run)
            {
                X.assign(instance.job + 1, 1);
                ops.resetWeights();
                init();
                t0 = 0.2 * computeFitness_fromCache() / std::log(2.0);

                auto start_time = std::chrono::high_resolution_clock::now();
                int iterCount = ISA(start_time);
                auto end_time = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> elapsed = end_time - start_time;
                double obj = computeTCT_fromCache();

                outFile << fileIdx << ',' << run << ',' << obj << ','
                        << elapsed.count() << ',' << iterCount << '\n';
                outFile.flush();

                std::cout << "[" << fileIdx << "/2160] Run " << run
                          << " | TCT=" << obj
                          << " | Time=" << std::fixed << std::setprecision(3)
                          << elapsed.count() << "s\n";
                std::cout.flush();
            }
        }
        outFile.close();
    }

    // ─────────────────────────────────────────────
    // Repair
    // ─────────────────────────────────────────────
    void repair()
    {
        int repairIter = 0;
        if (computeTEC_fromCache() <= bound) return;
    std::cerr << "Repairing... TEC=" << computeTEC_fromCache() << " bound=" << bound << "\n";

        std::vector<std::vector<int>> mj(instance.mach + 1);

        while (computeTEC_fromCache() > bound)
        {
            ++repairIter;
        if (repairIter > 10000) {
            std::cerr << "repair() STUCK! TEC=" << computeTEC_fromCache() << " bound=" << bound << "\n";
            break;  // thoát để không treo
        }
            for (int m = 0; m <= instance.mach; ++m) mj[m].clear();
            for (int i = 1; i <= instance.job; ++i) mj[X[i]].push_back(i);

            int gamma = 0;
            for (int m = instance.mach; m >= 1; --m)
                if (!mj[m].empty()) { gamma = m; break; }
            if (gamma == 0) break;

            for (int s = 1; s < gamma; ++s)
            {
                int target = gamma - s;
                if (target < 1 || mj[gamma].empty()) break;
                int job = mj[gamma][0];
                X[job] = target;
                mj[target].push_back(job);
                mj[gamma].erase(mj[gamma].begin());
            }

            for (int m = 1; m <= instance.mach; ++m)
                std::sort(mj[m].begin(), mj[m].end(),
                    [&](int a, int b){ return instance.proc_time[a] < instance.proc_time[b]; });

            rebuildCache();
        }
    }

    // ─────────────────────────────────────────────
    // Bounds
    // ─────────────────────────────────────────────
    long long calculateUpper()
    {
        std::vector<long long> x_tmp(instance.mach, 0);
        for (int i = 1; i <= instance.job; ++i)
        {
            int p = (int)(std::min_element(x_tmp.begin(), x_tmp.end()) - x_tmp.begin());
            x_tmp[p] += instance.proc_time[i];
        }
        long long U = 0;
        for (int i = 0; i < instance.mach; ++i)
            U += x_tmp[i] * instance.unit_cost[i];
        return U;
    }

    long long calculateLower()
    {
        long long U = 0;
        for (int i = 1; i <= instance.job; ++i)
            U += (long long)instance.proc_time[i] * instance.unit_cost[0];
        return U;
    }

    long long totalCompletionTime()   { return computeTCT_fromCache(); }
    long long totalEnergyConsumption(){ return computeTEC_fromCache(); }
    double    fitnessFunction()        { return computeFitness_fromCache(); }
};

#endif