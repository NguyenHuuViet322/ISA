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
#include <unordered_map>

// ParamConfig ở global scope để maintabu.cpp dùng được
struct ParamConfig { double score = 0; };

class HybridTabuSolver
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

    // ── Tabu parameters ──────────────────────────
    // tabu_tenure: số iter một move bị cấm
    // tabu_ls_freq: cứ N iter kể từ lần cuối improve thì chạy tabu LS
    int    tabu_tenure  = 7;
    int    tabu_ls_freq = 300;   // ~= reset_threshold

    std::vector<int> machJobCount;

    // ── Tabu structures ───────────────────────────
    // Key: job * 10007 + toMachine  →  expireIter
    std::unordered_map<int, int> tabuMap;

    void tabuAdd(int job, int fromM, int toM, int currentIter)
    {
        // Cấm reverse move: toM → fromM
        int key = job * 10007 + fromM;   // reverse: job về fromM là tabu
        tabuMap[key] = currentIter + tabu_tenure;
    }

    bool tabuCheck(int job, int toM, int currentIter)
    {
        int key = job * 10007 + toM;
        auto it = tabuMap.find(key);
        if (it == tabuMap.end()) return false;
        if (it->second <= currentIter) { tabuMap.erase(it); return false; }
        return true;
    }

    // ── Tabu Local Search (adaptive theo bound_ratio) ─────
    // bound_ratio cao (loose) → scan nhiều, quality cao
    // bound_ratio thấp (tight) → sample ít, giữ iteration
    bool tabuLS(std::vector<int>& Xbest, std::vector<long long>& loadBest,
                std::vector<int>& cntBest, double& bestFitness, int currentIter)
    {
        X            = Xbest;
        machLoad     = loadBest;
        machJobCount = cntBest;

        // bound_ratio: 0=tight (T_2140), 1=loose (T_2151)
        double boundRatio = (upper_bound > lower_bound)
            ? (double)(bound - lower_bound) / (upper_bound - lower_bound)
            : 0.5;

        // sampleJobs: loose→nhiều hơn, tight→ít hơn
        int baseJobs  = std::max(10, (int)std::sqrt((double)instance.job));
        int sampleJobs = std::min(instance.job,
                         (int)(baseJobs * (0.5 + boundRatio)));  // 0.5x..1.5x

        // sampleMach: loose→scan hết, tight→sample sqrt
        bool scanAll  = (boundRatio > 0.5);
        int sampleMach = scanAll ? instance.mach
                                 : std::max(5, (int)std::sqrt((double)instance.mach));

        std::uniform_int_distribution<int> jobDist(1, instance.job);
        std::uniform_int_distribution<int> machDist(1, instance.mach);

        bool      improved = false;
        long long curTEC   = computeTEC_fromCache();

        for (int s = 0; s < sampleJobs; ++s)
        {
            int job   = jobDist(rng);
            int fromM = X[job];

            double bestDelta = -1e-9;
            int    bestToM   = -1;

            auto tryMach = [&](int toM) {
                if (toM == fromM) return;
                if (tabuCheck(job, toM, currentIter)) return;

                long long deltaTCT = (long long)instance.proc_time[job] *
                                     (machJobCount[toM] - machJobCount[fromM] + 1);
                long long deltaTEC = (long long)instance.proc_time[job] *
                                     ((long long)instance.unit_cost[toM - 1] -
                                      (long long)instance.unit_cost[fromM - 1]);
                long long newTEC  = curTEC + deltaTEC;
                double oldPen = (curTEC > bound)
                    ? penalty_factor*(double)(curTEC-bound)*(curTEC-bound) : 0.0;
                double newPen = (newTEC > bound)
                    ? penalty_factor*(double)(newTEC-bound)*(newTEC-bound) : 0.0;
                double delta = (double)deltaTCT + newPen - oldPen;
                if (delta < bestDelta) { bestDelta = delta; bestToM = toM; }
            };

            if (scanAll)
                for (int toM = 1; toM <= instance.mach; ++toM) tryMach(toM);
            else
                for (int sm = 0; sm < sampleMach; ++sm) tryMach(machDist(rng));

            if (bestToM != -1)
            {
                machLoad[fromM]       -= instance.proc_time[job];
                machLoad[bestToM]     += instance.proc_time[job];
                machJobCount[fromM]   -= 1;
                machJobCount[bestToM] += 1;
                curTEC += (long long)instance.proc_time[job] *
                          ((long long)instance.unit_cost[bestToM-1] -
                           (long long)instance.unit_cost[fromM-1]);
                X[job] = bestToM;
                tabuAdd(job, fromM, bestToM, currentIter);

                double newFit = computeFitness_fromCache();
                if (newFit < bestFitness)
                {
                    bestFitness = newFit;
                    Xbest       = X;
                    loadBest    = machLoad;
                    cntBest     = machJobCount;
                    improved    = true;
                    curTEC      = computeTEC_fromCache();
                }
            }
        }

        return improved;
    }

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
    HybridTabuSolver()
    {
        // runSingle("data/T_2160.txt", 20, 4);
        // runAllInstances(10);
        // runBigData(10);
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
    // Redistribution
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
    // ISA + Tabu LS
    // ─────────────────────────────────────────────
    int ISA(std::chrono::high_resolution_clock::time_point start_time)
    {
        std::vector<int>       Xb    = X;
        std::vector<long long> loadB = machLoad;
        std::vector<int>       cntB  = machJobCount;

        double currentFitness = computeFitness_fromCache();
        double bestFitness    = currentFitness;

        double T           = t0;
        int stagnation       = 0;
        int sinceLastImprove = 0;   // iter kể từ lần cuối cải thiện best
        int tabuFailCount    = 0;   // số lần tabuLS liên tiếp không improve
        int maxInnerIter   = (instance.job + instance.mach - 1) / instance.mach;
        int blockSize      = std::max(2, (int)std::ceil(block_parametter * instance.job / instance.mach));
        int totalIter      = 0;

        // Tenure scale theo problem size
        // tenure scale theo job count: nhỏ=5, lớn=20
        tabu_tenure  = std::max(5, std::min(20, instance.job / 25));
        // ls_freq tỉ lệ với job/mach ratio:
        // bài nhỏ (100j/20m=5): freq=150, bài lớn (500j/50m=10): freq=300
        // freq adaptive: loose→thường hơn, tight→ít hơn
        {
            double br = (upper_bound > lower_bound)
                ? (double)(bound - lower_bound) / (upper_bound - lower_bound) : 0.5;
            // loose (br=0.8): freq=180, tight (br=0.2): freq=420
            tabu_ls_freq = std::max(150, std::min(500,
                           (int)(reset_threshold * (1.5 - br))));
        }
        tabuMap.clear();
        tabuMap.reserve(512);

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
                auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration<double>(now - start_time).count() >= time_limit)
                {
                    timeUp = true;
                    break;
                }

                ++totalIter;
                ++sinceLastImprove;

                // ── SA operator move (ISA gốc) ────────────────────
                int op = ops.selectOperator();

                X_snap    = X;
                load_snap = machLoad;
                cnt_snap  = machJobCount;

                ops.apply(op, X, instance.mach, blockSize,
                          instance.proc_time, machLoad);

                rebuildCache();
                redistrubutionBasedOnCost();

                double newFitness = computeFitness_fromCache();
                double prob = std::exp(-(newFitness - currentFitness) / T);
                double r = dist(rng);

                if (newFitness < bestFitness)
                {
                    Xb = X; loadB = machLoad; cntB = machJobCount;
                    bestFitness = currentFitness = newFitness;
                    stagnation       = 0;
                    sinceLastImprove = 0;
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

                // ── Tabu LS: chỉ chạy khi bài đủ lớn và thời gian đủ dài
                // Bài nhỏ (job<=100) hoặc thời gian ngắn (<=1s): skip tabuLS
                // vì overhead restore + search làm mất SA exploration momentum
                bool enableTabuLS = (instance.job > 100 || time_limit > 1.0);
                if (enableTabuLS && sinceLastImprove > 0 && sinceLastImprove % tabu_ls_freq == 0)
                {
                    bool improved = tabuLS(Xb, loadB, cntB, bestFitness, totalIter);
                    if (improved)
                    {
                        currentFitness   = bestFitness;
                        X                = Xb;
                        machLoad         = loadB;
                        machJobCount     = cntB;
                        stagnation       = 0;
                        sinceLastImprove = 0;
                        tabuFailCount    = 0;
                    }
                    else
                    {
                        ++tabuFailCount;
                        // Sau 3 lần tabuLS liên tiếp không improve
                        // → perturbation nhẹ từ Xbest để thoát local optima
                        if (tabuFailCount >= 3)
                        {
                            X            = Xb;
                            machLoad     = loadB;
                            machJobCount = cntB;

                            int perturbN = std::max(2, instance.job / 50);
                            std::uniform_int_distribution<int> jd(1, instance.job);
                            std::uniform_int_distribution<int> md(1, instance.mach);
                            for (int p = 0; p < perturbN; ++p)
                            {
                                int pj = jd(rng), pm = md(rng);
                                if (pm == X[pj]) continue;
                                machLoad[X[pj]]     -= instance.proc_time[pj];
                                machLoad[pm]        += instance.proc_time[pj];
                                machJobCount[X[pj]] -= 1;
                                machJobCount[pm]    += 1;
                                X[pj] = pm;
                            }
                            rebuildCache();
                            redistrubutionBasedOnCost();
                            currentFitness = computeFitness_fromCache();
                            tabuFailCount  = 0;
                            ops.resetWeights();
                        }
                    }
                    tabuMap.clear();
                }

                // ── Reset weights khi stagnation ─────────────────
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
                      << " | TCT="  << (long long)obj
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

        std::cout << "\n--- Summary ---\n"
                  << "Best="  << (long long)best
                  << " Worst=" << (long long)worst
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
        if (!outFile.is_open()) { std::cerr << "Cannot open output file!\n"; return; }

        outFile << "Instance,Run,Objective,Runtime,Iterations\n";
        outFile.flush();

        for (int fileIdx = 1; fileIdx <= 2160; ++fileIdx)
        {
            std::stringstream ss;
            ss << "data/T_" << fileIdx << ".txt";
            if (!instance.readFromFile(ss.str())) continue;

            int n = instance.job, m = instance.mach;
            time_limit = (n <= 12 && m <= 3) ? 0.2 :
                         (n <= 50 && m <= 10) ? 1 : 5;

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

                outFile << fileIdx << ',' << run << ','
                        << (long long)obj << ','
                        << elapsed.count() << ',' << iterCount << '\n';
                outFile.flush();

                std::cout << "[" << fileIdx << "/2160] Run " << run
                          << " | TCT=" << (long long)obj
                          << " | Time=" << std::fixed << std::setprecision(3)
                          << elapsed.count() << "s\n";
                std::cout.flush();
            }
        }
        outFile.close();
    }

    // ─────────────────────────────────────────────
    // runBigData
    // ─────────────────────────────────────────────
    void runBigData(int numRuns)
    {
        std::ofstream outFile("big_data_results.csv");
        if (!outFile.is_open()) { std::cerr << "Cannot open output file!\n"; return; }

        outFile << "Folder,Instance,Mode,Run,Objective,Runtime,Iterations\n";
        outFile.flush();

        struct FolderInfo { std::string name; double timeLimit; };
        std::vector<FolderInfo> folders = {
            {"200_2000",  5},
            {"200_5000",  5},
            {"500_10000", 5}
        };

        std::vector<std::string> prefixes = {"N_N", "N_U", "U_N", "U_U"};

        struct ModeInfo {
            Operators::OperatorMode mode;
            std::string label;
        };
        std::vector<ModeInfo> modes = {
            { Operators::OperatorMode::SMART,    "SMART"    }
        };

        for (const auto& folder : folders)
        {
            std::string basePath = "data/big_data/" + folder.name + "/";
            time_limit = folder.timeLimit;

            std::cout << "\n========== Folder: " << folder.name
                      << " | TimeLimit=" << folder.timeLimit << "s ==========\n";
            std::cout.flush();

            for (const auto& prefix : prefixes)
            {
                for (int idx = 1; idx <= 5; ++idx)
                {
                    std::string filename = basePath + prefix + "_" + std::to_string(idx) + ".txt";
                    std::string instName = prefix + "_" + std::to_string(idx);

                    if (!instance.readFromFile(filename))
                    {
                        std::cerr << "Cannot read: " << filename << "\n";
                        continue;
                    }

                    upper_bound = (int)calculateUpper();
                    lower_bound = (int)calculateLower();
                    bound = (int)((1.0 - instance.ctrl_factor) * lower_bound +
                                   instance.ctrl_factor * upper_bound);

                    std::cout << "[" << folder.name << "] " << instName
                              << " | Jobs=" << instance.job
                              << " Machines=" << instance.mach << "\n";
                    std::cout.flush();

                    for (const auto& modeInfo : modes)
                    {
                        ops.mode = modeInfo.mode;
                        std::cout << "  -- Mode: " << modeInfo.label << " --\n";
                        std::cout.flush();

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

                            outFile << folder.name << ',' << instName << ','
                                    << ',' << run << ','
                                    << (long long)obj << ','
                                    << std::fixed << std::setprecision(3)
                                    << elapsed.count() << ',' << iterCount << '\n';
                            outFile.flush();

                            std::cout << "    Run " << run
                                      << " | TCT=" << (long long)obj
                                      << " | Iter=" << iterCount
                                      << " | Time=" << std::fixed << std::setprecision(3)
                                      << elapsed.count() << "s\n";
                            std::cout.flush();
                        }
                    }
                }
            }
        }
        outFile.close();
        std::cout << "\nDone! Results saved to big_data_results.csv\n";
    }

    // ─────────────────────────────────────────────
    // Repair
    // ─────────────────────────────────────────────
    void repair()
    {
        int repairIter = 0;
        if (computeTEC_fromCache() <= bound) return;
        std::cerr << "Repairing... TEC=" << computeTEC_fromCache()
                  << " bound=" << bound << "\n";

        std::vector<std::vector<int>> mj(instance.mach + 1);

        while (computeTEC_fromCache() > bound)
        {
            ++repairIter;
            if (repairIter > 10000) {
                std::cerr << "repair() STUCK!\n";
                break;
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

    ParamConfig tuneParameters(
        const std::vector<std::string>& filenames,
        int    runsPerInstance = 1,
        double tunTimeLimit    = 5.0,
        int    gridSamples     = 40,
        int    refineSamples   = 30,
        int    localIter       = 20,
        const std::string& outFile = "tuning_results.csv")
    {
        std::cout << "[tuneParameters] Not implemented in this version.\n";
        return ParamConfig{};
    }

    long long totalCompletionTime()    { return computeTCT_fromCache(); }
    long long totalEnergyConsumption() { return computeTEC_fromCache(); }
    double    fitnessFunction()         { return computeFitness_fromCache(); }
};

#endif