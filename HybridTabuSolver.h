#ifndef HYBRID_TABU_SOLVER_H
#define HYBRID_TABU_SOLVER_H

#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <deque>
#include <functional>

#include "Instance.h"
#include "Operators.h"

// ═══════════════════════════════════════════════════════════════════
// Hash function for solution representation (để lưu trong tabu list)
// ═══════════════════════════════════════════════════════════════════
struct SolutionHash
{
    size_t operator()(const std::vector<int>& X) const
    {
        size_t seed = X.size();
        for (size_t i = 1; i < X.size(); ++i)
            seed ^= X[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

// ═══════════════════════════════════════════════════════════════════
// Move representation for attribute-based tabu
// ═══════════════════════════════════════════════════════════════════
struct TabuMove
{
    int job;
    int fromMachine;
    int toMachine;
    int expireIter;        // iteration tại đó move này hết hiệu lực

    bool operator==(const TabuMove& other) const
    {
        return job == other.job &&
               fromMachine == other.fromMachine &&
               toMachine == other.toMachine;
    }
};

struct TabuMoveHash
{
    size_t operator()(const TabuMove& m) const
    {
        return std::hash<int>()(m.job) ^
               (std::hash<int>()(m.fromMachine) << 8) ^
               (std::hash<int>()(m.toMachine) << 16);
    }
};

// ═══════════════════════════════════════════════════════════════════
// Parameter configuration (dùng cho tuning)
// ═══════════════════════════════════════════════════════════════════
struct ParamConfig
{
    int    tabu_tenure_base;
    int    tabu_tenure_dynamic;
    double aspiration_factor;
    double cooling_rate;
    double reheat_factor;
    double block_parameter;
    double adaptation_rate;
    int    intensify_thresh;
    int    diversify_thresh;
    double penalty_factor;
    int    neighborhood_cap;
    int    phase2_freq;

    double score = 1e18;   // avg TCT qua các run (thấp hơn = tốt hơn)

    void print(std::ostream& os) const
    {
        os << std::fixed << std::setprecision(4)
           << "tenure_base="    << tabu_tenure_base
           << " tenure_dyn="    << tabu_tenure_dynamic
           << " aspiration="    << aspiration_factor
           << " cooling="       << cooling_rate
           << " reheat="        << reheat_factor
           << " block="         << block_parameter
           << " adapt="         << adaptation_rate
           << " intensify="     << intensify_thresh
           << " diversify="     << diversify_thresh
           << " penalty="       << penalty_factor
           << " nbhd_cap="      << neighborhood_cap
           << " p2_freq="       << phase2_freq
           << " | score="       << score;
    }
};

// ═══════════════════════════════════════════════════════════════════
// Hybrid Tabu Search Solver
// ═══════════════════════════════════════════════════════════════════
class HybridTabuSolver
{
public:
    std::mt19937 rng{std::random_device{}()};
    Instance instance;
    Operators ops;

    std::vector<int> X;
    std::vector<long long> machLoad;
    std::vector<int> machJobCount;

    int upper_bound, lower_bound, bound;

    // ─────────────────────────────────────────────
    // Hybrid Parameters
    // ─────────────────────────────────────────────

    // Tabu parameters
    int    tabu_tenure_base    = 7;       // Base tabu tenure
    int    tabu_tenure_dynamic = 5;       // Dynamic component (random ± này)
    int    max_tabu_size       = 1000;    // Max size của solution-based hash set
    double aspiration_factor   = 0.98;    // Accept nếu < aspiration_factor * best

    // SA parameters (for hybrid)
    double T                   = 1000.0;
    double cooling_rate        = 0.995;
    double reheat_factor       = 1.5;     // Reheat khi stuck

    // Search parameters (default — sẽ được rescale theo problem size)
    double block_parameter     = 0.4;
    double adaptation_rate     = 0.15;
    int    intensify_thresh    = 200;
    int    diversify_thresh    = 600;
    double penalty_factor      = 50.0;
    double time_limit          = 5.0;

    // Neighborhood cap (tunable — override của công thức trong hybridTabuSearch)
    int    neighborhood_cap    = 50;      // 0 = dùng công thức tự động
    int    phase2_freq         = 10;      // Phase 2 chạy mỗi N iter

    // Tabu structures
    std::deque<TabuMove>       tabuList;     // Attribute-based tabu list
    std::unordered_set<size_t> visitedHash;  // Solution-based (hash only)
    int currentIteration = 0;

    // Statistics
    int intensifyCount  = 0;
    int diversifyCount  = 0;
    int aspirationCount = 0;

    // ─────────────────────────────────────────────
    // Apply / Extract ParamConfig
    // ─────────────────────────────────────────────
    void applyConfig(const ParamConfig& cfg)
    {
        tabu_tenure_base    = cfg.tabu_tenure_base;
        tabu_tenure_dynamic = cfg.tabu_tenure_dynamic;
        aspiration_factor   = cfg.aspiration_factor;
        cooling_rate        = cfg.cooling_rate;
        reheat_factor       = cfg.reheat_factor;
        block_parameter     = cfg.block_parameter;
        adaptation_rate     = cfg.adaptation_rate;
        intensify_thresh    = cfg.intensify_thresh;
        diversify_thresh    = cfg.diversify_thresh;
        penalty_factor      = cfg.penalty_factor;
        neighborhood_cap    = cfg.neighborhood_cap;
        phase2_freq         = cfg.phase2_freq;
    }

    ParamConfig extractConfig() const
    {
        ParamConfig cfg;
        cfg.tabu_tenure_base    = tabu_tenure_base;
        cfg.tabu_tenure_dynamic = tabu_tenure_dynamic;
        cfg.aspiration_factor   = aspiration_factor;
        cfg.cooling_rate        = cooling_rate;
        cfg.reheat_factor       = reheat_factor;
        cfg.block_parameter     = block_parameter;
        cfg.adaptation_rate     = adaptation_rate;
        cfg.intensify_thresh    = intensify_thresh;
        cfg.diversify_thresh    = diversify_thresh;
        cfg.penalty_factor      = penalty_factor;
        cfg.neighborhood_cap    = neighborhood_cap;
        cfg.phase2_freq         = phase2_freq;
        return cfg;
    }

    // ─────────────────────────────────────────────
    // Cache operations
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
    // Tabu Management
    // ─────────────────────────────────────────────

    int sampleTenure()
    {
        int dyn   = (tabu_tenure_dynamic > 0)
                    ? (int)(rng() % tabu_tenure_dynamic) : 0;
        int base  = tabu_tenure_base + dyn;
        int scale = std::max(1, instance.job / 50);
        return base * scale;
    }

    void addTabuMove(int job, int fromM, int toM)
    {
        int tenure = sampleTenure();
        tabuList.push_back({job, fromM, toM, currentIteration + tenure});

        // Cleanup các move đã expire
        while (!tabuList.empty() &&
               tabuList.front().expireIter <= currentIteration)
        {
            tabuList.pop_front();
        }
    }

    bool isTabu(int job, int fromM, int toM)
    {
        for (const auto& m : tabuList)
        {
            if (m.job == job &&
                m.toMachine == fromM &&
                m.fromMachine == toM &&
                currentIteration < m.expireIter)
            {
                return true;
            }
        }
        return false;
    }

    bool isAspirationMet(double newFitness, double bestFitness)
    {
        return newFitness < aspiration_factor * bestFitness;
    }

    void recordSolutionHash()
    {
        size_t h = SolutionHash{}(X);
        visitedHash.insert(h);
    }

    bool isSolutionVisited()
    {
        size_t h = SolutionHash{}(X);
        return visitedHash.find(h) != visitedHash.end();
    }

    // ─────────────────────────────────────────────
    // Fast Delta Evaluation
    // ─────────────────────────────────────────────
    double computeDeltaFitness(int job, int fromM, int toM)
    {
        long long deltaTCT = (long long)instance.proc_time[job] *
                             (machJobCount[toM] - machJobCount[fromM] + 1);

        long long deltaTEC = (long long)instance.proc_time[job] *
                             (instance.unit_cost[toM - 1] - instance.unit_cost[fromM - 1]);

        long long oldTEC = computeTEC_fromCache();
        long long newTEC = oldTEC + deltaTEC;

        double oldPenalty = (oldTEC > bound)
            ? penalty_factor * (double)(oldTEC - bound) * (oldTEC - bound) : 0.0;
        double newPenalty = (newTEC > bound)
            ? penalty_factor * (double)(newTEC - bound) * (newTEC - bound) : 0.0;

        return (double)deltaTCT + newPenalty - oldPenalty;
    }

    // ─────────────────────────────────────────────
    // Neighborhood Exploration
    // ─────────────────────────────────────────────
    struct Neighbor
    {
        int job;
        int fromMachine;
        int toMachine;
        double deltaFitness;
        bool   isTabu;
    };

    std::vector<Neighbor> generateNeighborhood(int sampleSize)
    {
        std::vector<Neighbor> neighbors;
        neighbors.reserve(sampleSize);

        std::uniform_int_distribution<int> jobDist(1, instance.job);
        std::uniform_int_distribution<int> machDist(1, instance.mach);

        for (int i = 0; i < sampleSize; ++i)
        {
            int job   = jobDist(rng);
            int fromM = X[job];
            int toM;
            do { toM = machDist(rng); } while (toM == fromM);

            double delta = computeDeltaFitness(job, fromM, toM);
            bool   tabu  = isTabu(job, fromM, toM);

            neighbors.push_back({job, fromM, toM, delta, tabu});
        }

        return neighbors;
    }

    // ─────────────────────────────────────────────
    // Intensification
    // ─────────────────────────────────────────────
    void intensify(std::vector<int>& Xbest, double& bestFitness)
    {
        ++intensifyCount;

        X = Xbest;
        rebuildCache();

        int maxIterLS = std::min(100, instance.job / 10);
        std::uniform_int_distribution<int> jobDist(1, instance.job);

        for (int iterLS = 0; iterLS < maxIterLS; ++iterLS)
        {
            int job = jobDist(rng);
            int oldM = X[job];

            for (int newM = 1; newM <= instance.mach; ++newM)
            {
                if (newM == oldM) continue;

                double delta = computeDeltaFitness(job, oldM, newM);

                if (delta < -1e-9)
                {
                    machLoad[oldM]     -= instance.proc_time[job];
                    machLoad[newM]     += instance.proc_time[job];
                    machJobCount[oldM] -= 1;
                    machJobCount[newM] += 1;
                    X[job] = newM;

                    double newFit = computeFitness_fromCache();
                    if (newFit < bestFitness)
                    {
                        bestFitness = newFit;
                        Xbest = X;
                    }
                    break;
                }
            }
        }
    }

    // ─────────────────────────────────────────────
    // Diversification
    // ─────────────────────────────────────────────
    void diversify()
    {
        ++diversifyCount;

        int perturbSize = std::max(3, instance.job / 5);
        std::uniform_int_distribution<int> jobDist(1, instance.job);
        std::uniform_int_distribution<int> machDist(1, instance.mach);

        for (int i = 0; i < perturbSize; ++i)
        {
            int job  = jobDist(rng);
            int oldM = X[job];
            int newM = machDist(rng);

            machLoad[oldM] -= instance.proc_time[job];
            machLoad[newM] += instance.proc_time[job];
            X[job] = newM;
        }

        tabuList.clear();
        T *= reheat_factor;
        ops.resetWeights();

        rebuildCache();
        redistrubutionBasedOnCost();
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
    // Rescale thresholds
    // ─────────────────────────────────────────────
    void rescaleThresholds()
    {
        int nm = std::max(1, instance.job / instance.mach);
        intensify_thresh = std::max(100, 2 * nm);
        diversify_thresh = std::max(300, 6 * nm);
    }

    // ─────────────────────────────────────────────
    // Constructor
    // ─────────────────────────────────────────────
    HybridTabuSolver() = default;

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

    // ═══════════════════════════════════════════════════════════════════
    // HYBRID TABU SEARCH
    // ═══════════════════════════════════════════════════════════════════
    int hybridTabuSearch(std::chrono::high_resolution_clock::time_point start_time)
    {
        std::vector<int>       Xbest    = X;
        std::vector<long long> loadBest = machLoad;
        std::vector<int>       cntBest  = machJobCount;

        double currentFitness = computeFitness_fromCache();
        double bestFitness    = currentFitness;

        tabuList.clear();
        visitedHash.clear();
        currentIteration = 0;

        int stagnation = 0;
        int blockSize  = std::max(2, (int)std::ceil(block_parameter * instance.job / instance.mach));

        rescaleThresholds();

        // neighborhood_cap == 0 → dùng công thức tự động; > 0 → dùng giá trị đã set
        int neighborhoodSize = (neighborhood_cap > 0)
            ? neighborhood_cap
            : std::min(50, std::max(10, instance.job / 20));

        T = 0.2 * currentFitness / std::log(2.0);

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
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration<double>(now - start_time).count() >= time_limit)
            {
                timeUp = true;
                break;
            }

            ++currentIteration;

            // ─── Phase 1: Tabu-guided ───────────────────
            auto neighbors = generateNeighborhood(neighborhoodSize);

            Neighbor* bestMove    = nullptr;
            Neighbor* bestNonTabu = nullptr;

            for (auto& n : neighbors)
            {
                if (!n.isTabu)
                {
                    if (!bestNonTabu || n.deltaFitness < bestNonTabu->deltaFitness)
                        bestNonTabu = &n;
                }
                if (!bestMove || n.deltaFitness < bestMove->deltaFitness)
                    bestMove = &n;
            }

            Neighbor* chosen = nullptr;

            if (bestMove &&
                (currentFitness + bestMove->deltaFitness) < aspiration_factor * bestFitness)
            {
                chosen = bestMove;
                ++aspirationCount;
            }
            else if (bestNonTabu)
            {
                chosen = bestNonTabu;
            }
            else if (bestMove)
            {
                chosen = bestMove;
            }

            if (chosen)
            {
                int job   = chosen->job;
                int fromM = chosen->fromMachine;
                int toM   = chosen->toMachine;

                machLoad[fromM]     -= instance.proc_time[job];
                machLoad[toM]       += instance.proc_time[job];
                machJobCount[fromM] -= 1;
                machJobCount[toM]   += 1;
                X[job] = toM;

                addTabuMove(job, fromM, toM);

                currentFitness = computeFitness_fromCache();

                if (currentFitness < bestFitness)
                {
                    bestFitness = currentFitness;
                    Xbest    = X;
                    loadBest = machLoad;
                    cntBest  = machJobCount;
                    stagnation = 0;
                }
                else
                {
                    ++stagnation;
                }
            }

            // ─── Phase 2: Operator perturbation ─────────
            bool runPhase2 = (currentIteration % phase2_freq == 0) || (stagnation >= 15);
            if (runPhase2)
            {
                X_snap    = X;
                load_snap = machLoad;
                cnt_snap  = machJobCount;

                int op = ops.selectOperator();
                ops.apply(op, X, instance.mach, blockSize, instance.proc_time);

                rebuildCache();
                redistrubutionBasedOnCost();

                double opFitness = computeFitness_fromCache();
                double prob = std::exp(-(opFitness - currentFitness) / T);

                bool accepted = false;

                if (opFitness < bestFitness)
                {
                    Xbest    = X;
                    loadBest = machLoad;
                    cntBest  = machJobCount;
                    bestFitness = currentFitness = opFitness;
                    stagnation  = 0;
                    ops.reward(op, 10.0);
                    accepted = true;
                }
                else if (opFitness < currentFitness)
                {
                    currentFitness = opFitness;
                    ops.reward(op, 7.0);
                    accepted = true;
                }
                else if (dist(rng) < prob)
                {
                    currentFitness = opFitness;
                    ops.reward(op, 5.0);
                    accepted = true;
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

                if (accepted)
                    tabuList.clear();
            }

            // ─── Phase 3: Intensify / Diversify ─────────
            if (stagnation >= diversify_thresh)
            {
                diversify();
                currentFitness = computeFitness_fromCache();
                stagnation = 0;
            }
            else if (stagnation > 0 &&
                     stagnation % intensify_thresh == 0)
            {
                intensify(Xbest, bestFitness);
                X            = Xbest;
                machLoad     = loadBest;
                machJobCount = cntBest;
                currentFitness = bestFitness;
            }

            // Cooling
            T *= cooling_rate;
            if (T < 1e-6) T = 1e-6;

            if (currentIteration % 30 == 0)
                ops.updateWeights(adaptation_rate);
        }

        X            = Xbest;
        machLoad     = loadBest;
        machJobCount = cntBest;

        repair();
        return currentIteration;
    }

    // ─────────────────────────────────────────────
    // Evaluate một config trên tập instances, trả về avg TCT
    // ─────────────────────────────────────────────
    double evaluateConfig(const ParamConfig& cfg,
                          const std::vector<std::string>& filenames,
                          int runsPerInstance,
                          double tunTimeLimit)
    {
        applyConfig(cfg);
        time_limit = tunTimeLimit;

        double totalTCT = 0.0;
        int    count    = 0;

        for (const auto& fname : filenames)
        {
            if (!instance.readFromFile(fname)) continue;

            upper_bound = (int)calculateUpper();
            lower_bound = (int)calculateLower();
            bound = (int)((1.0 - instance.ctrl_factor) * lower_bound +
                           instance.ctrl_factor * upper_bound);

            for (int r = 0; r < runsPerInstance; ++r)
            {
                X.assign(instance.job + 1, 1);
                ops.resetWeights();
                intensifyCount = diversifyCount = aspirationCount = 0;
                init();

                auto t0 = std::chrono::high_resolution_clock::now();
                hybridTabuSearch(t0);

                totalTCT += (double)computeTCT_fromCache();
                ++count;
            }
        }

        return (count > 0) ? totalTCT / count : 1e18;
    }

    // ═══════════════════════════════════════════════════════════════════
    // PARAMETER TUNING
    // Chiến lược: Grid Search thô → Random Search tinh → Local Search
    //
    // Tham số:
    //   filenames      — danh sách file instance dùng để đánh giá
    //   runsPerInstance— số lần chạy mỗi instance để lấy avg
    //   tunTimeLimit   — time limit cho mỗi single run khi tuning
    //   gridSamples    — số cấu hình random ở phase grid search
    //   refineSamples  — số cấu hình random quanh best ở phase refine
    //   localIter      — số bước local perturbation ở phase local search
    //   outFile        — tên file CSV lưu kết quả (rỗng = không lưu)
    // ═══════════════════════════════════════════════════════════════════
    ParamConfig tuneParameters(
        const std::vector<std::string>& filenames,
        int    runsPerInstance = 1,
        double tunTimeLimit    = 5.0,
        int    gridSamples     = 40,
        int    refineSamples   = 30,
        int    localIter       = 20,
        const std::string& outFile = "tuning_results.csv")
    {
        // ── Không gian tìm kiếm ────────────────────────────────────────
        // Mỗi vector là tập giá trị rời rạc được xem xét
        std::vector<int>    v_tenure_base   = {4, 7, 10, 15, 20};
        std::vector<int>    v_tenure_dyn    = {0, 3, 5, 8, 12};
        std::vector<double> v_aspiration    = {0.95, 0.97, 0.98, 0.99, 0.999};
        std::vector<double> v_cooling       = {0.990, 0.993, 0.995, 0.997, 0.999};
        std::vector<double> v_reheat        = {1.2, 1.5, 2.0, 2.5, 3.0};
        std::vector<double> v_block         = {0.2, 0.3, 0.4, 0.6, 0.8};
        std::vector<double> v_adapt         = {0.05, 0.10, 0.15, 0.25, 0.40};
        std::vector<int>    v_intensify     = {80, 150, 200, 300, 500};
        std::vector<int>    v_diversify     = {200, 400, 600, 900, 1500};
        std::vector<double> v_penalty       = {10, 30, 50, 100, 200, 400};
        std::vector<int>    v_nbhd          = {15, 25, 40, 60, 100};
        std::vector<int>    v_p2freq        = {5, 8, 10, 15, 25};

        // ── Helpers ────────────────────────────────────────────────────
        auto pickRand = [&](auto& vec) -> decltype(vec[0]) {
            std::uniform_int_distribution<int> d(0, (int)vec.size() - 1);
            return vec[d(rng)];
        };

        auto randomConfig = [&]() -> ParamConfig {
            ParamConfig c;
            c.tabu_tenure_base    = pickRand(v_tenure_base);
            c.tabu_tenure_dynamic = pickRand(v_tenure_dyn);
            c.aspiration_factor   = pickRand(v_aspiration);
            c.cooling_rate        = pickRand(v_cooling);
            c.reheat_factor       = pickRand(v_reheat);
            c.block_parameter     = pickRand(v_block);
            c.adaptation_rate     = pickRand(v_adapt);
            c.intensify_thresh    = pickRand(v_intensify);
            c.diversify_thresh    = pickRand(v_diversify);
            // đảm bảo intensify < diversify
            if (c.intensify_thresh >= c.diversify_thresh)
                c.diversify_thresh = c.intensify_thresh * 3;
            c.penalty_factor      = pickRand(v_penalty);
            c.neighborhood_cap    = pickRand(v_nbhd);
            c.phase2_freq         = pickRand(v_p2freq);
            return c;
        };

        // Tạo config "neighbor" bằng cách perturbation nhỏ từ cfg gốc
        auto perturbConfig = [&](const ParamConfig& base) -> ParamConfig {
            ParamConfig c = base;
            std::uniform_int_distribution<int> dim(0, 11);
            int d = dim(rng);
            switch (d) {
                case 0:  c.tabu_tenure_base    = pickRand(v_tenure_base);   break;
                case 1:  c.tabu_tenure_dynamic = pickRand(v_tenure_dyn);    break;
                case 2:  c.aspiration_factor   = pickRand(v_aspiration);    break;
                case 3:  c.cooling_rate        = pickRand(v_cooling);       break;
                case 4:  c.reheat_factor       = pickRand(v_reheat);        break;
                case 5:  c.block_parameter     = pickRand(v_block);         break;
                case 6:  c.adaptation_rate     = pickRand(v_adapt);         break;
                case 7:  c.intensify_thresh    = pickRand(v_intensify);     break;
                case 8:  c.diversify_thresh    = pickRand(v_diversify);     break;
                case 9:  c.penalty_factor      = pickRand(v_penalty);       break;
                case 10: c.neighborhood_cap    = pickRand(v_nbhd);          break;
                case 11: c.phase2_freq         = pickRand(v_p2freq);        break;
            }
            if (c.intensify_thresh >= c.diversify_thresh)
                c.diversify_thresh = c.intensify_thresh * 3;
            return c;
        };

        // ── Mở file CSV ────────────────────────────────────────────────
        std::ofstream csv;
        bool saveCSV = !outFile.empty();
        if (saveCSV)
        {
            csv.open(outFile);
            if (csv.is_open())
                csv << "Phase,Trial,tenure_base,tenure_dyn,aspiration,cooling,reheat,"
                       "block,adapt,intensify,diversify,penalty,nbhd_cap,p2_freq,score\n";
            else
            {
                std::cerr << "[Tuner] Cannot open " << outFile << " — CSV disabled\n";
                saveCSV = false;
            }
        }

        auto writeCSV = [&](const std::string& phase, int trial, const ParamConfig& c) {
            if (!saveCSV) return;
            csv << phase << ',' << trial << ','
                << c.tabu_tenure_base    << ','
                << c.tabu_tenure_dynamic << ','
                << std::fixed << std::setprecision(4)
                << c.aspiration_factor   << ','
                << c.cooling_rate        << ','
                << c.reheat_factor       << ','
                << c.block_parameter     << ','
                << c.adaptation_rate     << ','
                << c.intensify_thresh    << ','
                << c.diversify_thresh    << ','
                << c.penalty_factor      << ','
                << c.neighborhood_cap    << ','
                << c.phase2_freq         << ','
                << c.score               << '\n';
            csv.flush();
        };

        ParamConfig bestCfg = extractConfig();
        bestCfg.score = 1e18;

        std::cout << "\n╔══════════════════════════════════════════════╗\n";
        std::cout <<   "║        HYBRID TABU — PARAMETER TUNING       ║\n";
        std::cout <<   "╠══════════════════════════════════════════════╣\n";
        std::cout << "║ Instances : " << filenames.size()   << "\n";
        std::cout << "║ Runs/inst : " << runsPerInstance     << "\n";
        std::cout << "║ TimeLimit : " << tunTimeLimit << "s\n";
        std::cout << "║ Grid      : " << gridSamples  << " samples\n";
        std::cout << "║ Refine    : " << refineSamples << " samples\n";
        std::cout << "║ LocalIter : " << localIter    << "\n";
        std::cout << "╚══════════════════════════════════════════════╝\n\n";

        // ══════════════════════════════════════════════════════════════
        // PHASE 1 — Grid / Random Search
        // ══════════════════════════════════════════════════════════════
        std::cout << "── Phase 1: Random Grid Search (" << gridSamples << " configs) ──\n";

        for (int t = 0; t < gridSamples; ++t)
        {
            ParamConfig cfg = randomConfig();
            cfg.score = evaluateConfig(cfg, filenames, runsPerInstance, tunTimeLimit);

            std::cout << std::setw(3) << (t + 1) << "/" << gridSamples << " ";
            cfg.print(std::cout);
            std::cout << "\n";

            writeCSV("grid", t + 1, cfg);

            if (cfg.score < bestCfg.score)
            {
                bestCfg = cfg;
                std::cout << "  *** New best! score=" << bestCfg.score << "\n";
            }
        }

        std::cout << "\nBest after Phase 1: score=" << bestCfg.score << "\n";
        bestCfg.print(std::cout); std::cout << "\n\n";

        // ══════════════════════════════════════════════════════════════
        // PHASE 2 — Refine: Random Search quanh best
        // ══════════════════════════════════════════════════════════════
        std::cout << "── Phase 2: Refine Search (" << refineSamples << " configs) ──\n";

        for (int t = 0; t < refineSamples; ++t)
        {
            // 70% perturbation từ best, 30% hoàn toàn random
            std::uniform_real_distribution<double> u(0.0, 1.0);
            ParamConfig cfg = (u(rng) < 0.7) ? perturbConfig(bestCfg) : randomConfig();

            cfg.score = evaluateConfig(cfg, filenames, runsPerInstance, tunTimeLimit);

            std::cout << std::setw(3) << (t + 1) << "/" << refineSamples << " ";
            cfg.print(std::cout);
            std::cout << "\n";

            writeCSV("refine", t + 1, cfg);

            if (cfg.score < bestCfg.score)
            {
                bestCfg = cfg;
                std::cout << "  *** New best! score=" << bestCfg.score << "\n";
            }
        }

        std::cout << "\nBest after Phase 2: score=" << bestCfg.score << "\n";
        bestCfg.print(std::cout); std::cout << "\n\n";

        // ══════════════════════════════════════════════════════════════
        // PHASE 3 — Local Search: flip từng chiều 1 bước
        // ══════════════════════════════════════════════════════════════
        std::cout << "── Phase 3: Local Search (" << localIter << " steps) ──\n";

        bool improved = true;
        int  lsStep   = 0;

        while (improved && lsStep < localIter)
        {
            improved = false;
            ++lsStep;

            // Thử perturbation trên từng dimension
            for (int d = 0; d < 12; ++d)
            {
                ParamConfig candidate = bestCfg;
                switch (d) {
                    case 0:  candidate.tabu_tenure_base    = pickRand(v_tenure_base);   break;
                    case 1:  candidate.tabu_tenure_dynamic = pickRand(v_tenure_dyn);    break;
                    case 2:  candidate.aspiration_factor   = pickRand(v_aspiration);    break;
                    case 3:  candidate.cooling_rate        = pickRand(v_cooling);       break;
                    case 4:  candidate.reheat_factor       = pickRand(v_reheat);        break;
                    case 5:  candidate.block_parameter     = pickRand(v_block);         break;
                    case 6:  candidate.adaptation_rate     = pickRand(v_adapt);         break;
                    case 7:  candidate.intensify_thresh    = pickRand(v_intensify);     break;
                    case 8:  candidate.diversify_thresh    = pickRand(v_diversify);     break;
                    case 9:  candidate.penalty_factor      = pickRand(v_penalty);       break;
                    case 10: candidate.neighborhood_cap    = pickRand(v_nbhd);          break;
                    case 11: candidate.phase2_freq         = pickRand(v_p2freq);        break;
                }
                if (candidate.intensify_thresh >= candidate.diversify_thresh)
                    candidate.diversify_thresh = candidate.intensify_thresh * 3;

                candidate.score = evaluateConfig(candidate, filenames, runsPerInstance, tunTimeLimit);

                writeCSV("local", lsStep * 12 + d, candidate);

                if (candidate.score < bestCfg.score)
                {
                    bestCfg  = candidate;
                    improved = true;
                    std::cout << "  Step " << lsStep << " dim=" << d
                              << " *** Improved! score=" << bestCfg.score << "\n";
                }
            }
        }

        // ══════════════════════════════════════════════════════════════
        // Kết quả cuối
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n╔══════════════════════════════════════════════╗\n";
        std::cout <<   "║            TUNING COMPLETE                  ║\n";
        std::cout <<   "╠══════════════════════════════════════════════╣\n";
        std::cout << "Best score (avg TCT): " << bestCfg.score << "\n";
        bestCfg.print(std::cout);
        std::cout << "\n";

        // In ra đoạn code C++ sẵn sàng paste vào
        std::cout << "\n// ── Paste vào HybridTabuSolver (Hybrid Parameters) ──\n";
        std::cout << "int    tabu_tenure_base    = " << bestCfg.tabu_tenure_base    << ";\n";
        std::cout << "int    tabu_tenure_dynamic = " << bestCfg.tabu_tenure_dynamic << ";\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "double aspiration_factor   = " << bestCfg.aspiration_factor   << ";\n";
        std::cout << "double cooling_rate        = " << bestCfg.cooling_rate        << ";\n";
        std::cout << "double reheat_factor       = " << bestCfg.reheat_factor       << ";\n";
        std::cout << "double block_parameter     = " << bestCfg.block_parameter     << ";\n";
        std::cout << "double adaptation_rate     = " << bestCfg.adaptation_rate     << ";\n";
        std::cout << "int    intensify_thresh    = " << bestCfg.intensify_thresh    << ";\n";
        std::cout << "int    diversify_thresh    = " << bestCfg.diversify_thresh    << ";\n";
        std::cout << "double penalty_factor      = " << bestCfg.penalty_factor      << ";\n";
        std::cout << "int    neighborhood_cap    = " << bestCfg.neighborhood_cap    << ";\n";
        std::cout << "int    phase2_freq         = " << bestCfg.phase2_freq         << ";\n";
        std::cout << "╚══════════════════════════════════════════════╝\n\n";

        if (saveCSV)
        {
            csv.close();
            std::cout << "Tuning log saved to: " << outFile << "\n";
        }

        // Áp dụng best config vào solver
        applyConfig(bestCfg);
        return bestCfg;
    }

    // ─────────────────────────────────────────────
    // Repair
    // ─────────────────────────────────────────────
    void repair()
    {
        int repairIter = 0;
        if (computeTEC_fromCache() <= bound) return;

        std::vector<std::vector<int>> mj(instance.mach + 1);

        while (computeTEC_fromCache() > bound)
        {
            ++repairIter;
            if (repairIter > 10000) break;

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

    // ─────────────────────────────────────────────
    // Run functions
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

        std::cout << "=== HYBRID TABU: " << filename << " ===\n";
        std::cout << "Jobs=" << instance.job << " Machines=" << instance.mach << "\n";
        std::cout << "Bound=" << bound << "\n\n";

        std::vector<double> results, runtimes;

        for (int run = 1; run <= numRuns; ++run)
        {
            X.assign(instance.job + 1, 1);
            ops.resetWeights();
            intensifyCount = diversifyCount = aspirationCount = 0;
            init();

            auto start_time = std::chrono::high_resolution_clock::now();
            int iterCount = hybridTabuSearch(start_time);
            auto end_time = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end_time - start_time;
            double obj = computeTCT_fromCache();

            results.push_back(obj);
            runtimes.push_back(elapsed.count());

            std::cout << "Run " << run
                      << " | TCT=" << obj
                      << " | Iter=" << iterCount
                      << " | Intensify=" << intensifyCount
                      << " | Diversify=" << diversifyCount
                      << " | Aspiration=" << aspirationCount
                      << " | TEC=" << computeTEC_fromCache()
                      << " | Time=" << std::fixed << std::setprecision(3)
                      << elapsed.count() << "s\n";
        }

        double best = *std::min_element(results.begin(), results.end());
        double avg  = std::accumulate(results.begin(), results.end(), 0.0) / results.size();
        std::cout << "\nBest=" << best << " Avg=" << avg << "\n";
    }

    void runBigData(int numRuns)
    {
        std::ofstream outFile("hybrid_tabu_results.csv");
        if (!outFile.is_open())
        {
            std::cerr << "Cannot open output file!\n";
            return;
        }
        outFile << "Folder,Instance,Run,Objective,Runtime,Iterations,Intensify,Diversify,Aspiration\n";
        outFile.flush();

        struct FolderInfo { std::string name; double timeLimit; };
        std::vector<FolderInfo> folders = {
            {"200_2000",  5},
            {"200_5000",  5},
            {"500_10000", 5}
        };

        std::vector<std::string> prefixes = {"N_N", "N_U", "U_N", "U_U"};

        for (const auto& folder : folders)
        {
            std::string basePath = "data/big_data/" + folder.name + "/";
            time_limit = folder.timeLimit;

            std::cout << "\n========== HYBRID TABU: " << folder.name << " ==========\n";

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

                    for (int run = 1; run <= numRuns; ++run)
                    {
                        X.assign(instance.job + 1, 1);
                        ops.resetWeights();
                        intensifyCount = diversifyCount = aspirationCount = 0;
                        init();

                        auto start_time = std::chrono::high_resolution_clock::now();
                        int iterCount = hybridTabuSearch(start_time);
                        auto end_time = std::chrono::high_resolution_clock::now();

                        std::chrono::duration<double> elapsed = end_time - start_time;
                        double obj = computeTCT_fromCache();

                        outFile << folder.name << ',' << instName << ','
                                << run << ',' << (long long)obj << ','
                                << std::fixed << std::setprecision(3) << elapsed.count() << ','
                                << iterCount << ',' << intensifyCount << ','
                                << diversifyCount << ',' << aspirationCount << '\n';
                        outFile.flush();

                        std::cout << "  Run " << run
                                  << " | TCT=" << obj
                                  << " | Iter=" << iterCount
                                  << " | I/D/A=" << intensifyCount << "/"
                                  << diversifyCount << "/" << aspirationCount
                                  << " | TEC=" << computeTEC_fromCache()
                                  << " | Time=" << elapsed.count() << "s\n";
                    }
                }
            }
        }

        outFile.close();
        std::cout << "\nDone! Results saved to hybrid_tabu_results.csv\n";
    }

    void runAllInstances(int numRuns)
    {
        std::ofstream outFile("hybrid_tabu_batch_results.csv");
        if (!outFile.is_open())
        {
            std::cerr << "Cannot open output file!\n";
            return;
        }
        outFile << "Instance,Run,Objective,Runtime,Iterations,Intensify,Diversify,Aspiration\n";
        outFile.flush();

        for (int fileIdx = 1441; fileIdx <= 2160; ++fileIdx)
        {
            std::stringstream ss;
            ss << "data/T_" << fileIdx << ".txt";
            if (!instance.readFromFile(ss.str())) continue;

            int n = instance.job, m = instance.mach;
            time_limit = (n <= 20 && m <= 4) ? 1.0 :
                         (n <= 50 && m <= 6) ? 2.0 : 5.0;

            upper_bound = (int)calculateUpper();
            lower_bound = (int)calculateLower();
            bound = (int)((1.0 - instance.ctrl_factor) * lower_bound +
                           instance.ctrl_factor * upper_bound);

            for (int run = 1; run <= numRuns; ++run)
            {
                X.assign(instance.job + 1, 1);
                ops.resetWeights();
                intensifyCount = diversifyCount = aspirationCount = 0;
                init();

                auto start_time = std::chrono::high_resolution_clock::now();
                int iterCount = hybridTabuSearch(start_time);
                auto end_time = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> elapsed = end_time - start_time;
                double obj = computeTCT_fromCache();

                outFile << fileIdx << ',' << run << ',' << (long long)obj << ','
                        << std::fixed << std::setprecision(3) << elapsed.count() << ','
                        << iterCount << ',' << intensifyCount << ','
                        << diversifyCount << ',' << aspirationCount << '\n';
                outFile.flush();

                std::cout << "[" << fileIdx << "/2160] Run " << run
                          << " | TCT=" << obj
                          << " | I/D/A=" << intensifyCount << "/"
                          << diversifyCount << "/" << aspirationCount
                          << " | Time=" << elapsed.count() << "s\n";
            }
        }
        outFile.close();
        std::cout << "\nDone! Results saved to hybrid_tabu_batch_results.csv\n";
    }
};

#endif