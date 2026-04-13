#ifndef LNS_SOLVER_H
#define LNS_SOLVER_H

#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>

#include "Instance.h"

// ═══════════════════════════════════════════════════════════════════
// Large Neighborhood Search (ALNS) solver cho bài
//   Parallel Machine Scheduling with Energy Budget
//   - Minimize Total Completion Time (TCT)
//   - s.t. Total Energy Cost (TEC) ≤ bound
// ═══════════════════════════════════════════════════════════════════
class LNSSolver
{
public:
    std::mt19937 rng{std::random_device{}()};
    Instance instance;

    std::vector<int> X;                // X[j] = machine chạy job j
    std::vector<long long> machLoad;   // tổng proc_time trên mỗi máy
    std::vector<int> machJobCount;     // số job trên mỗi máy

    int upper_bound, lower_bound, bound;

    // ─────────────────────────────────────────────
    // LNS parameters
    // ─────────────────────────────────────────────
    double destroy_min_ratio = 0.10;   // phá tối thiểu 10% job
    double destroy_max_ratio = 0.40;   // phá tối đa   40% job
    double penalty_factor    = 50.0;
    double time_limit        = 5.0;

    // SA acceptance
    double T                 = 1000.0;
    double cooling_rate      = 0.9975;
    double reheat_factor     = 2.0;
    int    stuck_limit       = 200;    // reheat sau bấy nhiêu iter không cải thiện best

    // ALNS adaptive
    static constexpr int NUM_DESTROY = 4;
    static constexpr int NUM_REPAIR  = 2;
    std::vector<double> destroyWeight, destroyScore;
    std::vector<int>    destroyUsage;
    std::vector<double> repairWeight, repairScore;
    std::vector<int>    repairUsage;
    double adaptation_rate = 0.2;
    int    segment_length  = 30;

    // Statistics
    int acceptCount     = 0;
    int improveCount    = 0;
    int globalBestCount = 0;

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

    // Bỏ qua job có X[j] == 0 (đang “unassigned” sau destroy)
    void rebuildCachePartial()
    {
        int m = instance.mach;
        machLoad.assign(m + 1, 0);
        machJobCount.assign(m + 1, 0);
        for (int j = 1; j <= instance.job; ++j)
        {
            if (X[j] == 0) continue;
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
    // ALNS adaptive weights
    // ─────────────────────────────────────────────
    void resetAdaptive()
    {
        destroyWeight.assign(NUM_DESTROY, 1.0);
        destroyScore .assign(NUM_DESTROY, 0.0);
        destroyUsage .assign(NUM_DESTROY, 0);
        repairWeight .assign(NUM_REPAIR,  1.0);
        repairScore  .assign(NUM_REPAIR,  0.0);
        repairUsage  .assign(NUM_REPAIR,  0);
    }

    int rouletteSelect(const std::vector<double>& w)
    {
        double sum = std::accumulate(w.begin(), w.end(), 0.0);
        if (sum <= 0.0) return rng() % w.size();
        std::uniform_real_distribution<double> dist(0.0, sum);
        double r = dist(rng), acc = 0.0;
        for (int i = 0; i < (int)w.size(); ++i)
        {
            acc += w[i];
            if (r <= acc) return i;
        }
        return (int)w.size() - 1;
    }

    void updateAdaptive()
    {
        for (int i = 0; i < NUM_DESTROY; ++i)
        {
            if (destroyUsage[i] > 0)
            {
                double avg = destroyScore[i] / destroyUsage[i];
                destroyWeight[i] = (1.0 - adaptation_rate) * destroyWeight[i]
                                 + adaptation_rate * avg;
            }
            destroyWeight[i] = std::max(destroyWeight[i], 0.05);
            destroyScore[i]  = 0.0;
            destroyUsage[i]  = 0;
        }
        for (int i = 0; i < NUM_REPAIR; ++i)
        {
            if (repairUsage[i] > 0)
            {
                double avg = repairScore[i] / repairUsage[i];
                repairWeight[i] = (1.0 - adaptation_rate) * repairWeight[i]
                                + adaptation_rate * avg;
            }
            repairWeight[i] = std::max(repairWeight[i], 0.05);
            repairScore[i]  = 0.0;
            repairUsage[i]  = 0;
        }
    }

    // ─────────────────────────────────────────────
    // DESTROY operators
    //   Trả về danh sách job đã bị gỡ. Job bị gỡ đánh dấu X[j] = 0.
    //   Cache KHÔNG được maintain trong destroy — sẽ rebuild ở repair.
    // ─────────────────────────────────────────────

    // 1) Random removal
    std::vector<int> destroyRandom(int q)
    {
        std::vector<int> all(instance.job);
        std::iota(all.begin(), all.end(), 1);
        std::shuffle(all.begin(), all.end(), rng);
        std::vector<int> removed(all.begin(), all.begin() + q);
        for (int j : removed) X[j] = 0;
        return removed;
    }

    // 2) Worst removal: gỡ job có “đóng góp” lớn nhất vào fitness
    //    contribution ~ proc_time[j] * jobs_on_machine * (1 + unit_cost)
    std::vector<int> destroyWorst(int q)
    {
        std::vector<int> cnt(instance.mach + 1, 0);
        for (int j = 1; j <= instance.job; ++j) cnt[X[j]]++;

        std::vector<std::pair<double,int>> score;
        score.reserve(instance.job);
        std::uniform_real_distribution<double> noise(0.8, 1.2);
        for (int j = 1; j <= instance.job; ++j)
        {
            double s = (double)instance.proc_time[j] * cnt[X[j]]
                       * (1.0 + instance.unit_cost[X[j] - 1]);
            s *= noise(rng);   // jitter để tránh deterministic
            score.push_back({s, j});
        }
        std::sort(score.begin(), score.end(), std::greater<>());

        std::vector<int> removed;
        removed.reserve(q);
        for (int i = 0; i < q; ++i)
        {
            removed.push_back(score[i].second);
            X[score[i].second] = 0;
        }
        return removed;
    }

    // 3) Machine removal: gỡ toàn bộ job trên 1-2 máy (chọn ngẫu nhiên)
    std::vector<int> destroyMachine(int q)
    {
        std::vector<int> removed;
        std::vector<int> machOrder(instance.mach);
        std::iota(machOrder.begin(), machOrder.end(), 1);
        std::shuffle(machOrder.begin(), machOrder.end(), rng);

        for (int m : machOrder)
        {
            for (int j = 1; j <= instance.job && (int)removed.size() < q; ++j)
            {
                if (X[j] == m)
                {
                    removed.push_back(j);
                    X[j] = 0;
                }
            }
            if ((int)removed.size() >= q) break;
        }
        // pad bằng random nếu chưa đủ q
        if ((int)removed.size() < q)
        {
            std::vector<int> pool;
            for (int j = 1; j <= instance.job; ++j)
                if (X[j] != 0) pool.push_back(j);
            std::shuffle(pool.begin(), pool.end(), rng);
            for (int j : pool)
            {
                if ((int)removed.size() >= q) break;
                removed.push_back(j);
                X[j] = 0;
            }
        }
        return removed;
    }

    // 4) Related (Shaw) removal: gỡ job có proc_time tương tự job seed
    std::vector<int> destroyRelated(int q)
    {
        std::uniform_int_distribution<int> jd(1, instance.job);
        int seed = jd(rng);
        int p0 = instance.proc_time[seed];

        std::vector<std::pair<int,int>> diff;
        diff.reserve(instance.job);
        for (int j = 1; j <= instance.job; ++j)
            diff.push_back({std::abs(instance.proc_time[j] - p0), j});
        std::sort(diff.begin(), diff.end());

        std::vector<int> removed;
        removed.reserve(q);
        for (int i = 0; i < q; ++i)
        {
            removed.push_back(diff[i].second);
            X[diff[i].second] = 0;
        }
        return removed;
    }

    std::vector<int> applyDestroy(int op, int q)
    {
        switch (op)
        {
            case 0: return destroyRandom(q);
            case 1: return destroyWorst(q);
            case 2: return destroyMachine(q);
            case 3: return destroyRelated(q);
        }
        return destroyRandom(q);
    }

    // ─────────────────────────────────────────────
    // REPAIR operators
    // ─────────────────────────────────────────────

    // Chi phí chèn job vào máy m (delta trên partial solution)
    double insertCost(int job, int m)
    {
        // delta TCT xấp xỉ tuyến tính theo số job đã có trên máy
        long long deltaTCT = (long long)instance.proc_time[job]
                             * (machJobCount[m] + 1);
        long long deltaTEC = (long long)instance.proc_time[job]
                             * instance.unit_cost[m - 1];
        long long newTEC = computeTEC_fromCache() + deltaTEC;

        double pen = 0.0;
        if (newTEC > bound)
        {
            double d = (double)(newTEC - bound);
            pen = penalty_factor * d * d;
        }
        return (double)deltaTCT + pen;
    }

    // 1) Greedy insertion — random order, mỗi job chọn máy tốt nhất
    void repairGreedy(std::vector<int>& removed)
    {
        rebuildCachePartial();
        std::shuffle(removed.begin(), removed.end(), rng);
        for (int job : removed)
        {
            int bestM = 1;
            double bestC = std::numeric_limits<double>::infinity();
            for (int m = 1; m <= instance.mach; ++m)
            {
                double c = insertCost(job, m);
                if (c < bestC) { bestC = c; bestM = m; }
            }
            X[job] = bestM;
            machLoad[bestM]     += instance.proc_time[job];
            machJobCount[bestM] += 1;
        }
    }

    // 2) Regret-2 insertion — chèn trước job có regret lớn nhất
    void repairRegret(std::vector<int>& removed)
    {
        rebuildCachePartial();
        std::vector<int> pool = removed;

        while (!pool.empty())
        {
            int    bestJobIdx  = -1;
            int    bestMachine = -1;
            double bestRegret  = -std::numeric_limits<double>::infinity();
            double bestInsC    = std::numeric_limits<double>::infinity();

            for (int i = 0; i < (int)pool.size(); ++i)
            {
                int job = pool[i];
                double b1 = std::numeric_limits<double>::infinity();
                double b2 = std::numeric_limits<double>::infinity();
                int    bm1 = 1;

                for (int m = 1; m <= instance.mach; ++m)
                {
                    double c = insertCost(job, m);
                    if (c < b1) { b2 = b1; b1 = c; bm1 = m; }
                    else if (c < b2) { b2 = c; }
                }

                double regret = (b2 == std::numeric_limits<double>::infinity())
                                ? 0.0 : (b2 - b1);

                // tie-break: regret cao hơn; bằng thì chọn chi phí chèn nhỏ hơn
                if (regret > bestRegret ||
                    (regret == bestRegret && b1 < bestInsC))
                {
                    bestRegret  = regret;
                    bestJobIdx  = i;
                    bestMachine = bm1;
                    bestInsC    = b1;
                }
            }

            int job = pool[bestJobIdx];
            X[job] = bestMachine;
            machLoad[bestMachine]     += instance.proc_time[job];
            machJobCount[bestMachine] += 1;
            pool.erase(pool.begin() + bestJobIdx);
        }
    }

    void applyRepair(int op, std::vector<int>& removed)
    {
        if (op == 0) repairGreedy(removed);
        else         repairRegret(removed);
    }

    // ─────────────────────────────────────────────
    // Redistribution: sort máy giảm dần theo load
    // ─────────────────────────────────────────────
    void redistributionBasedOnCost()
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
    // Initial construction (giữ nguyên từ bản cũ)
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

            redistributionBasedOnCost();
            cost = computeTEC_fromCache();
            ++iter;
            if (iter > instance.job) break;
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // LNS main loop
    // ═══════════════════════════════════════════════════════════════════
    int lnsSearch(std::chrono::high_resolution_clock::time_point start_time)
    {
        std::vector<int>       Xbest    = X;
        std::vector<long long> loadBest = machLoad;
        std::vector<int>       cntBest  = machJobCount;

        double currentFitness = computeFitness_fromCache();
        double bestFitness    = currentFitness;

        resetAdaptive();
        acceptCount = improveCount = globalBestCount = 0;

        // Kích thước destroy
        int qMin = std::max(2, (int)(destroy_min_ratio * instance.job));
        int qMax = std::max(qMin + 1, (int)(destroy_max_ratio * instance.job));

        // Nhiệt độ SA khởi tạo theo fitness ban đầu
        T = 0.05 * currentFitness / std::log(2.0);
        double Tmin = std::max(1.0, 1e-4 * currentFitness);

        std::uniform_real_distribution<double> rdist(0.0, 1.0);

        // Snapshot trước mỗi iteration để rollback
        std::vector<int>       Xprev;
        std::vector<long long> loadPrev;
        std::vector<int>       cntPrev;

        int iteration = 0;
        int sinceBest = 0;

        while (true)
        {
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration<double>(now - start_time).count() >= time_limit)
                break;

            ++iteration;

            // Lưu snapshot
            Xprev    = X;
            loadPrev = machLoad;
            cntPrev  = machJobCount;

            // Chọn operator + kích thước destroy
            std::uniform_int_distribution<int> qDist(qMin, qMax);
            int q = qDist(rng);
            int dOp = rouletteSelect(destroyWeight);
            int rOp = rouletteSelect(repairWeight);
            destroyUsage[dOp]++;
            repairUsage[rOp]++;

            // Destroy + Repair
            std::vector<int> removed = applyDestroy(dOp, q);
            applyRepair(rOp, removed);
            redistributionBasedOnCost();

            double newFitness = computeFitness_fromCache();
            double reward = 0.0;
            bool   accept = false;

            // Acceptance (SA style)
            if (newFitness < bestFitness - 1e-9)
            {
                bestFitness = newFitness;
                Xbest    = X;
                loadBest = machLoad;
                cntBest  = machJobCount;
                reward = 10.0;
                ++globalBestCount;
                ++improveCount;
                ++acceptCount;
                accept = true;
                sinceBest = 0;
            }
            else if (newFitness < currentFitness - 1e-9)
            {
                reward = 4.0;
                ++improveCount;
                ++acceptCount;
                accept = true;
            }
            else
            {
                double delta = newFitness - currentFitness;
                double prob  = std::exp(-delta / std::max(T, 1e-9));
                if (rdist(rng) < prob)
                {
                    reward = 1.0;
                    ++acceptCount;
                    accept = true;
                }
            }

            if (accept)
            {
                currentFitness = newFitness;
            }
            else
            {
                // Rollback
                X = std::move(Xprev);
                machLoad = std::move(loadPrev);
                machJobCount = std::move(cntPrev);
            }

            destroyScore[dOp] += reward;
            repairScore[rOp]  += reward;

            // Cooling
            T *= cooling_rate;
            if (T < Tmin) T = Tmin;

            // Cập nhật trọng số định kỳ
            if (iteration % segment_length == 0)
                updateAdaptive();

            // Diversification: reheat + quay về best khi stuck
            ++sinceBest;
            if (sinceBest > stuck_limit)
            {
                T *= reheat_factor;
                sinceBest = 0;
                X = Xbest;
                machLoad = loadBest;
                machJobCount = cntBest;
                currentFitness = bestFitness;
            }
        }

        // Khôi phục best
        X = Xbest;
        machLoad = loadBest;
        machJobCount = cntBest;

        repair();
        return iteration;
    }

    // ─────────────────────────────────────────────
    // Energy repair (giữ nguyên)
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
    // Bounds (giữ nguyên)
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
    // Run functions (giữ nguyên shape với bản tabu)
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

        std::cout << "=== LNS: " << filename << " ===\n";
        std::cout << "Jobs=" << instance.job << " Machines=" << instance.mach << "\n";
        std::cout << "Bound=" << bound << "\n\n";

        std::vector<double> results, runtimes;
        for (int run = 1; run <= numRuns; ++run)
        {
            X.assign(instance.job + 1, 1);
            init();

            auto start_time = std::chrono::high_resolution_clock::now();
            int iterCount = lnsSearch(start_time);
            auto end_time = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end_time - start_time;
            double obj = computeTCT_fromCache();

            results.push_back(obj);
            runtimes.push_back(elapsed.count());

            std::cout << "Run " << run
                      << " | TCT=" << obj
                      << " | Iter=" << iterCount
                      << " | Accept=" << acceptCount
                      << " | Improve=" << improveCount
                      << " | GBest=" << globalBestCount
                      << " | Time=" << std::fixed << std::setprecision(3)
                      << elapsed.count() << "s\n";
        }
        double best = *std::min_element(results.begin(), results.end());
        double avg  = std::accumulate(results.begin(), results.end(), 0.0) / results.size();
        std::cout << "\nBest=" << best << " Avg=" << avg << "\n";
    }

    void runBigData(int numRuns)
    {
        std::ofstream outFile("lns_results.csv");
        if (!outFile.is_open())
        {
            std::cerr << "Cannot open output file!\n";
            return;
        }
        outFile << "Folder,Instance,Run,Objective,Runtime,Iterations,Accept,Improve,GBest\n";
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

            std::cout << "\n========== LNS: " << folder.name << " ==========\n";
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
                        init();

                        auto start_time = std::chrono::high_resolution_clock::now();
                        int iterCount = lnsSearch(start_time);
                        auto end_time = std::chrono::high_resolution_clock::now();

                        std::chrono::duration<double> elapsed = end_time - start_time;
                        double obj = computeTCT_fromCache();

                        outFile << folder.name << ',' << instName << ','
                                << run << ',' << (long long)obj << ','
                                << std::fixed << std::setprecision(3) << elapsed.count() << ','
                                << iterCount << ',' << acceptCount << ','
                                << improveCount << ',' << globalBestCount << '\n';
                        outFile.flush();

                        std::cout << "  Run " << run
                                  << " | TCT=" << obj
                                  << " | Iter=" << iterCount
                                  << " | A/I/G=" << acceptCount << "/"
                                  << improveCount << "/" << globalBestCount
                                  << " | Time=" << elapsed.count() << "s\n";
                    }
                }
            }
        }
        outFile.close();
        std::cout << "\nDone! Results saved to lns_results.csv\n";
    }

    void runAllInstances(int numRuns)
    {
        std::ofstream outFile("lns_batch_results.csv");
        if (!outFile.is_open())
        {
            std::cerr << "Cannot open output file!\n";
            return;
        }
        outFile << "Instance,Run,Objective,Runtime,Iterations,Accept,Improve,GBest\n";
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
                init();

                auto start_time = std::chrono::high_resolution_clock::now();
                int iterCount = lnsSearch(start_time);
                auto end_time = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> elapsed = end_time - start_time;
                double obj = computeTCT_fromCache();

                outFile << fileIdx << ',' << run << ',' << (long long)obj << ','
                        << std::fixed << std::setprecision(3) << elapsed.count() << ','
                        << iterCount << ',' << acceptCount << ','
                        << improveCount << ',' << globalBestCount << '\n';
                outFile.flush();

                std::cout << "[" << fileIdx << "/2160] Run " << run
                          << " | TCT=" << obj
                          << " | A/I/G=" << acceptCount << "/"
                          << improveCount << "/" << globalBestCount
                          << " | Time=" << elapsed.count() << "s\n";
            }
        }
        outFile.close();
        std::cout << "\nDone! Results saved to lns_batch_results.csv\n";
    }
};

#endif