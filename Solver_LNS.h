#ifndef SOLVER_LNS_H
#define SOLVER_LNS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <limits>
#include <iomanip>
#include "Instance.h"

struct LNSParams
{
    int max_iter = 500000;
    int stagnation_limit = 2000;
    double destroy_ratio = 0.30;
    double perturb_ratio = 0.50;
    int destroy_min = 2;
    int destroy_max = 50;
    int repair_top_k = 3;
    double repair_p0 = 0.65;
    double repair_p1 = 0.88;
    double penalty_factor = 50.0;
    // ALNS
    double r_weight = 0.1;
    double sigma1 = 10.0;
    double sigma2 = 4.0;
    double sigma3 = 1.0;
    bool verbose = false;
};

class SolverLNS
{
public:
    Instance instance;
    LNSParams params;
    std::mt19937 rng{std::random_device{}()};
    std::vector<int> Xbest;
    long long bound = 0;
    long long bestTCT = 0;

    static const int ND = 7, NR = 2; // +1 destroy operator
    double wD[ND], wR[NR], sD[ND], sR[NR];
    int uD[ND], uR[NR];

    explicit SolverLNS(const LNSParams &p = LNSParams{}) : params(p) {}

    bool loadInstance(const std::string &fn)
    {
        if (!instance.readFromFile(fn))
        {
            std::cerr << "[ALNS] Cannot read: " << fn << "\n";
            return false;
        }
        return true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Objective
    // ─────────────────────────────────────────────────────────────────────────
    long long calcTCT(const std::vector<int> &sol) const
    {
        long long total = 0;
        for (int m = 1; m <= instance.mach; ++m)
        {
            long long t = 0;
            for (int j = 1; j <= instance.job; ++j)
                if (sol[j] == m)
                {
                    t += instance.proc_time[j];
                    total += t;
                }
        }
        return total;
    }

    long long calcTEC(const std::vector<int> &sol) const
    {
        std::vector<long long> ld(instance.mach + 1, 0);
        for (int j = 1; j <= instance.job; ++j)
            if (sol[j] > 0)
                ld[sol[j]] += instance.proc_time[j];
        long long u = 0;
        for (int m = 1; m <= instance.mach; ++m)
            u += (long long)(ld[m] * instance.unit_cost[m - 1]);
        return u;
    }

    double calcFit(const std::vector<int> &sol) const
    {
        long long tct = calcTCT(sol), tec = calcTEC(sol);
        if (tec <= bound)
            return (double)tct;
        double ex = (double)(tec - bound);
        return (double)tct + params.penalty_factor * ex * ex;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Bound & Init
    // ─────────────────────────────────────────────────────────────────────────
    long long calcUpper() const
    {
        std::vector<long long> xt(instance.mach, 0);
        for (int p : instance.proc_time)
        {
            int mp = (int)(std::min_element(xt.begin(), xt.end()) - xt.begin());
            xt[mp] += p;
        }
        long long U = 0;
        for (int i = 0; i < instance.mach; ++i)
            U += (long long)(xt[i] * instance.unit_cost[i]);
        return U;
    }

    long long calcLower() const
    {
        long long U = 0;
        for (int i = 0; i <= instance.job; ++i)
            U += (long long)(instance.proc_time[i] * instance.unit_cost[0]);
        return U;
    }

    std::vector<int> runInitISA()
    {
        std::vector<int> X(instance.job + 1, 1);
        auto tec = [&]()
        {
            std::vector<long long> ld(instance.mach + 1, 0);
            for (int j = 1; j <= instance.job; ++j)
                ld[X[j]] += instance.proc_time[j];
            long long u = 0;
            for (int m = 1; m <= instance.mach; ++m)
                u += (long long)(ld[m] * instance.unit_cost[m - 1]);
            return u;
        };
        auto redist = [&]()
        {
            std::vector<int> ct(instance.mach + 1, 0);
            std::vector<std::vector<int>> mj(instance.mach + 1);
            for (int i = 0; i <= instance.job; ++i)
            {
                ct[X[i]] += instance.proc_time[i];
                mj[X[i]].push_back(i);
            }
            std::vector<int> idx(instance.mach + 1);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b)
                      { return ct[a] > ct[b]; });
            std::vector<std::vector<int>> nj(instance.mach + 1);
            for (int k = 0; k <= instance.mach; ++k)
                nj[k] = mj[idx[k]];
            for (int mac = 0; mac < instance.mach; ++mac)
                for (int job : nj[mac])
                    X[job] = mac + 1;
        };
        double cost = (double)tec();
        long long theta = instance.mach, iter = 1;
        while (true)
        {
            theta = instance.mach;
            while (instance.proc_time[iter] * (instance.unit_cost[theta - 1] - instance.unit_cost[1]) > bound - cost && theta > 1)
                --theta;
            if (theta == 1)
                break;
            X[iter] = theta;
            cost = (double)tec();
            iter++;
            redist();
            if (iter == instance.job)
                break;
        }
        return X;
    }

    std::vector<int> runInitGreedy()
    {
        int n = instance.job, m = instance.mach;
        std::vector<int> X(n + 1, 0);
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 1);
        std::sort(order.begin(), order.end(), [&](int a, int b)
                  { return instance.proc_time[a] > instance.proc_time[b]; });
        std::vector<long long> completion(m + 1, 0);
        std::vector<long long> load(m + 1, 0);
        long long curTEC = 0;
        for (int j : order)
        {
            long long p = instance.proc_time[j];
            int bestM = -1;
            long long bestDelta = LLONG_MAX;
            for (int mac = 1; mac <= m; ++mac)
            {
                long long dTEC = (long long)(p * instance.unit_cost[mac - 1]);
                if (curTEC + dTEC > bound)
                    continue;
                long long delta = completion[mac] + p;
                if (delta < bestDelta)
                {
                    bestDelta = delta;
                    bestM = mac;
                }
            }
            if (bestM < 0)
            {
                bestM = 1;
                for (int mac = 2; mac <= m; ++mac)
                    if (instance.unit_cost[mac - 1] < instance.unit_cost[bestM - 1])
                        bestM = mac;
            }
            X[j] = bestM;
            completion[bestM] += p;
            load[bestM] += p;
            curTEC += (long long)(p * instance.unit_cost[bestM - 1]);
        }
        return X;
    }

    std::vector<int> runInit()
    {
        std::vector<int> X1 = runInitISA();
        std::vector<int> X2 = runInitGreedy();
        return calcFit(X1) <= calcFit(X2) ? X1 : X2;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PREFIX SUM CACHE — O(n·m) build, O(1) delta lookup
    //
    // prefM[m][j] = sum of proc_time[k] for k in [1..j-1] where sol[k]==m
    // cntAfterM[m][j] = count of k in [j+1..n] where sol[k]==m
    //
    // After building, delta for moving job j from src→dst:
    //   dTCT = (prefM[dst][j] + p*cntAfterM[dst][j])
    //        - (prefM[src][j] + p*cntAfterM[src][j])
    // ─────────────────────────────────────────────────────────────────────────
    struct PrefixCache
    {
        // prefM[m][j]     : sum proc_time[k], k<j, sol[k]==m  (index 0..n)
        // cntAfterM[m][j] : count k>j, sol[k]==m              (index 0..n+1)
        std::vector<std::vector<long long>> prefM;
        std::vector<std::vector<long long>> cntAfterM;
        int n, m;
    };

    PrefixCache buildCache(const std::vector<int> &sol) const
    {
        int n = instance.job, m = instance.mach;
        PrefixCache C;
        C.n = n;
        C.m = m;
        C.prefM.assign(m + 1, std::vector<long long>(n + 2, 0));
        C.cntAfterM.assign(m + 1, std::vector<long long>(n + 2, 0));

        // forward pass: prefix sums
        for (int j = 1; j <= n; ++j)
            for (int mac = 1; mac <= m; ++mac)
            {
                C.prefM[mac][j] = C.prefM[mac][j - 1];
                if (sol[j - 1] == mac && j > 1)
                    ; // handled below
            }
        // rebuild more clearly
        for (int mac = 1; mac <= m; ++mac)
        {
            C.prefM[mac][1] = 0;
            for (int j = 2; j <= n + 1; ++j)
                C.prefM[mac][j] = C.prefM[mac][j - 1] +
                                  (sol[j - 1] == mac ? instance.proc_time[j - 1] : 0);
        }
        // backward pass: suffix counts
        for (int mac = 1; mac <= m; ++mac)
        {
            C.cntAfterM[mac][n + 1] = 0;
            for (int j = n; j >= 1; --j)
                C.cntAfterM[mac][j] = C.cntAfterM[mac][j + 1] +
                                      (sol[j + 1 <= n && sol[j + 1] == mac ? j + 1 : 0] == mac ? 1 : 0);
        }
        // simpler suffix count rebuild
        for (int mac = 1; mac <= m; ++mac)
        {
            C.cntAfterM[mac][n] = 0;
            C.cntAfterM[mac][n + 1] = 0;
            for (int j = n - 1; j >= 1; --j)
                C.cntAfterM[mac][j] = C.cntAfterM[mac][j + 1] + (sol[j + 1] == mac ? 1 : 0);
        }
        return C;
    }

    // O(1) delta TCT for relocating job j from sol[j] to dst
    inline long long deltaTCT_fast(const PrefixCache &C, const std::vector<int> &sol, int j, int dst) const
    {
        int src = sol[j];
        long long p = instance.proc_time[j];
        // prefM[mac][j] = sum of proc_time[k] for k < j on machine mac
        long long prefSrc = C.prefM[src][j];
        long long prefDst = C.prefM[dst][j];
        // cntAfterM[mac][j] = count of k > j on machine mac
        long long caSrc = C.cntAfterM[src][j];
        long long caDst = C.cntAfterM[dst][j];
        return (prefDst + p * caDst) - (prefSrc + p * caSrc);
    }

    // O(1) delta TCT for swapping jobs a and b (a < b, different machines)
    long long deltaSwap_fast(const PrefixCache &C, const std::vector<int> &sol, int a, int b) const
    {
        int ma = sol[a], mb = sol[b];
        long long pa = instance.proc_time[a], pb = instance.proc_time[b];

        // Prefix sums at positions a and b, for both machines
        long long prefA_ma = C.prefM[ma][a]; // sum p[k], k<a, on ma
        long long prefA_mb = C.prefM[mb][a]; // sum p[k], k<a, on mb
        long long prefB_ma = C.prefM[ma][b]; // sum p[k], k<b, on ma (includes pa)
        long long prefB_mb = C.prefM[mb][b]; // sum p[k], k<b, on mb (includes pb)

        // cntAfterM[mac][j] doesn't include j itself, counts k > j
        long long cntAfterB_ma = C.cntAfterM[ma][b];
        long long cntAfterB_mb = C.cntAfterM[mb][b];

        // Jobs strictly between a and b on each machine
        // = prefB_ma - prefA_ma - pa  →  converted to count via divide by avg? No.
        // We need COUNT, not sum. So we need cntBetween separately.
        // Use: cntBetween_ma = cntAfterM[ma][a] - cntAfterM[ma][b]  (minus b itself if on ma)
        long long cntAB_ma = C.cntAfterM[ma][a] - C.cntAfterM[ma][b] - (sol[b] == ma ? 1 : 0);
        long long cntAB_mb = C.cntAfterM[mb][a] - C.cntAfterM[mb][b] - (sol[b] == mb ? 1 : 0);

        // Contribution change:
        // When a moves from ma to mb: loses prefA_ma contribution to all jobs after a on ma,
        //   gains prefA_mb contribution to all after a on mb.
        // This is the same formula as the original deltaSwap but with O(1) lookups.
        return (prefA_mb - prefA_ma) + (prefB_ma - prefB_mb) - pa * cntAB_ma + pa * cntAB_mb + (pb - pa) * cntAfterB_ma + (pa - pb) * cntAfterB_mb;
    }

    // Update cache after moving job j from old_mac to new_mac — O(n)
    void updateCache(PrefixCache &C, std::vector<int> &sol, int j, int new_mac) const
    {
        int old_mac = sol[j];
        long long p = instance.proc_time[j];
        int n = instance.job;

        // Update prefM: for positions k > j, subtract p from old_mac, add p to new_mac
        for (int k = j + 1; k <= n + 1; ++k)
        {
            C.prefM[old_mac][k] -= p;
            C.prefM[new_mac][k] += p;
        }
        // Update cntAfterM: for positions k < j, decrement old_mac, increment new_mac
        for (int k = 1; k < j; ++k)
        {
            C.cntAfterM[old_mac][k] -= 1;
            C.cntAfterM[new_mac][k] += 1;
        }
        sol[j] = new_mac;
    }

    // Update cache after swapping jobs a and b — O(n)
    void updateCacheSwap(PrefixCache &C, std::vector<int> &sol, int a, int b) const
    {
        // a < b, sol[a] != sol[b]
        int ma = sol[a], mb = sol[b];
        long long pa = instance.proc_time[a], pb = instance.proc_time[b];
        int n = instance.job;

        // Relocate a: ma → mb
        // For k > a: prefM[ma][k] -= pa, prefM[mb][k] += pa
        // For k < a: cntAfterM[ma][k] -= 1, cntAfterM[mb][k] += 1
        for (int k = a + 1; k <= n + 1; ++k)
        {
            C.prefM[ma][k] -= pa;
            C.prefM[mb][k] += pa;
        }
        for (int k = 1; k < a; ++k)
        {
            C.cntAfterM[ma][k] -= 1;
            C.cntAfterM[mb][k] += 1;
        }

        // Relocate b: mb → ma
        for (int k = b + 1; k <= n + 1; ++k)
        {
            C.prefM[mb][k] -= pb;
            C.prefM[ma][k] += pb;
        }
        for (int k = 1; k < b; ++k)
        {
            C.cntAfterM[mb][k] -= 1;
            C.cntAfterM[ma][k] += 1;
        }

        sol[a] = mb;
        sol[b] = ma;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FAST LOCAL SEARCH — O(n²) per pass using prefix cache
    // ─────────────────────────────────────────────────────────────────────────

    // Relocate pass: O(n·m) per pass (each job: m machines, each O(1))
    bool ls_relocate_fast(std::vector<int> &sol, PrefixCache &C)
    {
        long long curTEC = calcTEC(sol);
        bool improved = false;

        for (int j = 1; j <= instance.job; ++j)
        {
            int src = sol[j];
            int bestDst = src;
            long long bestDelta = 0;
            long long p = instance.proc_time[j];

            for (int dst = 1; dst <= instance.mach; ++dst)
            {
                if (dst == src)
                    continue;
                long long dTEC = (long long)(p * (instance.unit_cost[dst - 1] - instance.unit_cost[src - 1]));
                if (curTEC + dTEC > bound)
                    continue;

                long long dTCT = deltaTCT_fast(C, sol, j, dst);
                if (dTCT < bestDelta)
                {
                    bestDelta = dTCT;
                    bestDst = dst;
                }
            }
            if (bestDst != src)
            {
                curTEC += (long long)(p * (instance.unit_cost[bestDst - 1] - instance.unit_cost[src - 1]));
                updateCache(C, sol, j, bestDst);
                improved = true;
            }
        }
        return improved;
    }

    // Swap pass: O(n²) per pass — only pairs on different machines
    // Groups jobs by machine first to skip same-machine pairs efficiently
    bool ls_swap_fast(std::vector<int> &sol, PrefixCache &C)
    {
        long long curTEC = calcTEC(sol);
        bool improved = false;
        int n = instance.job;

        for (int a = 1; a <= n - 1; ++a)
        {
            for (int b = a + 1; b <= n; ++b)
            {
                if (sol[a] == sol[b])
                    continue; // skip same machine

                int ma = sol[a], mb = sol[b];
                long long pa = instance.proc_time[a], pb = instance.proc_time[b];
                long long dTEC = (long long)((pb - pa) * instance.unit_cost[ma - 1] + (pa - pb) * instance.unit_cost[mb - 1]);
                if (curTEC + dTEC > bound)
                    continue;

                long long dTCT = deltaSwap_fast(C, sol, a, b);
                if (dTCT < 0)
                {
                    updateCacheSwap(C, sol, a, b);
                    curTEC += dTEC;
                    improved = true;
                }
            }
        }
        return improved;
    }

    void localSearch(std::vector<int> &sol, int maxPasses = INT_MAX)
    {
        // PrefixCache C = buildCache(sol);
        // bool any = true;
        // int passes = 0;
        // while (any && passes < maxPasses)
        // {
        //     any = ls_relocate_fast(sol, C);
        //     any |= ls_swap_fast(sol, C);
        //     ++passes;
        // }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PERTURBATION OPERATORS (4 ISA operators + ruin-recreate)
    // ─────────────────────────────────────────────────────────────────────────
    void applyOp(int op, std::vector<int> &sol, int blockSize)
    {
        int n = instance.job, m = instance.mach;
        std::uniform_int_distribution<int> rJob(1, n);
        std::uniform_int_distribution<int> rMach(1, m);

        switch (op)
        {
        case 0: // O1: change machine of 1 job
        {
            int j = rJob(rng);
            int dst;
            do
            {
                dst = rMach(rng);
            } while (dst == sol[j]);
            sol[j] = dst;
            break;
        }
        case 1: // O2: swap 2 random jobs
        {
            int a = rJob(rng), b;
            do
            {
                b = rJob(rng);
            } while (b == a);
            std::swap(sol[a], sol[b]);
            break;
        }
        case 2: // O3: relocate block
        {
            int src = rMach(rng);
            std::vector<int> on_src;
            for (int j = 1; j <= n; ++j)
                if (sol[j] == src)
                    on_src.push_back(j);
            if ((int)on_src.size() < blockSize)
                break;
            std::uniform_int_distribution<int> rStart(0, (int)on_src.size() - blockSize);
            int start = rStart(rng);
            int dst;
            do
            {
                dst = rMach(rng);
            } while (dst == src);
            for (int i = start; i < start + blockSize; ++i)
                sol[on_src[i]] = dst;
            break;
        }
        case 3: // O4: swap 2 blocks between 2 machines
        {
            int ma = rMach(rng), mb;
            do
            {
                mb = rMach(rng);
            } while (mb == ma);
            std::vector<int> ja, jb;
            for (int j = 1; j <= n; ++j)
            {
                if (sol[j] == ma)
                    ja.push_back(j);
                if (sol[j] == mb)
                    jb.push_back(j);
            }
            if ((int)ja.size() < blockSize || (int)jb.size() < blockSize)
                break;
            std::uniform_int_distribution<int> rA(0, (int)ja.size() - blockSize);
            std::uniform_int_distribution<int> rB(0, (int)jb.size() - blockSize);
            int sa = rA(rng), sb = rB(rng);
            for (int i = 0; i < blockSize; ++i)
                std::swap(sol[ja[sa + i]], sol[jb[sb + i]]);
            break;
        }
        case 4: // O5: ruin & recreate (20% jobs)
        {
            int k = n / 5;
            std::vector<int> removed;
            for (int i = 0; i < k; ++i)
            {
                int j = rJob(rng);
                removed.push_back(j);
                sol[j] = 0;
            }
            for (int j : removed)
                sol[j] = rMach(rng);
            break;
        }
        case 5: // O6: random double bridge (4-opt like perturbation)
        {
            // Pick 4 random jobs and rotate their machine assignments
            std::vector<int> jobs = {rJob(rng), rJob(rng), rJob(rng), rJob(rng)};
            int t = sol[jobs[0]];
            sol[jobs[0]] = sol[jobs[1]];
            sol[jobs[1]] = sol[jobs[2]];
            sol[jobs[2]] = sol[jobs[3]];
            sol[jobs[3]] = t;
            break;
        }
        case 6: // O7: ejection chain
        {
            int chainLen = 2 + (rng() % 4);
            op_ejectionChain(sol, chainLen);
            break;
        }
        }
    }

    void op_ejectionChain(std::vector<int> &sol, int chainLen)
    {
        int n = instance.job;
        int m = instance.mach;
        if (chainLen < 2)
            chainLen = 2;

        std::uniform_int_distribution<int> rJob(1, n);
        std::uniform_int_distribution<int> rMach(1, m);

        int startJob = rJob(rng);
        int curJob = startJob;
        int curMach = sol[curJob];

        std::vector<int> visitedJobs;
        visitedJobs.push_back(curJob);

        for (int step = 0; step < chainLen; ++step)
        {
            int nextMach;
            do
            {
                nextMach = rMach(rng);
            } while (nextMach == curMach);

            // find a job on nextMach to eject
            int ejectJob = -1;
            for (int j = 1; j <= n; ++j)
            {
                if (sol[j] == nextMach)
                {
                    ejectJob = j;
                    break;
                }
            }

            // move current job
            sol[curJob] = nextMach;

            if (ejectJob == -1)
                break;

            curJob = ejectJob;
            curMach = nextMach;
            visitedJobs.push_back(curJob);
        }

        // close chain
        sol[curJob] = sol[startJob];
    }

    std::vector<long long> buildLoad(const std::vector<int> &sol) const
    {
        std::vector<long long> ld(instance.mach + 1, 0);
        for (int j = 1; j <= instance.job; ++j)
            if (sol[j] > 0)
                ld[sol[j]] += instance.proc_time[j];
        return ld;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DESTROY OPERATORS
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<int> d_heavyMachine(const std::vector<int> &sol, int k)
    {
        auto ld = buildLoad(sol);
        std::vector<int> ord(instance.job);
        std::iota(ord.begin(), ord.end(), 1);
        std::sort(ord.begin(), ord.end(), [&](int a, int b)
                  { return ld[sol[a]] > ld[sol[b]]; });
        std::vector<int> ns = sol;
        for (int i = 0; i < std::min(k, instance.job); ++i)
            ns[ord[i]] = 0;
        return ns;
    }

    std::vector<int> d_worstCompletion(const std::vector<int> &sol, int k)
    {
        std::vector<long long> pref(instance.mach + 1, 0), ct(instance.job + 1, 0);
        for (int j = 1; j <= instance.job; ++j)
        {
            pref[sol[j]] += instance.proc_time[j];
            ct[j] = pref[sol[j]];
        }
        std::vector<int> ord(instance.job);
        std::iota(ord.begin(), ord.end(), 1);
        std::sort(ord.begin(), ord.end(), [&](int a, int b)
                  { return ct[a] > ct[b]; });
        std::vector<int> ns = sol;
        for (int i = 0; i < std::min(k, instance.job); ++i)
            ns[ord[i]] = 0;
        return ns;
    }

    std::vector<int> d_random(const std::vector<int> &sol, int k)
    {
        std::vector<int> jobs(instance.job);
        std::iota(jobs.begin(), jobs.end(), 1);
        std::shuffle(jobs.begin(), jobs.end(), rng);
        std::vector<int> ns = sol;
        for (int i = 0; i < std::min(k, instance.job); ++i)
            ns[jobs[i]] = 0;
        return ns;
    }

    std::vector<int> d_shaw(const std::vector<int> &sol, int k)
    {
        std::vector<int> ns = sol;
        std::uniform_int_distribution<int> rj(1, instance.job);
        int seed = rj(rng);
        std::vector<bool> removed(instance.job + 1, false);
        removed[seed] = true;
        ns[seed] = 0;
        int rem = 1;
        while (rem < k)
        {
            long long ps = instance.proc_time[seed];
            int best = -1;
            long long bd = LLONG_MAX;
            for (int j = 1; j <= instance.job; ++j)
            {
                if (removed[j])
                    continue;
                long long d = std::abs((long long)instance.proc_time[j] - ps);
                if (d < bd)
                {
                    bd = d;
                    best = j;
                }
            }
            if (best < 0)
                break;
            removed[best] = true;
            ns[best] = 0;
            ++rem;
            seed = best;
        }
        return ns;
    }

    // NEW: remove k jobs with highest energy cost contribution
    std::vector<int> d_highCost(const std::vector<int> &sol, int k)
    {
        std::vector<int> ord(instance.job);
        std::iota(ord.begin(), ord.end(), 1);
        // Sort by energy cost = proc_time * unit_cost[machine] descending
        std::sort(ord.begin(), ord.end(), [&](int a, int b)
                  {
            long long ca = (sol[a] > 0) ? (long long)instance.proc_time[a] * instance.unit_cost[sol[a]-1] : 0;
            long long cb = (sol[b] > 0) ? (long long)instance.proc_time[b] * instance.unit_cost[sol[b]-1] : 0;
            return ca > cb; });
        std::vector<int> ns = sol;
        for (int i = 0; i < std::min(k, instance.job); ++i)
            ns[ord[i]] = 0;
        return ns;
    }

    std::vector<int> applyDestroy(int op, const std::vector<int> &sol, int k)
    {
        switch (op)
        {
        case 0:
            return d_heavyMachine(sol, k);
        case 1:
            return d_worstCompletion(sol, k);
        case 2:
            return d_random(sol, k);
        case 3:
            return d_shaw(sol, k);
        default:
            return d_highCost(sol, k);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // REPAIR — Regret-2 insertion
    //
    // Classic regret insertion: for each unassigned job, compute the best (c1)
    // and second-best (c2) insertion cost. The job with highest regret (c2-c1)
    // is inserted first, because it suffers the most if not inserted optimally.
    //
    // Falls back to greedy (op==0) for speed when top_k==1.
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<int> applyRepair(int op, const std::vector<int> &partial)
    {
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        std::vector<int> sol = partial;
        bool use_regret = (op == 1);
        int topk = use_regret ? params.repair_top_k : 1;
        double p0 = use_regret ? params.repair_p0 : 1.0;
        double p1 = use_regret ? params.repair_p1 : 1.0;

        long long curTEC = calcTEC(sol);

        std::vector<int> unassigned;
        for (int j = 1; j <= instance.job; ++j)
            if (sol[j] == 0)
                unassigned.push_back(j);

        if (!use_regret)
        {
            // Fast greedy: sort by proc_time descending, insert best machine
            std::sort(unassigned.begin(), unassigned.end(), [&](int a, int b)
                      { return instance.proc_time[a] > instance.proc_time[b]; });
            for (int j : unassigned)
            {
                long long p = instance.proc_time[j];
                std::vector<std::pair<long long, int>> opts;
                for (int m = 1; m <= instance.mach; ++m)
                {
                    if (curTEC + (long long)(p * instance.unit_cost[m - 1]) > bound)
                        continue;
                    long long prefDst = 0, caDst = 0;
                    for (int k = 1; k < j; ++k)
                        if (sol[k] == m)
                            prefDst += instance.proc_time[k];
                    for (int k = j + 1; k <= instance.job; ++k)
                        if (sol[k] == m)
                            ++caDst;
                    opts.emplace_back(prefDst + p + p * caDst, m);
                }
                int chosen = 1;
                if (opts.empty())
                {
                    for (int m = 2; m <= instance.mach; ++m)
                        if (instance.unit_cost[m - 1] < instance.unit_cost[chosen - 1])
                            chosen = m;
                }
                else
                {
                    std::sort(opts.begin(), opts.end());
                    double r = u01(rng);
                    int k2 = std::min(topk, (int)opts.size());
                    int idx = (r < p0) ? 0 : (r < p1 && k2 > 1) ? 1
                                                                : (k2 > 2 ? 2 : 0);
                    chosen = opts[idx].second;
                }
                sol[j] = chosen;
                curTEC += (long long)(p * instance.unit_cost[chosen - 1]);
            }
            return sol;
        }

        // Regret-2 insertion
        // While there are unassigned jobs, pick the one with highest regret
        while (!unassigned.empty())
        {
            int bestRegretJob = -1;
            long long bestRegret = LLONG_MIN;
            int bestMachine = 1;

            for (int j : unassigned)
            {
                long long p = instance.proc_time[j];
                long long best1 = LLONG_MAX, best2 = LLONG_MAX;
                int bestM1 = -1;

                for (int m = 1; m <= instance.mach; ++m)
                {
                    if (curTEC + (long long)(p * instance.unit_cost[m - 1]) > bound)
                        continue;
                    // Compute insertion cost delta TCT
                    long long prefDst = 0, caDst = 0;
                    for (int k = 1; k < j; ++k)
                        if (sol[k] == m)
                            prefDst += instance.proc_time[k];
                    for (int k = j + 1; k <= instance.job; ++k)
                        if (sol[k] == m)
                            ++caDst;
                    long long cost = prefDst + p + p * caDst;

                    if (cost < best1)
                    {
                        best2 = best1;
                        best1 = cost;
                        bestM1 = m;
                    }
                    else if (cost < best2)
                    {
                        best2 = cost;
                    }
                }

                if (bestM1 < 0)
                {
                    // No feasible machine → forced insertion into cheapest
                    bestM1 = 1;
                    for (int m = 2; m <= instance.mach; ++m)
                        if (instance.unit_cost[m - 1] < instance.unit_cost[bestM1 - 1])
                            bestM1 = m;
                    best1 = 0;
                    best2 = 0;
                }

                // Regret = best2 - best1 (loss of not picking best)
                long long regret = (best2 == LLONG_MAX) ? 0 : (best2 - best1);
                if (regret > bestRegret)
                {
                    bestRegret = regret;
                    bestRegretJob = j;
                    bestMachine = bestM1;
                }
            }

            // Insert bestRegretJob into bestMachine
            sol[bestRegretJob] = bestMachine;
            curTEC += (long long)(instance.proc_time[bestRegretJob] * instance.unit_cost[bestMachine - 1]);
            unassigned.erase(std::find(unassigned.begin(), unassigned.end(), bestRegretJob));
        }
        return sol;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Repair energy constraint
    // ─────────────────────────────────────────────────────────────────────────
    void repairEnergy(std::vector<int> &sol)
    {
        bool imp = true;
        while (imp && calcTEC(sol) > bound)
        {
            imp = false;
            for (int j = 1; j <= instance.job; ++j)
            {
                int src = sol[j];
                for (int m = 1; m <= instance.mach; ++m)
                {
                    if (m == src || instance.unit_cost[m - 1] >= instance.unit_cost[src - 1])
                        continue;
                    sol[j] = m;
                    if (calcTEC(sol) <= bound)
                        return;
                    imp = true;
                    break;
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ALNS weight management
    // ─────────────────────────────────────────────────────────────────────────
    void initWeights()
    {
        for (int i = 0; i < ND; ++i)
        {
            wD[i] = 1.0;
            sD[i] = 0.0;
            uD[i] = 0;
        }
        for (int i = 0; i < NR; ++i)
        {
            wR[i] = 1.0;
            sR[i] = 0.0;
            uR[i] = 0;
        }
    }

    int selectOp(const double *w, int n)
    {
        double tot = 0;
        for (int i = 0; i < n; ++i)
            tot += w[i];
        std::uniform_real_distribution<double> u(0, tot);
        double r = u(rng), acc = 0;
        for (int i = 0; i < n; ++i)
        {
            acc += w[i];
            if (r <= acc)
                return i;
        }
        return n - 1;
    }

    void updateWeights(double *w, double *s, int *u, int n)
    {
        for (int i = 0; i < n; ++i)
        {
            if (u[i] > 0)
                w[i] = w[i] * (1 - params.r_weight) + params.r_weight * (s[i] / u[i]);
            w[i] = std::max(w[i], 0.01);
            s[i] = 0;
            u[i] = 0;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SOLVE
    //
    // Cải tiến so với phiên bản gốc:
    // 1. LS dùng prefix cache → O(n²) thay vì O(n³)
    // 2. Khi stagnation, dùng destroy+repair+LS thay vì random perturbation
    // 3. Thêm destroy operator d_highCost
    // 4. Thêm perturbation operator double bridge (O6)
    // 5. Regret-2 repair (op==1)
    // ─────────────────────────────────────────────────────────────────────────
    void solve()
    {
        int n = instance.job;
        bound = (long long)((1.0 - instance.ctrl_factor) * calcLower() + instance.ctrl_factor * calcUpper());

        std::vector<int> Xinit = runInit();
        localSearch(Xinit);

        std::vector<int> cur = Xinit, best = Xinit;
        double bestFit = calcFit(best), curFit = bestFit;

        int blockSize = 1 + (rng() % std::max(1, n / instance.mach));

        initWeights();
        const int SEG = 200;
        const int LS_FREQ = 5;
        int stag = 0, acceptCount = 0;

        for (int iter = 0; iter < params.max_iter; ++iter)
        {
            bool isPerturb = (stag >= params.stagnation_limit);
            if (isPerturb)
            {
                // Escape: destroy + regret repair + full LS
                cur = best;
                curFit = bestFit;
                stag = 0;
                acceptCount = 0;

                int k = std::max(params.destroy_min,
                                 std::min(params.destroy_max,
                                          (int)(params.destroy_ratio * n)));

                // Alternate between destroy operators for diversity
                int dOp = (iter / params.stagnation_limit) % (ND - 1); // exclude highCost for escaping
                std::vector<int> partial = applyDestroy(dOp, cur, k);
                // Use regret repair for better reconstruction
                cur = applyRepair(1, partial);
                localSearch(cur);
                curFit = calcFit(cur);

                if (curFit < bestFit)
                {
                    best = cur;
                    bestFit = curFit;
                }
                continue;
            }

            int op = selectOp(wD, ND);
            ++uD[op];

            std::vector<int> ns = cur;
            applyOp(op % 6, ns, blockSize); // map ND ops to 6 perturbation types

            double fitNew = calcFit(ns);
            double reward = 0;

            if (fitNew < curFit)
            {
                cur = ns;
                curFit = fitNew;
                ++acceptCount;
                reward = (fitNew < bestFit) ? params.sigma1 : params.sigma2;

                if (acceptCount % LS_FREQ == 0)
                {
                    localSearch(cur);
                    curFit = calcFit(cur);
                    if (curFit < bestFit)
                    {
                        best = cur;
                        bestFit = curFit;
                        stag = 0;
                        reward = params.sigma1;
                    }
                }
            }
            else
            {
                ++stag;
                reward = params.sigma3;
            }

            sD[op] += reward;
            if ((iter + 1) % SEG == 0)
                updateWeights(wD, sD, uD, ND);

            if (params.verbose && iter % 500 == 0)
                std::cout << "iter=" << iter << " | best=" << bestFit
                          << " | stag=" << stag << "\n";
        }

        if (calcTEC(best) > bound)
            repairEnergy(best);
        Xbest = best;
        bestTCT = calcTCT(best);
        std::cout << "Total Completion Time: " << bestTCT << "\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    void runFile(const std::string &fn, int numRuns, std::ostream &out)
    {
        std::cout << "\n=== Running " << fn << " ===\n";
        if (!loadInstance(fn))
            return;
        for (int run = 1; run <= numRuns; ++run)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            solve();
            auto t1 = std::chrono::high_resolution_clock::now();
            double el = std::chrono::duration<double>(t1 - t0).count();
            long long tct = calcTCT(Xbest), tec = calcTEC(Xbest);
            std::cout << " run=" << run << " | TCT=" << tct << " | TEC=" << tec
                      << " | bound=" << bound
                      << " | feasible=" << (tec <= bound ? "YES" : "NO")
                      << " | time=" << el << "s\n";
            out << fn << "," << run << "," << tct << "," << tec << "," << bound
                << "," << (tec <= bound ? 1 : 0) << "," << el << "\n";
        }
    }

    void runAllInstances(int numRuns = 1, const std::string &outf = "lns_results.csv")
    {
        std::ofstream f(outf);
        if (!f.is_open())
        {
            std::cerr << "Cannot open " << outf << "\n";
            return;
        }
        f << "instance,run,TCT,TEC,bound,feasible,runtime\n";
        for (int idx = 1; idx <= 2160; ++idx)
        {
            std::stringstream ss;
            ss << "data/T_" << idx << ".txt";
            runFile(ss.str(), numRuns, f);
            f.flush();
        }
        f.close();
        std::cout << "[ALNS] Saved to " << outf << "\n";
    }

    void printSolution() const
    {
        for (int m = 1; m <= instance.mach; ++m)
        {
            std::cout << "Machine " << m << ": ";
            for (int j = 1; j <= instance.job; ++j)
                if (Xbest[j] == m)
                    std::cout << "J" << j << "(p=" << instance.proc_time[j] << ") ";
            std::cout << "\n";
        }
        std::cout << "TCT   = " << calcTCT(Xbest) << "\n";
        std::cout << "TEC   = " << calcTEC(Xbest) << "\n";
        std::cout << "Bound = " << bound << "\n";
        std::cout << "Feasible: " << (calcTEC(Xbest) <= bound ? "YES" : "NO") << "\n";
    }
};

#endif
