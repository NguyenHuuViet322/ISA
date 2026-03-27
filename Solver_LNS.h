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
    int    max_iter         = 500000;
    int    stagnation_limit = 2000;
    double destroy_ratio    = 0.30;
    double perturb_ratio    = 0.50;
    int    destroy_min      = 2;
    int    destroy_max      = 50;
    int    repair_top_k     = 3;
    double repair_p0        = 0.65;
    double repair_p1        = 0.88;
    double penalty_factor   = 50.0;
    // ALNS
    double r_weight = 0.1;
    double sigma1   = 10.0;
    double sigma2   = 4.0;
    double sigma3   = 1.0;
    bool   verbose  = false;
};

class SolverLNS
{
public:
    Instance      instance;
    LNSParams     params;
    std::mt19937  rng{ std::random_device{}() };
    std::vector<int> Xbest;
    long long        bound   = 0;
    long long        bestTCT = 0;

    static const int ND=4, NR=2;
    double wD[ND], wR[NR], sD[ND], sR[NR];
    int    uD[ND], uR[NR];

    explicit SolverLNS(const LNSParams& p = LNSParams{}) : params(p) {}

    bool loadInstance(const std::string& fn)
    {
        if (!instance.readFromFile(fn))
        { std::cerr << "[ALNS] Cannot read: " << fn << "\n"; return false; }
        return true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Objective
    // ─────────────────────────────────────────────────────────────────────────
    long long calcTCT(const std::vector<int>& sol) const
    {
        long long total = 0;
        for (int m = 1; m <= instance.mach; ++m)
        {
            long long t = 0;
            for (int j = 1; j <= instance.job; ++j)
                if (sol[j] == m) { t += instance.proc_time[j]; total += t; }
        }
        return total;
    }

    long long calcTEC(const std::vector<int>& sol) const
    {
        std::vector<long long> ld(instance.mach+1, 0);
        for (int j = 1; j <= instance.job; ++j)
            if (sol[j] > 0) ld[sol[j]] += instance.proc_time[j];
        long long u = 0;
        for (int m = 1; m <= instance.mach; ++m)
            u += (long long)(ld[m] * instance.unit_cost[m-1]);
        return u;
    }

    double calcFit(const std::vector<int>& sol) const
    {
        long long tct = calcTCT(sol), tec = calcTEC(sol);
        if (tec <= bound) return (double)tct;
        double ex = (double)(tec - bound);
        return (double)tct + params.penalty_factor * ex * ex;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Bound & Init (giống ISA)
    // ─────────────────────────────────────────────────────────────────────────
    long long calcUpper() const
    {
        std::vector<long long> xt(instance.mach, 0);
        for (int p : instance.proc_time)
        {
            int mp = (int)(std::min_element(xt.begin(),xt.end()) - xt.begin());
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

    // Init 1: ISA gốc
    std::vector<int> runInitISA()
    {
        std::vector<int> X(instance.job+1, 1);
        auto tec = [&](){
            std::vector<long long> ld(instance.mach+1,0);
            for (int j=1;j<=instance.job;++j) ld[X[j]]+=instance.proc_time[j];
            long long u=0;
            for (int m=1;m<=instance.mach;++m) u+=(long long)(ld[m]*instance.unit_cost[m-1]);
            return u;
        };
        auto redist = [&](){
            std::vector<int> ct(instance.mach+1,0);
            std::vector<std::vector<int>> mj(instance.mach+1);
            for (int i=0;i<=instance.job;++i){ct[X[i]]+=instance.proc_time[i];mj[X[i]].push_back(i);}
            std::vector<int> idx(instance.mach+1); std::iota(idx.begin(),idx.end(),0);
            std::sort(idx.begin(),idx.end(),[&](int a,int b){return ct[a]>ct[b];});
            std::vector<std::vector<int>> nj(instance.mach+1);
            for (int k=0;k<=instance.mach;++k) nj[k]=mj[idx[k]];
            for (int mac=0;mac<instance.mach;++mac) for (int job:nj[mac]) X[job]=mac+1;
        };
        double cost=(double)tec(); long long theta=instance.mach, iter=1;
        while(true){
            theta=instance.mach;
            while(instance.proc_time[iter]*(instance.unit_cost[theta-1]-instance.unit_cost[1])
                  >bound-cost && theta>1) --theta;
            if(theta==1) break;
            X[iter]=theta; cost=(double)tec(); iter++;
            redist();
            if(iter==instance.job) break;
        }
        return X;
    }

    // Init 2: Greedy TCT-aware
    // Sắp xếp job theo proc_time giảm dần (LPT), gán vào máy có
    // completion time thấp nhất mà không vượt bound TEC
    std::vector<int> runInitGreedy()
    {
        int n = instance.job, m = instance.mach;
        std::vector<int> X(n+1, 0);

        // Sắp xếp job theo proc_time giảm dần
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 1);
        std::sort(order.begin(), order.end(), [&](int a, int b){
            return instance.proc_time[a] > instance.proc_time[b];
        });

        // completion[m] = thời điểm hoàn thành cuối cùng trên máy m
        std::vector<long long> completion(m+1, 0);
        std::vector<long long> load(m+1, 0);
        long long curTEC = 0;

        for (int j : order)
        {
            long long p = instance.proc_time[j];
            int bestM = -1;
            long long bestTCT_delta = LLONG_MAX;

            for (int mac = 1; mac <= m; ++mac)
            {
                // TEC check
                long long dTEC = (long long)(p * instance.unit_cost[mac-1]);
                if (curTEC + dTEC > bound) continue;

                // Delta TCT khi thêm job j vào máy mac:
                // job đứng sau tất cả job hiện tại → completion[mac] + p
                long long delta = completion[mac] + p;
                if (delta < bestTCT_delta)
                {
                    bestTCT_delta = delta;
                    bestM = mac;
                }
            }

            if (bestM < 0)
            {
                // Không máy nào thỏa bound → chọn máy rẻ nhất
                bestM = 1;
                for (int mac = 2; mac <= m; ++mac)
                    if (instance.unit_cost[mac-1] < instance.unit_cost[bestM-1])
                        bestM = mac;
            }

            X[j] = bestM;
            completion[bestM] += p;
            load[bestM] += p;
            curTEC += (long long)(p * instance.unit_cost[bestM-1]);
        }
        return X;
    }

    // Chọn init tốt hơn trong 2 cách
    std::vector<int> runInit()
    {
        std::vector<int> X1 = runInitISA();
        std::vector<int> X2 = runInitGreedy();
        return calcFit(X1) <= calcFit(X2) ? X1 : X2;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4 OPERATORS của ISA – random perturbation (không phải best-improving)
    // O1: đổi máy 1 job ngẫu nhiên
    // O2: swap 2 job ngẫu nhiên
    // O3: relocate block ngẫu nhiên sang máy khác
    // O4: swap 2 block giữa 2 máy
    // ─────────────────────────────────────────────────────────────────────────

    void applyOp(int op, std::vector<int>& sol, int blockSize)
    {
        int n = instance.job, m = instance.mach;
        std::uniform_int_distribution<int> rJob(1, n);
        std::uniform_int_distribution<int> rMach(1, m);

        switch (op)
        {
        case 0: // O1: đổi máy 1 job
        {
            int j = rJob(rng);
            int dst;
            do { dst = rMach(rng); } while (dst == sol[j]);
            sol[j] = dst;
            break;
        }
        case 1: // O2: swap 2 job bất kỳ
        {
            int a = rJob(rng), b;
            do { b = rJob(rng); } while (b == a);
            std::swap(sol[a], sol[b]);
            break;
        }
        case 2: // O3: relocate block (blockSize job liên tiếp trên 1 máy)
        {
            int src = rMach(rng);
            std::vector<int> on_src;
            for (int j=1;j<=n;++j) if (sol[j]==src) on_src.push_back(j);
            if ((int)on_src.size() < blockSize) break;
            std::uniform_int_distribution<int> rStart(0, (int)on_src.size()-blockSize);
            int start = rStart(rng);
            int dst;
            do { dst = rMach(rng); } while (dst == src);
            for (int i=start; i<start+blockSize; ++i) sol[on_src[i]] = dst;
            break;
        }
        case 3: // O4: swap 2 block giữa 2 máy
        {
            int ma = rMach(rng), mb;
            do { mb = rMach(rng); } while (mb == ma);
            std::vector<int> ja, jb;
            for (int j=1;j<=n;++j)
            {
                if (sol[j]==ma) ja.push_back(j);
                if (sol[j]==mb) jb.push_back(j);
            }
            if ((int)ja.size()<blockSize||(int)jb.size()<blockSize) break;
            std::uniform_int_distribution<int> rA(0,(int)ja.size()-blockSize);
            std::uniform_int_distribution<int> rB(0,(int)jb.size()-blockSize);
            int sa=rA(rng), sb=rB(rng);
            for (int i=0;i<blockSize;++i) std::swap(sol[ja[sa+i]], sol[jb[sb+i]]);
            break;
        }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LOCAL SEARCH: best-improving relocate + swap đến convergence
    // Dùng calcTCT O(n*m) – đơn giản, đúng
    // ─────────────────────────────────────────────────────────────────────────

    // Delta TCT khi di chuyển job j từ src sang dst – O(n) chính xác
    // prefSrc = sum p[k], k<j, sol[k]=src
    // prefDst = sum p[k], k<j, sol[k]=dst
    // caSrc   = count k>j, sol[k]=src
    // caDst   = count k>j, sol[k]=dst
    // dTCT = (prefDst+p+p*caDst) - (prefSrc+p+p*caSrc)
    long long deltaTCT(const std::vector<int>& sol, int j, int dst) const
    {
        int src = sol[j];
        long long p = instance.proc_time[j];
        long long prefSrc=0, prefDst=0, caSrc=0, caDst=0;
        for (int k=1; k<j; ++k)
        {
            if (sol[k]==src) prefSrc += instance.proc_time[k];
            if (sol[k]==dst) prefDst += instance.proc_time[k];
        }
        for (int k=j+1; k<=instance.job; ++k)
        {
            if (sol[k]==src) ++caSrc;
            if (sol[k]==dst) ++caDst;
        }
        return (prefDst + p*caDst) - (prefSrc + p*caSrc);
    }

    // Delta TCT khi swap job a (máy ma) và job b (máy mb), a < b – O(n)
    long long deltaSwap(const std::vector<int>& sol, int a, int b) const
    {
        int ma=sol[a], mb=sol[b];
        long long pa=instance.proc_time[a], pb=instance.proc_time[b];
        // Tính prefix và suffix counts
        long long prefA_ma=0, prefA_mb=0, prefB_ma=0, prefB_mb=0;
        long long cntAB_ma=0, cntAB_mb=0, cntAfterB_ma=0, cntAfterB_mb=0;
        for (int k=1; k<=instance.job; ++k)
        {
            if (k==a||k==b) continue;
            if (sol[k]==ma)
            {
                if (k<a) prefA_ma+=instance.proc_time[k];
                if (k<b) prefB_ma+=instance.proc_time[k];
                if (k>a&&k<b) ++cntAB_ma;
                if (k>b) ++cntAfterB_ma;
            }
            if (sol[k]==mb)
            {
                if (k<a) prefA_mb+=instance.proc_time[k];
                if (k<b) prefB_mb+=instance.proc_time[k];
                if (k>a&&k<b) ++cntAB_mb;
                if (k>b) ++cntAfterB_mb;
            }
        }
        return (prefA_mb - prefA_ma)
             + (prefB_ma - prefB_mb)
             - pa*cntAB_ma + pa*cntAB_mb
             + (pb-pa)*cntAfterB_ma
             + (pa-pb)*cntAfterB_mb;
    }

    // Relocate: 1 pass, O(n²) total – commit best improving move per job
    bool ls_relocate(std::vector<int>& sol)
    {
        long long curTEC = calcTEC(sol);
        bool improved = false;

        for (int j = 1; j <= instance.job; ++j)
        {
            int bestDst = sol[j];
            long long bestDelta = 0;
            long long p = instance.proc_time[j];

            for (int dst = 1; dst <= instance.mach; ++dst)
            {
                if (dst == sol[j]) continue;
                long long dTEC = (long long)(p*(instance.unit_cost[dst-1]-instance.unit_cost[sol[j]-1]));
                if (curTEC + dTEC > bound) continue;
                long long dTCT = deltaTCT(sol, j, dst);
                if (dTCT < bestDelta) { bestDelta = dTCT; bestDst = dst; }
            }
            if (bestDst != sol[j])
            {
                curTEC += (long long)(p*(instance.unit_cost[bestDst-1]-instance.unit_cost[sol[j]-1]));
                sol[j] = bestDst;
                improved = true;
            }
        }
        return improved;
    }

    // Swap: 1 pass O(n²), delta O(n) mỗi cặp
    bool ls_swap(std::vector<int>& sol)
    {
        long long curTEC = calcTEC(sol);
        bool improved = false;

        for (int a = 1; a <= instance.job; ++a)
        for (int b = a+1; b <= instance.job; ++b)
        {
            if (sol[a] == sol[b]) continue;
            long long pa=instance.proc_time[a], pb=instance.proc_time[b];
            long long dTEC=(long long)((pb-pa)*instance.unit_cost[sol[a]-1]
                                      +(pa-pb)*instance.unit_cost[sol[b]-1]);
            if (curTEC + dTEC > bound) continue;
            long long dTCT = deltaSwap(sol, a, b);
            if (dTCT < 0)
            {
                std::swap(sol[a], sol[b]);
                curTEC += dTEC;
                improved = true;
            }
        }
        return improved;
    }

    void localSearch(std::vector<int>& sol)
    {
        bool any = true;
        while (any)
        {
            any  = ls_relocate(sol);
            any |= ls_swap(sol);
        }
    }

    std::vector<long long> buildLoad(const std::vector<int>& sol) const
    {
        std::vector<long long> ld(instance.mach+1, 0);
        for (int j = 1; j <= instance.job; ++j)
            if (sol[j] > 0) ld[sol[j]] += instance.proc_time[j];
        return ld;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DESTROY operators
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<int> d_heavyMachine(const std::vector<int>& sol, int k)
    {
        auto ld = buildLoad(sol);
        std::vector<int> ord(instance.job); std::iota(ord.begin(),ord.end(),1);
        std::sort(ord.begin(),ord.end(),[&](int a,int b){return ld[sol[a]]>ld[sol[b]];});
        std::vector<int> ns=sol;
        for (int i=0;i<std::min(k,instance.job);++i) ns[ord[i]]=0;
        return ns;
    }

    std::vector<int> d_worstCompletion(const std::vector<int>& sol, int k)
    {
        std::vector<long long> pref(instance.mach+1,0), ct(instance.job+1,0);
        for (int j=1;j<=instance.job;++j)
        { pref[sol[j]]+=instance.proc_time[j]; ct[j]=pref[sol[j]]; }
        std::vector<int> ord(instance.job); std::iota(ord.begin(),ord.end(),1);
        std::sort(ord.begin(),ord.end(),[&](int a,int b){return ct[a]>ct[b];});
        std::vector<int> ns=sol;
        for (int i=0;i<std::min(k,instance.job);++i) ns[ord[i]]=0;
        return ns;
    }

    std::vector<int> d_random(const std::vector<int>& sol, int k)
    {
        std::vector<int> jobs(instance.job); std::iota(jobs.begin(),jobs.end(),1);
        std::shuffle(jobs.begin(),jobs.end(),rng);
        std::vector<int> ns=sol;
        for (int i=0;i<std::min(k,instance.job);++i) ns[jobs[i]]=0;
        return ns;
    }

    std::vector<int> d_shaw(const std::vector<int>& sol, int k)
    {
        std::vector<int> ns=sol;
        std::uniform_int_distribution<int> rj(1,instance.job);
        int seed=rj(rng);
        std::vector<bool> removed(instance.job+1,false);
        removed[seed]=true; ns[seed]=0; int rem=1;
        while(rem<k)
        {
            long long ps=instance.proc_time[seed]; int best=-1; long long bd=LLONG_MAX;
            for (int j=1;j<=instance.job;++j)
            {
                if(removed[j]) continue;
                long long d=std::abs((long long)instance.proc_time[j]-ps);
                if(d<bd){bd=d;best=j;}
            }
            if(best<0) break;
            removed[best]=true; ns[best]=0; ++rem; seed=best;
        }
        return ns;
    }

    std::vector<int> applyDestroy(int op, const std::vector<int>& sol, int k)
    {
        switch(op){
            case 0: return d_heavyMachine(sol,k);
            case 1: return d_worstCompletion(sol,k);
            case 2: return d_random(sol,k);
            default:return d_shaw(sol,k);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // REPAIR – delta TCT thực O(n) mỗi máy
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<int> applyRepair(int op, const std::vector<int>& partial)
    {
        std::uniform_real_distribution<double> u01(0.0,1.0);
        std::vector<int> sol = partial;
        int topk=(op==0)?1:params.repair_top_k;
        double p0=(op==0)?1.0:params.repair_p0;
        double p1=(op==0)?1.0:params.repair_p1;

        long long curTEC = calcTEC(sol);

        std::vector<int> unassigned;
        for (int j=1;j<=instance.job;++j) if (sol[j]==0) unassigned.push_back(j);
        std::sort(unassigned.begin(),unassigned.end(),[&](int a,int b){
            return instance.proc_time[a]>instance.proc_time[b];
        });

        for (int j : unassigned)
        {
            long long p = instance.proc_time[j];
            std::vector<std::pair<long long,int>> opts;

            for (int m = 1; m <= instance.mach; ++m)
            {
                if (curTEC + (long long)(p*instance.unit_cost[m-1]) > bound) continue;
                // Delta TCT: prefDst(j) + p + p*cntAfter(j,m)
                long long prefDst=0, caDst=0;
                for (int k=1;k<j;++k)  if (sol[k]==m) prefDst+=instance.proc_time[k];
                for (int k=j+1;k<=instance.job;++k) if (sol[k]==m) ++caDst;
                opts.emplace_back(prefDst + p + p*caDst, m);
            }

            int chosen=1;
            if (opts.empty())
            {
                int ch=1;
                for (int m=2;m<=instance.mach;++m)
                    if (instance.unit_cost[m-1]<instance.unit_cost[ch-1]) ch=m;
                chosen=ch;
            }
            else
            {
                std::sort(opts.begin(),opts.end());
                int k=std::min(topk,(int)opts.size());
                double r=u01(rng);
                int idx=(r<p0)?0:(r<p1&&k>1)?1:(k>2?2:0);
                chosen=opts[idx].second;
            }
            sol[j]=chosen;
            curTEC += (long long)(p*instance.unit_cost[chosen-1]);
        }
        return sol;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Repair energy
    // ─────────────────────────────────────────────────────────────────────────
    void repairEnergy(std::vector<int>& sol)
    {
        bool imp=true;
        while(imp && calcTEC(sol)>bound)
        {
            imp=false;
            for (int j=1;j<=instance.job;++j)
            {
                int src=sol[j];
                for (int m=1;m<=instance.mach;++m)
                {
                    if (m==src||instance.unit_cost[m-1]>=instance.unit_cost[src-1]) continue;
                    sol[j]=m;
                    if (calcTEC(sol)<=bound) return;
                    imp=true; break;
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ALNS
    // ─────────────────────────────────────────────────────────────────────────
    void initWeights()
    {
        for(int i=0;i<ND;++i){wD[i]=1.0;sD[i]=0.0;uD[i]=0;}
        for(int i=0;i<NR;++i){wR[i]=1.0;sR[i]=0.0;uR[i]=0;}
    }

    int selectOp(const double* w, int n)
    {
        double tot=0; for(int i=0;i<n;++i) tot+=w[i];
        std::uniform_real_distribution<double> u(0,tot);
        double r=u(rng),acc=0;
        for(int i=0;i<n;++i){acc+=w[i];if(r<=acc) return i;}
        return n-1;
    }

    void updateWeights(double* w, double* s, int* u, int n)
    {
        for(int i=0;i<n;++i)
        {
            if(u[i]>0) w[i]=w[i]*(1-params.r_weight)+params.r_weight*(s[i]/u[i]);
            w[i]=std::max(w[i],0.01);
            s[i]=0; u[i]=0;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SOLVE: ILS (Iterated Local Search) với 4 operator ISA
    // Flow:
    //   init → LS(convergence)
    //   loop {
    //     applyOp (random perturbation, không LS)  ← nhanh
    //     accept nếu fit tốt hơn cur                ← không LS
    //     mỗi K accept: LS(convergence) → update cur/best
    //     perturbation mạnh khi stagnate
    //   }
    // ─────────────────────────────────────────────────────────────────────────
    void solve()
    {
        int n=instance.job;
        bound=(long long)((1.0-instance.ctrl_factor)*calcLower()
                         + instance.ctrl_factor*calcUpper());

        std::vector<int> Xinit=runInit();
        localSearch(Xinit);

        std::vector<int> cur=Xinit, best=Xinit;
        double bestFit=calcFit(best), curFit=bestFit;

        int blockSize = std::max(1, (int)std::round(0.15 * n / instance.mach));

        initWeights();
        const int SEG=200;
        // Chạy LS mỗi LS_FREQ accept
        const int LS_FREQ = 5;
        int stag=0, acceptCount=0;

        for (int iter=0; iter<params.max_iter; ++iter)
        {
            bool isPerturb=(stag>=params.stagnation_limit);
            if (isPerturb)
            {
                // Double perturbation: apply op 3-5 lần rồi LS
                cur=best; curFit=bestFit; stag=0;
                int nPerturb = 3 + rng()%3;
                for (int t=0;t<nPerturb;++t)
                    applyOp(rng()%ND, cur, blockSize*2);
                localSearch(cur);
                curFit=calcFit(cur);
                if (curFit<bestFit){best=cur;bestFit=curFit;}
                acceptCount=0;
                continue;
            }

            int op = selectOp(wD, ND);
            ++uD[op];

            std::vector<int> ns = cur;
            applyOp(op, ns, blockSize);  // random perturbation – NO LS

            double fitNew = calcFit(ns);
            double reward = 0;

            if (fitNew < curFit)
            {
                cur=ns; curFit=fitNew;
                ++acceptCount; ++stag;
                reward = (fitNew<bestFit) ? params.sigma1 : params.sigma2;

                // Chạy LS sau LS_FREQ accept liên tiếp
                if (acceptCount % LS_FREQ == 0)
                {
                    localSearch(cur);
                    curFit=calcFit(cur);
                    if (curFit<bestFit)
                    {
                        best=cur; bestFit=curFit; stag=0;
                        reward=params.sigma1;
                    }
                }
            }
            else
            {
                ++stag;
                reward=params.sigma3;
            }

            sD[op]+=reward;
            if ((iter+1)%SEG==0) updateWeights(wD,sD,uD,ND);

            if (params.verbose&&iter%500==0)
                std::cout<<"iter="<<iter<<" | best="<<bestFit<<" | stag="<<stag<<"\n";
        }

        if (calcTEC(best)>bound) repairEnergy(best);
        Xbest=best; bestTCT=calcTCT(best);
        std::cout<<"Total Completion Time: "<<bestTCT<<"\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    void runFile(const std::string& fn, int numRuns, std::ostream& out)
    {
        std::cout<<"\n=== Running "<<fn<<" ===\n";
        if (!loadInstance(fn)) return;
        for (int run=1;run<=numRuns;++run)
        {
            auto t0=std::chrono::high_resolution_clock::now();
            solve();
            auto t1=std::chrono::high_resolution_clock::now();
            double el=std::chrono::duration<double>(t1-t0).count();
            long long tct=calcTCT(Xbest), tec=calcTEC(Xbest);
            std::cout<<" run="<<run<<" | TCT="<<tct<<" | TEC="<<tec
                     <<" | bound="<<bound
                     <<" | feasible="<<(tec<=bound?"YES":"NO")
                     <<" | time="<<el<<"s\n";
            out<<fn<<","<<run<<","<<tct<<","<<tec<<","<<bound
               <<","<<(tec<=bound?1:0)<<","<<el<<"\n";
        }
    }

    void runAllInstances(int numRuns=1, const std::string& outf="lns_results.csv")
    {
        std::ofstream f(outf);
        if (!f.is_open()){std::cerr<<"Cannot open "<<outf<<"\n";return;}
        f<<"instance,run,TCT,TEC,bound,feasible,runtime\n";
        for (int idx=1;idx<=2160;++idx)
        {
            std::stringstream ss; ss<<"data/T_"<<idx<<".txt";
            runFile(ss.str(),numRuns,f); f.flush();
        }
        f.close();
        std::cout<<"[ALNS] Saved to "<<outf<<"\n";
    }

    void printSolution() const
    {
        for (int m=1;m<=instance.mach;++m)
        {
            std::cout<<"Machine "<<m<<": ";
            for (int j=1;j<=instance.job;++j)
                if (Xbest[j]==m) std::cout<<"J"<<j<<"(p="<<instance.proc_time[j]<<") ";
            std::cout<<"\n";
        }
        std::cout<<"TCT   = "<<calcTCT(Xbest)<<"\n";
        std::cout<<"TEC   = "<<calcTEC(Xbest)<<"\n";
        std::cout<<"Bound = "<<bound<<"\n";
        std::cout<<"Feasible: "<<(calcTEC(Xbest)<=bound?"YES":"NO")<<"\n";
    }
};

#endif