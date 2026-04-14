#ifndef OPERATORS_H
#define OPERATORS_H

#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <iostream>

class Operators
{
public:
    static constexpr int OP_COUNT = 4;
    static constexpr double MIN_WEIGHT = 10e-3;
    enum class OperatorMode { STANDARD, SMART };

    OperatorMode mode = OperatorMode::STANDARD;

    std::mt19937 rng{std::random_device{}()};

    // ISA adaptive parameters
    std::vector<double> weight; // roulette weights
    std::vector<double> score;  // accumulated rewards
    std::vector<int> usage;     // usage count

    Operators()
    {
        weight.assign(OP_COUNT, 10.0);
        score.assign(OP_COUNT, 0.0);
        usage.assign(OP_COUNT, 0);
    }

    // ============================
    // Roulette wheel selection
    // ============================
    int selectOperator()
    {
        double sum = std::accumulate(weight.begin(), weight.end(), 0.0);

        // Guard chống crash
        if (sum <= 0.0)
        {
            resetWeights();
            sum = OP_COUNT;
        }

        std::uniform_real_distribution<double> dist(0.0, sum);

        double r = dist(rng);
        double acc = 0.0;

        for (int i = 0; i < OP_COUNT; ++i)
        {
            acc += weight[i];
            if (r <= acc)
                return i;
        }
        return OP_COUNT - 1;
    }

    // ============================
    // Apply selected operator
    // ============================
    void apply(int op,
           std::vector<int> &X,
           int machCount,
           int blockSize,
           std::vector<int>       procTime  = std::vector<int>(),
           std::vector<long long> machLoad  = std::vector<long long>())
    {
        switch (op)
        {
        case 0: O1(X, machCount); break;
        case 1: O2(X); break;
        case 2: O4(X, machCount, blockSize); break;
        case 3:
            O3_SmartBlock(X, machCount, blockSize, procTime);
            break;
        }
        usage[op]++;
    }

    // ============================
    // Reward & penalty
    // ============================
    void reward(int op, double value)
    {
        score[op] += value;
    }

    void penalize(int op, double factor = 1.0)
    {
        score[op] -= 0.1 * factor;
    }

    // ============================
    // Adaptive weight update
    // ============================
    void updateWeights(double rho)
    {
        for (int i = 0; i < OP_COUNT; ++i)
        {
            if (usage[i] > 0)
                weight[i] = (1.0 - rho) * weight[i] + rho * score[i];
            
            weight[i] = std::max(weight[i], MIN_WEIGHT);
            score[i] = 0.0;
            usage[i] = 0;
        }
    }

    // ============================
    // Reset all weights
    // ============================
    void resetWeights()
    {
        std::fill(weight.begin(), weight.end(), 10.0);
        std::fill(score.begin(), score.end(), 0.0);
        std::fill(usage.begin(), usage.end(), 0);
    }

private:
    // =====================================================
    // O1: Change machine of a random job
    // =====================================================
    void O1(std::vector<int> &X, int machCount)
    {
        if (X.size() <= 1)
            return;

        std::uniform_int_distribution<int> jobDist(1, X.size() - 1);
        int job = jobDist(rng);

        int newMach;
        do
        {
            newMach = 1 + rng() % machCount;
        } while (newMach == X[job]);

        X[job] = newMach;
    }

    // =====================================================
    // O2: Swap two jobs
    // =====================================================
    void O2(std::vector<int> &X)
    {
        std::uniform_int_distribution<int> dist(1, X.size() - 1);
        int i = dist(rng);
        int j;
        int attempts = 0;
    do
    {
        j = dist(rng);
        attempts++;
        if (attempts > 10) return; // thoát nếu không tìm được
    } while (i == j || X[i] == X[j]);

        std::swap(X[i], X[j]);
    }

    // =====================================================
    // O3: Assign a block to one machine
    // =====================================================
    void O3(std::vector<int> &X, int machCount, int blockSize)
    {
        int m1 = 1 + rng() % machCount;

        // 2. Lấy các job thuộc m1
        std::vector<int> idx;
        for (int i = 1; i < (int)X.size(); ++i)
            if (X[i] == m1)
                idx.push_back(i);

        if ((int)idx.size() < blockSize)
            return;

        // 3. Chọn block liên tiếp trong idx
        int start = rng() % (idx.size() - blockSize + 1);

        // 4. Chọn máy đích
        int m2;
        do
        {
            m2 = 1 + rng() % machCount;
        } while (m2 == m1);

        // 5. Relocate block
        for (int k = start; k < start + blockSize; ++k)
            X[idx[k]] = m2;
    }

    void O3_SmartBlock(std::vector<int> &X, int machCount, int blockSize,
                   const std::vector<int> &procTime)
{
    // Chọn máy nguồn ngẫu nhiên như cũ
    int m1 = 1 + rng() % machCount;

    std::vector<int> idx;
    for (int i = 1; i < (int)X.size(); ++i)
        if (X[i] == m1) idx.push_back(i);

    if ((int)idx.size() < blockSize) return;

    // Thay vì block liên tiếp ngẫu nhiên
    // → chọn block có tổng procTime LỚN NHẤT
    // → đây là block đóng góp TCT nhiều nhất
    int bestStart = 0;
    long long bestSum = 0;
    for (int s = 0; s <= (int)idx.size() - blockSize; ++s)
    {
        long long sum = 0;
        for (int k = s; k < s + blockSize; ++k)
            sum += procTime[idx[k]];
        if (sum > bestSum) { bestSum = sum; bestStart = s; }
    }

    int m2;
    do { m2 = 1 + rng() % machCount; } while (m2 == m1);

    for (int k = bestStart; k < bestStart + blockSize; ++k)
        X[idx[k]] = m2;
}



    // =====================================================
    // O4: Exchange blocks between two machines
    // =====================================================
    void O4(std::vector<int> &X, int machCount, int blockSize)
    {
        int m1 = 1 + rng() % machCount;
        int m2;
        do
        {
            m2 = 1 + rng() % machCount;
        } while (m2 == m1);

        // job list theo machine
        std::vector<int> idx1, idx2;
        for (int i = 1; i < (int)X.size(); ++i)
        {
            if (X[i] == m1)
                idx1.push_back(i);
            else if (X[i] == m2)
                idx2.push_back(i);
        }

        if ((int)idx1.size() < blockSize ||
            (int)idx2.size() < blockSize)
            return;

        // chọn block liên tiếp
        int s1 = rng() % (idx1.size() - blockSize + 1);
        int s2 = rng() % (idx2.size() - blockSize + 1);

        // swap block
        for (int k = 0; k < blockSize; ++k)
            std::swap(X[idx1[s1 + k]], X[idx2[s2 + k]]);
    }

    
    // =====================================================
    // O5: Ejection Chain
    // =====================================================
    void O5_EjectionChain(std::vector<int> &X, int machCount)
    {
        if (X.size() <= 2)
            return;

        int chainLength = 3 + rng() % 4; // chain 3-6

        std::uniform_int_distribution<int> jobDist(1, X.size() - 1);

        int job = jobDist(rng);
        int currentMachine = X[job];

        for (int step = 0; step < chainLength; ++step)
        {
            int nextMachine;
            do
            {
                nextMachine = 1 + rng() % machCount;
            } while (nextMachine == currentMachine);

            // tìm job trên nextMachine để eject
            std::vector<int> candidates;
            for (int i = 1; i < (int)X.size(); ++i)
                if (X[i] == nextMachine)
                    candidates.push_back(i);

            if (candidates.empty())
            {
                // nếu machine rỗng thì chỉ move job
                X[job] = nextMachine;
                return;
            }

            int nextJob = candidates[rng() % candidates.size()];

            // eject
            std::swap(X[job], X[nextJob]);

            job = nextJob;
            currentMachine = X[job];
        }
    }

    void O6_DoubleBridge(std::vector<int> &X, int machCount)
    {
        int n = X.size() - 1;
        if (n < 4) return;

        // Chọn 4 vị trí ngẫu nhiên không trùng
        std::uniform_int_distribution<int> dist(1, n);
        int a, b, c, d;
        do { a = dist(rng); } while (false);
        do { b = dist(rng); } while (b == a);
        do { c = dist(rng); } while (c == a || c == b);
        do { d = dist(rng); } while (d == a || d == b || d == c);

        // Reassign 4 job sang 4 machine ngẫu nhiên hoàn toàn
        std::uniform_int_distribution<int> machDist(1, machCount);
        X[a] = machDist(rng);
        X[b] = machDist(rng);
        X[c] = machDist(rng);
        X[d] = machDist(rng);
    }

    void O5_EjectionChain_Improved(std::vector<int> &X, int machCount)
    {
        int n = X.size();
        if (n <= 2) return;

        std::uniform_int_distribution<int> jobDist(1, n - 1);

        int chainLength = 3 + rng() % 4; // 3–6

        std::vector<int> jobs;
        std::vector<int> machines;

        // 1. chọn job đầu
        int startJob = jobDist(rng);
        jobs.push_back(startJob);
        machines.push_back(X[startJob]);

        int currentMachine = X[startJob];

        // 2. build chain
        for (int step = 1; step < chainLength; ++step)
        {
            int nextMachine;
            do {
                nextMachine = 1 + rng() % machCount;
            } while (nextMachine == currentMachine);

            // tìm job trên machine đó
            std::vector<int> candidates;
            for (int i = 1; i < n; ++i)
                if (X[i] == nextMachine)
                    candidates.push_back(i);

            if (candidates.empty())
            {
                // nếu machine rỗng → kết thúc chain sớm
                break;
            }

            int nextJob = candidates[rng() % candidates.size()];

            jobs.push_back(nextJob);
            machines.push_back(nextMachine);

            currentMachine = nextMachine;
        }

        int k = jobs.size();
        if (k <= 1) return;

        // 3. ROTATION (quan trọng nhất)
        int lastMachine = machines.back();

        for (int i = k - 1; i > 0; --i)
        {
            X[jobs[i]] = machines[i - 1];
        }

        X[jobs[0]] = lastMachine;
    }

    void O7_TargetedRestart(std::vector<int> &X, int machCount,
                            const std::vector<int> &proc_time)
    {
        int n = X.size() - 1;
        if (n <= 0) return;

        // 1. Tính completion time từng machine
        std::vector<long long> ct(machCount + 1, 0);
        for (int i = 1; i <= n; ++i)
            ct[X[i]] += proc_time[i];

        // 2. Tìm machine bottleneck — stochastic trong top-2
        std::vector<int> rank;
        for (int m = 1; m <= machCount; ++m)
            if (ct[m] > 0) rank.push_back(m);
        std::sort(rank.begin(), rank.end(),
                [&](int a, int b){ return ct[a] > ct[b]; });

        int topK = std::min(2, (int)rank.size());
        std::uniform_int_distribution<int> pickM(0, topK - 1);
        int srcM = rank[pickM(rng)];

        // 3. Tìm job có proc_time lớn nhất trên srcM
        //    → job này đóng góp CT nhiều nhất vì nằm cuối queue
        int targetJob = -1;
        long long maxP = -1;
        for (int i = 1; i <= n; ++i)
        {
            if (X[i] == srcM && proc_time[i] > maxP)
            {
                maxP = proc_time[i];
                targetJob = i;
            }
        }
        if (targetJob == -1) return;

        // 4. Thử tất cả machine, tính delta TCT
        //    delta = proc_time[job] * (jobs_on_dest) - proc_time[job] * (jobs_on_src - 1)
        //          = proc_time[job] * (jobs_on_dest - jobs_on_src + 1)
        std::vector<int> jobCount(machCount + 1, 0);
        for (int i = 1; i <= n; ++i)
            jobCount[X[i]]++;

        std::vector<std::pair<long long, int>> options;
        for (int m = 1; m <= machCount; ++m)
        {
            if (m == srcM) continue;
            // delta TCT khi move targetJob từ srcM → m
            long long delta = (long long)proc_time[targetJob] *
                            (jobCount[m] - jobCount[srcM] + 1);
            options.push_back({delta, m});
        }

        if (options.empty()) return;
        std::sort(options.begin(), options.end()); // delta nhỏ nhất = tốt nhất

        // 5. Stochastic: chọn ngẫu nhiên trong top-3 tốt nhất
        //    → tránh deterministic, tăng diversity
        int topJ = std::min(3, (int)options.size());

        // Bias mạnh hơn về best khi T thấp (cuối run)
        // Nhưng vì không có T ở đây, dùng weighted random đơn giản
        std::uniform_real_distribution<double> r(0.0, 1.0);
        int chosen;
        double rv = r(rng);
        if      (rv < 0.6) chosen = 0;
        else if (rv < 0.85 && topJ > 1) chosen = 1;
        else if (topJ > 2) chosen = 2;
        else chosen = 0;

        X[targetJob] = options[chosen].second;
    }
};

#endif