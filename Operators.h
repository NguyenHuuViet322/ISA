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
    static constexpr int OP_COUNT = 6;
    static constexpr double MIN_WEIGHT = 10e-3;

    std::mt19937 rng{std::random_device{}()};

    // ISA adaptive parameters
    std::vector<double> weight; // roulette weights
    std::vector<double> score;  // accumulated rewards
    std::vector<int> usage;     // usage count

    Operators()
    {
        weight.assign(OP_COUNT, 1.0);
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
               int blockSize)
    {
        switch (op)
        {
        case 0:
            O1(X, machCount);
            break;
        case 1:
            O2(X);
            break;
        case 2:
            O3(X, machCount, blockSize);
            break;
        case 3:
            O4(X, machCount, blockSize);
            break;
        case 4:
            O5_EjectionChain(X, machCount);
            break;
        case 5:
            O6_BalanceMove(X, machCount);
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
        // Tính avgScore cho từng op
        std::vector<double> avgScore(OP_COUNT, 0.0);
        for (int i = 0; i < OP_COUNT; ++i)
            if (usage[i] > 0)
                avgScore[i] = score[i] / usage[i];

        // Shift về dương: trừ min rồi + epsilon
        double minAvg = *std::min_element(avgScore.begin(), avgScore.end());
        for (int i = 0; i < OP_COUNT; ++i)
            avgScore[i] -= minAvg - 1e-3; // shift để min > 0

        for (int i = 0; i < OP_COUNT; ++i)
        {
            if (usage[i] > 0)
                weight[i] = (1.0 - rho) * weight[i] + rho * avgScore[i];

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
        std::fill(weight.begin(), weight.end(), 1.0);
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
        if (X.size() <= 2)
            return;

        std::uniform_int_distribution<int> dist(1, X.size() - 1);
        int i = dist(rng);
        int j;
        do
        {
            j = dist(rng);
        } while (i == j);

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

    // =====================================================
    // O6: Move job from busiest machine to least loaded
    // =====================================================
    void O6_BalanceMove(std::vector<int> &X, int machCount)
    {
        if (X.size() <= 2)
            return;

        std::vector<int> load(machCount + 1, 0);

        for (int i = 1; i < (int)X.size(); ++i)
            load[X[i]]++;

        int maxM = 1;
        int minM = 1;

        for (int m = 2; m <= machCount; ++m)
        {
            if (load[m] > load[maxM])
                maxM = m;
            if (load[m] < load[minM])
                minM = m;
        }

        if (maxM == minM)
            return;

        // lấy job từ machine nhiều nhất
        std::vector<int> jobs;
        for (int i = 1; i < (int)X.size(); ++i)
            if (X[i] == maxM)
                jobs.push_back(i);

        if (jobs.empty())
            return;

        int job = jobs[rng() % jobs.size()];
        X[job] = minM;
    }
};

#endif