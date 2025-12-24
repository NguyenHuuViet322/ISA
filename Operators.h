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
    static constexpr double MIN_WEIGHT = 1e-6;

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

    void penalize(int op)
    {
        score[op] -= 0.1;
    }

    // ============================
    // Adaptive weight update
    // ============================
    void updateWeights(double rho)
    {
        for (int i = 0; i < OP_COUNT; ++i)
        {
            if (usage[i] > 0)
            {
                double avgScore = score[i] / usage[i];
                weight[i] = (1.0 - rho) * weight[i] + rho * avgScore;
            }

            // CLAMP weight
            if (weight[i] < MIN_WEIGHT)
                weight[i] = MIN_WEIGHT;

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
        if ((int)X.size() <= blockSize + 1)
            return;

        std::uniform_int_distribution<int> startDist(
            1, X.size() - blockSize - 1);

        int start = startDist(rng);
        int newMach = 1 + rng() % machCount;

        for (int i = start; i < start + blockSize; ++i)
            X[i] = newMach;
    }

    // =====================================================
    // O4: Exchange blocks between two machines
    // =====================================================
    void O4(std::vector<int> &X, int machCount, int blockSize)
    {
        if (machCount < 2)
            return;

        int m1 = 1 + rng() % machCount;
        int m2;
        do
        {
            m2 = 1 + rng() % machCount;
        } while (m1 == m2);

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

        std::shuffle(idx1.begin(), idx1.end(), rng);
        std::shuffle(idx2.begin(), idx2.end(), rng);

        for (int k = 0; k < blockSize; ++k)
            std::swap(X[idx1[k]], X[idx2[k]]);
    }

    
};

#endif