#include <vector>
#include <random>
#include <algorithm>

class Operators {
public:
    std::mt19937 rng{std::random_device{}()};

    // O1: Đổi máy của một job ngẫu nhiên
    void O1(std::vector<int>& X, int machCount) {
        if (X.empty()) return;
        int job = rand() % X.size();
        int newMach;
        do {
            newMach = 1 + rand() % machCount; // máy từ 1..machCount
        } while (newMach == X[job]);
        X[job] = newMach;
    }

    // O2: Hoán đổi hai job
    void O2(std::vector<int>& X) {
        if (X.size() < 2) return;
        int i = rand() % X.size();
        int j;
        do {
            j = rand() % X.size();
        } while (i == j);
        std::swap(X[i], X[j]);
    }

    // O3: Gán lại block liên tiếp các job cho cùng một máy
    void O3(std::vector<int>& X, int machCount, int block_size) {
        if (X.size() < block_size) return;
        int start = rand() % (X.size() - block_size + 1);
        int newMach = 1 + rand() % machCount; // máy từ 1..machCount
        for (int i = start; i < start + block_size; ++i)
            X[i] = newMach;
    }

    // O4: Trao đổi block giữa hai máy khác nhau
    void O4(std::vector<int>& X, int machCount, int block_size) {
        if (machCount < 2 || X.size() < 2 * block_size) return;

        int m1 = 1 + rand() % machCount;
        int m2;
        do { 
            m2 = 1 + rand() % machCount; 
        } while (m1 == m2);

        std::vector<int> idx1, idx2;
        for (int i = 0; i < X.size(); ++i) {
            if (X[i] == m1) idx1.push_back(i);
            else if (X[i] == m2) idx2.push_back(i);
        }

        if (idx1.size() < block_size || idx2.size() < block_size) return;

        std::shuffle(idx1.begin(), idx1.end(), rng);
        std::shuffle(idx2.begin(), idx2.end(), rng);

        for (int k = 0; k < block_size; ++k)
            std::swap(X[idx1[k]], X[idx2[k]]);
    }
};
