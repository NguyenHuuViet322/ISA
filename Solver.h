#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include "Instance.h"
#include <vector>
#include <numeric>
#include "Operators.h"

class Solver
{
public:
    std::mt19937 rng{std::random_device{}()};
    Instance instance;
    Operators ops;
    std::vector<int> X; // X[i] có nghĩ là việc i được xử lý bằng máy x[i]
    std::vector<int> cost;
    int upper_bound;
    int lower_bound;
    int bound;
    double block_parametter = 0.4, cooling_rate = 0.995, adaptation_rate = 0.15;
    double phi1_score_factor = 0.8, phi2_score_factor = 0.7, reset_threshold = 300;
    double penalty_factor = 50, t0 = 1000.0, beta = 15;

    Solver()
    {
        if (instance.readFromFile("data/T_1501.txt"))
        {
            // X.resize(instance.job + 1, 1);
            // cost.resize(instance.mach + 1, 0);
            // instance.print();
            // upper_bound = calculateUpper();
            // lower_bound = calculateLower();
            // bound = (1 - instance.ctrl_factor) * lower_bound + instance.ctrl_factor * upper_bound;
            // std::cout << "Upper Bound: " << upper_bound << "\n";
            // std::cout << "Lower Bound: " << lower_bound << "\n";
            // std::cout << "Bound: " << bound << "\n";
            // std::cout << "Cost: " << totalEnergyConsumption() << "\n";
            // init();
            // t0 = 0.2 * totalCompletionTime() / log(2);
            // ISA();

            parameterTuning();
        }
        else
        {
            std::cerr << "Không thể đọc dữ liệu instance!\n";
        }
    }

    void init()
    {
        double cost = totalEnergyConsumption();
        long long theta = instance.mach;
        long long iter = 1;
        while (true)
        {
            theta = instance.mach;
            while (instance.proc_time[iter] * (instance.unit_cost[theta - 1] - instance.unit_cost[1] > bound - cost) && theta > 1)
            {
                --theta;
            }

            if (theta == 1)
                break;
            else
            {
                X[iter] = theta;
                cost = totalEnergyConsumption();
                iter++;
            }
            redistrubutionBasedOnCost();

            if (iter == instance.job)
                break;
        }
    }

    void ISA()
    {
        std::vector<int> Xb = X;
        double bestFitness = fitnessFunction();

        double T = t0;
        int stagnation = 0;

        int maxInnerIter = std::ceil((double)instance.job / instance.mach);
        int blockSize = std::max(1, (int)(block_parametter * instance.job));

        double beta = 10.0;              // ✅ Thêm beta cơ bản
        double phi1 = phi1_score_factor; // 0.8
        double phi2 = phi2_score_factor; // 0.7

        while (T > 1e-4)
        {
            for (int iter = 0; iter < maxInnerIter; ++iter)
            {
                int op = ops.selectOperator();

                std::vector<int> X_old = X;
                double oldFitness = fitnessFunction();

                ops.apply(op, X, instance.mach, blockSize);
                double newFitness = fitnessFunction();

                // ✅ Sửa logic acceptance và reward
                if (newFitness < bestFitness)
                {
                    // Best improvement: β điểm
                    Xb = X;
                    bestFitness = newFitness;
                    stagnation = 0;
                    ops.reward(op, beta);
                }
                else if (newFitness < oldFitness)
                {
                    // Improvement: φ1·β điểm
                    stagnation++;
                    ops.reward(op, phi1 * beta);
                }
                else
                {
                    // Worse solution → SA acceptance
                    double prob = std::exp(-(newFitness - oldFitness) / T);
                    std::uniform_real_distribution<double> dist(0.0, 1.0);
                    double r = dist(rng);

                    if (r < prob)
                    {
                        // ✅ Chấp nhận: φ2·β điểm
                        stagnation++;
                        ops.reward(op, phi2 * beta);
                    }
                    else
                    {
                        // ✅ Reject: không reward, không penalize
                        X = X_old;
                    }
                }

                if (stagnation >= reset_threshold)
                {
                    ops.resetWeights();
                    stagnation = 0;
                }
            }

            ops.updateWeights(adaptation_rate);
            T *= cooling_rate;
        }

        X = Xb;
        // fastLocalSearch();
        repair();
        std::cout << "Total Completion Time: " << totalCompletionTime() << "\n";
        // printSolution();
    }

    void fastLocalSearch()
    {
        int maxIterations = 100; // Giới hạn số lần thử

        for (int iter = 0; iter < maxIterations; ++iter)
        {
            bool improved = false;
            double currentFit = fitnessFunction();

            // Chỉ thử một số cặp random thay vì tất cả
            int numTries = std::min(50, instance.job); // Chỉ thử 50 cặp

            for (int attempt = 0; attempt < numTries; ++attempt)
            {
                // Random 2 jobs
                int i = 1 + (rng() % instance.job);
                int j = 1 + (rng() % instance.job);

                if (i == j)
                    continue;

                // Thử swap
                std::swap(X[i], X[j]);

                double newFit = fitnessFunction();
                if (newFit < currentFit)
                {
                    // ✅ Chấp nhận ngay và break
                    improved = true;
                    break;
                }
                else
                {
                    // Undo
                    std::swap(X[i], X[j]);
                }
            }

            if (!improved)
                break; // Không còn cải thiện thì dừng
        }
    }

    void simpleLocalSearch()
    {
        bool improved = true;
        int maxNoImprove = 20;
        int noImprove = 0;

        while (improved && noImprove < maxNoImprove)
        {
            improved = false;
            double currentFit = fitnessFunction();

            // Thử swap mọi cặp job
            for (int i = 1; i <= instance.job && !improved; ++i)
            {
                for (int j = i + 1; j <= instance.job; ++j)
                {
                    // Swap
                    std::swap(X[i], X[j]);

                    double newFit = fitnessFunction();
                    if (newFit < currentFit)
                    {
                        currentFit = newFit;
                        improved = true;
                        noImprove = 0;
                        break;
                    }
                    else
                    {
                        // Undo swap
                        std::swap(X[i], X[j]);
                    }
                }
            }

            if (!improved)
                noImprove++;
        }
    }

    void repair()
    {
        if (totalEnergyConsumption() <= bound)
            return;

        std::vector<std::vector<int>> machine_jobs(instance.mach + 1);
        while (totalEnergyConsumption() > bound)
        {
            int gamma = -1;

            for (int i = 0; i <= instance.job; ++i)
            {
                machine_jobs[X[i]].push_back(i);
            }
            for (int i = instance.mach; i > 0; --i)
            {
                if (!machine_jobs[i].empty())
                {
                    gamma = i;
                    break;
                }
            }

            if (gamma == 0)
                break;
            int first_job = machine_jobs[gamma][0];
            X[first_job] = gamma - 1;

            for (int m = 1; m <= instance.mach; ++m)
            {
                std::sort(machine_jobs[m].begin(), machine_jobs[m].end(),
                          [&](int a, int b)
                          { return instance.proc_time[a] < instance.proc_time[b]; });
            }
        }
    }

    void redistrubutionBasedOnCost()
    {
        std::vector<int> cost_tmp(instance.mach + 1, 0);
        std::vector<std::vector<int>> machine_jobs(instance.mach + 1);

        for (int i = 0; i <= instance.job; ++i)
        {
            cost_tmp[X[i]] += instance.proc_time[i];
            machine_jobs[X[i]].push_back(i);
        }

        std::vector<int> idx(instance.mach + 1);
        std::iota(idx.begin(), idx.end(), 0);

        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b)
                  { return cost_tmp[a] > cost_tmp[b]; });

        std::vector<int> new_cost(instance.mach + 1);
        std::vector<std::vector<int>> new_jobs(instance.mach + 1);

        for (int k = 0; k <= instance.mach; ++k)
        {
            new_cost[k] = cost_tmp[idx[k]];
            new_jobs[k] = machine_jobs[idx[k]];
        }

        cost_tmp.swap(new_cost);
        machine_jobs.swap(new_jobs);

        for (int machine = 0; machine < instance.mach; ++machine)
        {
            for (int job : machine_jobs[machine])
            {
                X[job] = machine + 1;
            }
        }
    }

    long long calculateUpper()
    {
        std::vector<long long> x_tmp(instance.mach, 0);
        for (int job : instance.proc_time)
        {
            long long min_pos = std::min_element(x_tmp.begin(), x_tmp.end()) - x_tmp.begin();
            x_tmp[min_pos] += job;
        }
        long long U = 0;
        for (long long i = 0; i < instance.mach; ++i)
        {
            U += x_tmp[i] * instance.unit_cost[i];
        }

        return U;
    }

    long long calculateLower()
    {
        int U = 0;
        for (int i = 0; i <= instance.job; ++i)
        {
            U += instance.proc_time[i] * instance.unit_cost[0];
        }

        return U;
    }

    long long totalCompletionTime()
    {
        long long total = 0;

        for (int m = 1; m <= instance.mach; ++m)
        {
            long long time = 0;

            for (int j = 1; j <= instance.job; ++j)
            {
                if (X[j] == m)
                {
                    time += instance.proc_time[j];
                    total += time;
                }
            }
        }

        return total;
    }

    long long totalEnergyConsumption()
    {
        std::vector<long long> machineTime(instance.mach + 1, 0);

        for (int j = 1; j <= instance.job; ++j)
        {
            int m = X[j];
            machineTime[m] += instance.proc_time[j];
        }

        long long U = 0;
        for (int m = 1; m <= instance.mach; ++m)
        {
            U += machineTime[m] * instance.unit_cost[m - 1];
        }

        return U;
    }

    double fitnessFunction()
    {
        long long TCT = totalCompletionTime();
        long long TEC = totalEnergyConsumption();

        if (TEC <= bound)
        {
            return TCT;
        }
        else
        {
            return TCT + (penalty_factor * (bound - TEC) * (bound - TEC));
        }
    }

    void printSolution()
    {
        for (int i = 1; i <= instance.mach; ++i)
        {
            std::cout << "Machine " << i << ": ";
            for (int j = 1; j < instance.job + 1; ++j)
            {
                if (X[j] == i)
                {
                    std::cout << "Job " << j << " (Proc time: " << instance.proc_time[j] << "), ";
                }
            }
            std::cout << "\n";
        }
    }

    struct ParameterSet
    {
        double cooling_rate;
        double adaptation_rate;
        double block_param;
        int reset_threshold;
        double phi1;
        double phi2;
        double penalty_factor;
        double t0_factor;
    };

    void parameterTuning()
    {
        std::vector<ParameterSet> params = {

            {0.9999, 0.10, 0.4, 300, 0.85, 0.6, 50, 0.2}, // Extremely slow cooling
            // ===== Group 1: Điều chỉnh adaptation_rate =====
            {0.9999, 0.10, 0.1, 300, 0.8, 0.7, 50, 0.2}, // Giảm adaptation → ổn định hơn
            {0.9999, 0.12, 0.1, 300, 0.8, 0.7, 50, 0.2},
            {0.9999, 0.18, 0.1, 300, 0.8, 0.7, 50, 0.2}, // Tăng adaptation → học nhanh hơn
            {0.9999, 0.20, 0.1, 300, 0.8, 0.7, 50, 0.2},

            // ===== Group 2: Điều chỉnh block_param =====
            {0.9999, 0.15, 0.05, 300, 0.8, 0.7, 50, 0.2}, // Block nhỏ hơn
            {0.9999, 0.15, 0.08, 300, 0.8, 0.7, 50, 0.2},
            {0.9999, 0.15, 0.15, 300, 0.8, 0.7, 50, 0.2}, // Block lớn hơn
            {0.9999, 0.15, 0.20, 300, 0.8, 0.7, 50, 0.2},

            // ===== Group 3: Điều chỉnh reset_threshold =====
            {0.9999, 0.15, 0.1, 200, 0.8, 0.7, 50, 0.2}, // Reset sớm hơn
            {0.9999, 0.15, 0.1, 250, 0.8, 0.7, 50, 0.2},
            {0.9999, 0.15, 0.1, 400, 0.8, 0.7, 50, 0.2}, // Reset muộn hơn
            {0.9999, 0.15, 0.1, 500, 0.8, 0.7, 50, 0.2},

            // ===== Group 4: Điều chỉnh phi1 (reward for improvement) =====
            {0.9999, 0.15, 0.1, 300, 0.75, 0.7, 50, 0.2}, // Giảm reward improvement
            {0.9999, 0.15, 0.1, 300, 0.85, 0.7, 50, 0.2}, // Tăng reward improvement
            {0.9999, 0.15, 0.1, 300, 0.90, 0.7, 50, 0.2},

            // ===== Group 5: Điều chỉnh phi2 (reward for SA acceptance) =====
            {0.9999, 0.15, 0.1, 300, 0.8, 0.5, 50, 0.2}, // Giảm reward SA
            {0.9999, 0.15, 0.1, 300, 0.8, 0.6, 50, 0.2},
            {0.9999, 0.15, 0.1, 300, 0.8, 0.8, 50, 0.2}, // Tăng reward SA

            // ===== Group 6: Điều chỉnh penalty_factor =====
            {0.9999, 0.15, 0.1, 300, 0.8, 0.7, 30, 0.2}, // Penalty nhẹ hơn
            {0.9999, 0.15, 0.1, 300, 0.8, 0.7, 75, 0.2}, // Penalty nặng hơn
            {0.9999, 0.15, 0.1, 300, 0.8, 0.7, 100, 0.2},

            // ===== Group 7: Điều chỉnh t0_factor =====
            {0.9999, 0.15, 0.1, 300, 0.8, 0.7, 50, 0.15}, // T0 thấp hơn
            {0.9999, 0.15, 0.1, 300, 0.8, 0.7, 50, 0.25}, // T0 cao hơn
            {0.9999, 0.15, 0.1, 300, 0.8, 0.7, 50, 0.30},

            // ===== Group 8: Kết hợp tốt nhất từ các group =====
            // Dựa trên quan sát, thử kết hợp:
            {0.9999, 0.12, 0.08, 300, 0.85, 0.6, 50, 0.2}, // Adaptive chậm, block nhỏ, phi1 cao
            {0.9999, 0.18, 0.15, 300, 0.8, 0.8, 75, 0.25}, // Adaptive nhanh, block lớn, SA reward cao
            {0.9999, 0.10, 0.1, 400, 0.85, 0.5, 50, 0.2},  // Rất ổn định, reset muộn
            {0.9999, 0.15, 0.05, 250, 0.90, 0.6, 75, 0.2}, // Block nhỏ, reward cao, reset sớm

            // ===== Group 9: Extreme cases để test ranh giới =====
            {0.9999, 0.05, 0.1, 300, 0.8, 0.7, 50, 0.2},  // Adaptive rất chậm
            {0.9999, 0.25, 0.1, 300, 0.8, 0.7, 50, 0.2},  // Adaptive rất nhanh
            {0.9999, 0.15, 0.03, 300, 0.8, 0.7, 50, 0.2}, // Block rất nhỏ
            {0.9999, 0.15, 0.25, 300, 0.8, 0.7, 50, 0.2}, // Block rất lớn
            {0.9999, 0.15, 0.1, 100, 0.8, 0.7, 50, 0.2},  // Reset rất sớm
            {0.9999, 0.15, 0.1, 600, 0.8, 0.7, 50, 0.2},  // Reset rất muộn

            // ===== Group 10: Fine-tuning around best config =====
            // Nếu base config (0.9999, 0.15, 0.1, 300, 0.8, 0.7, 50, 0.2) tốt nhất
            // thử các biến thể nhỏ xung quanh nó:
            {0.9999, 0.14, 0.09, 300, 0.8, 0.7, 50, 0.2},
            {0.9999, 0.16, 0.11, 300, 0.8, 0.7, 50, 0.2},
            {0.9999, 0.15, 0.1, 280, 0.8, 0.7, 50, 0.2},
            {0.9999, 0.15, 0.1, 320, 0.8, 0.7, 50, 0.2},
            {0.9999, 0.15, 0.1, 300, 0.78, 0.68, 50, 0.2},
            {0.9999, 0.15, 0.1, 300, 0.82, 0.72, 50, 0.2},
            {0.9999, 0.15, 0.1, 300, 0.8, 0.7, 45, 0.19},
            {0.9999, 0.15, 0.1, 300, 0.8, 0.7, 55, 0.21},

            // ===== Group 11: Theo giả thuyết =====
            // Giả thuyết 1: Block nhỏ + Reset sớm → Diversification cao
            {0.9999, 0.15, 0.05, 200, 0.8, 0.7, 50, 0.2},

            // Giả thuyết 2: Block lớn + Reset muộn → Intensification cao
            {0.9999, 0.15, 0.20, 500, 0.8, 0.7, 50, 0.2},

            // Giả thuyết 3: Adaptive nhanh + Penalty cao → Ép vào feasible nhanh
            {0.9999, 0.20, 0.1, 300, 0.8, 0.7, 100, 0.2},

            // Giả thuyết 4: Adaptive chậm + T0 cao → Explore lâu hơn
            {0.9999, 0.10, 0.1, 300, 0.8, 0.7, 50, 0.30},

            // Giả thuyết 5: Reward balance - ưu tiên best improvement
            {0.9999, 0.15, 0.1, 300, 0.95, 0.5, 50, 0.2},

            // Giả thuyết 6: Reward balance - ưu tiên SA exploration
            {0.9999, 0.15, 0.1, 300, 0.7, 0.85, 50, 0.2},

        };

        // ✅ Mở file CSV để ghi
        std::ofstream outFile("parameter_tuning_results.csv");

        if (!outFile.is_open())
        {
            std::cerr << "Cannot open output file!\n";
            return;
        }

        // ✅ Ghi header
        outFile << "cooling_rate,adaptation_rate,block_param,reset_threshold,"
                << "phi1,phi2,penalty_factor,t0_factor,best,worst,avg,stddev\n";

        std::cout << "Starting parameter tuning...\n";
        std::cout << "Progress: ";

        int paramIndex = 0;
        for (const auto &p : params)
        {
            paramIndex++;
            std::cout << paramIndex << "/" << params.size() << "... ";
            std::cout.flush();

            // Set parameters
            cooling_rate = p.cooling_rate;
            adaptation_rate = p.adaptation_rate;
            block_parametter = p.block_param;
            reset_threshold = p.reset_threshold;
            phi1_score_factor = p.phi1;
            phi2_score_factor = p.phi2;
            penalty_factor = p.penalty_factor;

            std::vector<double> results;

            // Run 10 times
            for (int run = 0; run < 10; ++run)
            {
                X.clear();
                X.resize(instance.job + 1, 1);
                cost.clear();
                cost.resize(instance.mach + 1, 0);

                upper_bound = calculateUpper();
                lower_bound = calculateLower();
                bound = (1 - instance.ctrl_factor) * lower_bound + instance.ctrl_factor * upper_bound;

                init();
                t0 = p.t0_factor * totalCompletionTime() / log(2);
                ISA();

                results.push_back(totalCompletionTime());
            }

            // ✅ Calculate statistics
            double best = *std::min_element(results.begin(), results.end());
            double worst = *std::max_element(results.begin(), results.end());
            double avg = std::accumulate(results.begin(), results.end(), 0.0) / results.size();

            // Standard deviation
            double variance = 0.0;
            for (double val : results)
                variance += (val - avg) * (val - avg);
            double stddev = std::sqrt(variance / results.size());

            // ✅ Ghi vào file CSV
            outFile << std::fixed
                    << p.cooling_rate << ","
                    << p.adaptation_rate << ","
                    << p.block_param << ","
                    << p.reset_threshold << ","
                    << p.phi1 << ","
                    << p.phi2 << ","
                    << p.penalty_factor << ","
                    << p.t0_factor << ","
                    << best << ","
                    << worst << ","
                    << avg << ","
                    << stddev << "\n";

            outFile.flush(); // Đảm bảo ghi ngay
        }

        outFile.close();
        std::cout << "\n✅ Results saved to: parameter_tuning_results.csv\n";
    }
};

#endif
