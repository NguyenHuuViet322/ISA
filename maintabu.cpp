#include <iostream>
#include <string>
#include <cstring>
#include "HybridTabuSolver.h"

void printUsage(const char* programName)
{
    std::cout << "Usage:\n";
    std::cout << "  " << programName << "                                         # Run big data benchmark\n";
    std::cout << "  " << programName << " --single <file> [runs] [time]           # Run single instance\n";
    std::cout << "  " << programName << " --batch [runs]                          # Run all T_1..T_2160\n";
    std::cout << "  " << programName << " --big [runs]                            # Run big data instances\n";
    std::cout << "  " << programName << " --tune <f1> [f2 ...] [options]          # Tune parameters\n";
    std::cout << "\nTune options (sau danh sách file):\n";
    std::cout << "  --runs   <n>    Số runs/instance khi đánh giá   (default: 1)\n";
    std::cout << "  --time   <sec>  Time limit mỗi run khi tuning   (default: 2.0)\n";
    std::cout << "  --grid   <n>    Số config ở phase grid search    (default: 40)\n";
    std::cout << "  --refine <n>    Số config ở phase refine         (default: 30)\n";
    std::cout << "  --local  <n>    Số bước ở phase local search     (default: 20)\n";
    std::cout << "  --out    <file> Tên file CSV lưu log             (default: tuning_results.csv)\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << programName << " --single data/T_100.txt 20 3.0\n";
    std::cout << "  " << programName << " --batch 10\n";
    std::cout << "  " << programName << " --big 5\n";
    std::cout << "  " << programName << " --tune data/T_1.txt data/T_50.txt data/T_100.txt\n";
    std::cout << "  " << programName << " --tune data/T_1.txt data/T_50.txt --runs 2 --time 1.5 --grid 50\n";
}

int main(int argc, char* argv[])
{
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     HYBRID TABU SEARCH + SIMULATED ANNEALING SOLVER       ║\n";
    std::cout << "║     For Parallel Machine Scheduling with TCT + TEC        ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";

    HybridTabuSolver solver;

    if (argc == 1)
    {
        solver.runBigData(10);
        return 0;
    }

    std::string mode = argv[1];

    if (mode == "--help" || mode == "-h")
    {
        printUsage(argv[0]);
        return 0;
    }
    else if (mode == "--single")
    {
        if (argc < 3)
        {
            std::cerr << "Error: --single requires a filename\n";
            printUsage(argv[0]);
            return 1;
        }

        std::string filename  = argv[2];
        int         numRuns   = (argc >= 4) ? std::stoi(argv[3]) : 10;
        double      timeLimit = (argc >= 5) ? std::stod(argv[4]) : 5.0;

        std::cout << "Running single instance: " << filename << "\n";
        std::cout << "Runs: " << numRuns << ", Time limit: " << timeLimit << "s\n\n";

        solver.runSingle(filename, numRuns, timeLimit);
        return 0;
    }
    else if (mode == "--batch")
    {
        int numRuns = (argc >= 3) ? std::stoi(argv[2]) : 10;
        std::cout << "Running batch (T_1..T_2160) with " << numRuns << " runs each\n\n";
        solver.runAllInstances(numRuns);
        return 0;
    }
    else if (mode == "--big")
    {
        int numRuns = (argc >= 3) ? std::stoi(argv[2]) : 10;
        std::cout << "Running big data instances with " << numRuns << " runs each\n\n";
        solver.runBigData(numRuns);
        return 0;
    }
    else if (mode == "--tune")
    {
        if (argc < 3)
        {
            std::cerr << "Error: --tune requires at least one instance file\n";
            printUsage(argv[0]);
            return 1;
        }

        // ── Thu thập danh sách file (cho đến khi gặp flag --xxx) ──────
        std::vector<std::string> tuneFiles;
        int i = 2;
        while (i < argc && argv[i][0] != '-')
        {
            tuneFiles.push_back(argv[i]);
            ++i;
        }

        if (tuneFiles.empty())
        {
            std::cerr << "Error: --tune requires at least one instance file\n";
            printUsage(argv[0]);
            return 1;
        }

        // ── Parse các option còn lại ───────────────────────────────────
        int         runs        = 1;
        double      tunTime     = 5.0;
        int         gridSamples = 40;
        int         refine      = 30;
        int         local       = 20;
        std::string outCSV      = "tuning_results.csv";

        while (i < argc)
        {
            std::string flag = argv[i];
            if ((flag == "--runs"   || flag == "--run") && i + 1 < argc)
                { runs        = std::stoi(argv[++i]); }
            else if (flag == "--time"   && i + 1 < argc)
                { tunTime     = std::stod(argv[++i]); }
            else if (flag == "--grid"   && i + 1 < argc)
                { gridSamples = std::stoi(argv[++i]); }
            else if (flag == "--refine" && i + 1 < argc)
                { refine      = std::stoi(argv[++i]); }
            else if (flag == "--local"  && i + 1 < argc)
                { local       = std::stoi(argv[++i]); }
            else if (flag == "--out"    && i + 1 < argc)
                { outCSV      = argv[++i]; }
            else
            {
                std::cerr << "Warning: unknown tune option '" << flag << "' — ignored\n";
            }
            ++i;
        }

        // ── In tóm tắt cấu hình ───────────────────────────────────────
        std::cout << "Mode        : parameter tuning\n";
        std::cout << "Instances   : ";
        for (const auto& f : tuneFiles) std::cout << f << "  ";
        std::cout << "\n";
        std::cout << "Runs/inst   : " << runs        << "\n";
        std::cout << "Time/run    : " << tunTime     << "s\n";
        std::cout << "Grid        : " << gridSamples << "\n";
        std::cout << "Refine      : " << refine      << "\n";
        std::cout << "Local iter  : " << local       << "\n";
        std::cout << "CSV out     : " << outCSV      << "\n\n";

        // ── Chạy tuning ───────────────────────────────────────────────
        ParamConfig best = solver.tuneParameters(
            tuneFiles, runs, tunTime, gridSamples, refine, local, outCSV);

        // ── Hỏi có chạy luôn với config vừa tìm được không ───────────
        std::cout << "\nRun --single với config vừa tune? (y/n): ";
        char ans = 'n';
        std::cin >> ans;
        if (ans == 'y' || ans == 'Y')
        {
            std::cout << "Nhập file instance để chạy: ";
            std::string fname;
            std::cin >> fname;

            std::cout << "Số runs (default 10): ";
            int nr = 10;
            std::cin >> nr;

            std::cout << "Time limit giây (default 5.0): ";
            double tl = 5.0;
            std::cin >> tl;

            solver.runSingle(fname, nr, tl);
        }

        return 0;
    }
    else
    {
        std::cerr << "Unknown option: " << mode << "\n";
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}