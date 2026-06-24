#include <iostream>
#include <string>
#include <cstring>
#include "HybridTabuSolver.h"

void printUsage(const char* programName)
{
    std::cout << "Usage:\n";
    std::cout << "  " << programName << "                                              # Run big data benchmark (time-limit)\n";
    std::cout << "  " << programName << " --single <file> [runs] [time]               # Run single instance\n";
    std::cout << "  " << programName << " --batch [runs]                              # Run all T_1..T_2160\n";
    std::cout << "  " << programName << " --big [runs]                                # Run big data (time-limit, default 5s)\n";
    std::cout << "  " << programName << " --bigtest [runs] [temp_stop]               # Run big data until T <= temp_stop\n";
    std::cout << "  " << programName << " --tune <f1> [f2 ...] [options]             # Tune parameters\n";
    std::cout << "\nBigtest options:\n";
    std::cout << "  runs       Number of runs per instance    (default: 10)\n";
    std::cout << "  temp_stop  SA stop temperature threshold  (default: 0.001)\n";
    std::cout << "\nTune options (after file list):\n";
    std::cout << "  --runs   <n>    Runs/instance when evaluating   (default: 1)\n";
    std::cout << "  --time   <sec>  Time limit per run when tuning  (default: 2.0)\n";
    std::cout << "  --grid   <n>    Configs in grid search phase     (default: 40)\n";
    std::cout << "  --refine <n>    Configs in refine phase          (default: 30)\n";
    std::cout << "  --local  <n>    Steps in local search phase      (default: 20)\n";
    std::cout << "  --out    <file> CSV output filename              (default: tuning_results.csv)\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << programName << " --single data/T_100.txt 20 3.0\n";
    std::cout << "  " << programName << " --batch 10\n";
    std::cout << "  " << programName << " --big 5\n";
    std::cout << "  " << programName << " --bigtest 10 0.001\n";
    std::cout << "  " << programName << " --bigtest 5 0.01\n";
    std::cout << "  " << programName << " --tune data/T_1.txt data/T_50.txt data/T_100.txt\n";
    std::cout << "  " << programName << " --tune data/T_1.txt data/T_50.txt --runs 2 --time 1.5 --grid 50\n";
}

int main(int argc, char* argv[])
{
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     HYBRID TABU SEARCH + SIMULATED ANNEALING SOLVER       ║\n";
    std::cout << "║     For Parallel Machine Scheduling with TCT + TEC        ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";

    ISA_GLS_Solver solver;

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
        std::cout << "Running big data instances (time-limit mode) with "
                  << numRuns << " runs each\n\n";
        solver.runBigData(numRuns);
        return 0;
    }
    else if (mode == "--bigtest")
    {
        int    numRuns  = (argc >= 3) ? std::stoi(argv[2]) : 10;
        double tempStop = (argc >= 4) ? std::stod(argv[3]) : 0.001;

        std::cout << "Running big data instances (temp-stop mode)\n";
        std::cout << "Stop condition : T <= " << tempStop << "\n";
        std::cout << "Runs per inst  : " << numRuns << "\n";
        std::cout << "Output file    : big_data_results_tempstop.csv\n\n";

        solver.runBigDataUntilTempStop(numRuns, tempStop);
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

        // Collect file list (until we hit a flag starting with --)
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

        // Parse remaining options
        int         runs        = 1;
        double      tunTime     = 5.0;
        int         gridSamples = 40;
        int         refine      = 30;
        int         local       = 20;
        std::string outCSV      = "tuning_results.csv";

        while (i < argc)
        {
            std::string flag = argv[i];
            if ((flag == "--runs" || flag == "--run") && i + 1 < argc)
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

        ParamConfig best = solver.tuneParameters(
            tuneFiles, runs, tunTime, gridSamples, refine, local, outCSV);

        std::cout << "\nRun --single with tuned config? (y/n): ";
        char ans = 'n';
        std::cin >> ans;
        if (ans == 'y' || ans == 'Y')
        {
            std::cout << "Instance file: ";
            std::string fname;
            std::cin >> fname;

            std::cout << "Runs (default 10): ";
            int nr = 10;
            std::cin >> nr;

            std::cout << "Time limit in seconds (default 5.0): ";
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