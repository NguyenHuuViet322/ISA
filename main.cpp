#include <iostream>
#include <string>
#include <vector>
#include "Instance.h"
#include "Solver.h"

void printUsage(const char* programName)
{
    std::cout << "Usage:\n";
    std::cout << "  " << programName << " --single <file> [runs] [time]      # Run single instance\n";
    std::cout << "  " << programName << " --batch [runs]                     # Run all T_1..T_2160\n";
    std::cout << "  " << programName << " --big [runs]                       # Run big data (time-limit, default 5s)\n";
    std::cout << "  " << programName << " --bigtest [runs] [temp_stop]      # Run big data until T <= temp_stop\n";
    std::cout << "  " << programName << " --help\n";
    std::cout << "\nBigtest defaults: runs=10, temp_stop=0.001\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << programName << " --single data/T_100.txt 20 3.0\n";
    std::cout << "  " << programName << " --batch 10\n";
    std::cout << "  " << programName << " --big 5\n";
    std::cout << "  " << programName << " --bigtest 10 0.001\n";
    std::cout << "  " << programName << " --bigtest 5 0.01\n";
}

int main(int argc, char* argv[])
{
    std::cout << "===== SOLVER PROGRAM =====\n\n";

    Solver solver;

    if (argc == 1)
    {
        std::cout << "No arguments provided.\n";
        printUsage(argv[0]);
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
            return 1;
        }

        std::string filename  = argv[2];
        int         numRuns   = (argc >= 4) ? std::stoi(argv[3]) : 10;
        double      timeLimit = (argc >= 5) ? std::stod(argv[4]) : 5.0;

        std::cout << "Running single instance: " << filename << "\n";
        std::cout << "Runs: " << numRuns << ", Time limit: " << timeLimit << "s\n\n";

        solver.runSingle(filename, numRuns, timeLimit);
    }
    else if (mode == "--batch")
    {
        int numRuns = (argc >= 3) ? std::stoi(argv[2]) : 10;
        std::cout << "Running batch (T_1..T_2160) with " << numRuns << " runs each\n\n";
        solver.runAllInstances(numRuns);
    }
    else if (mode == "--big")
    {
        int numRuns = (argc >= 3) ? std::stoi(argv[2]) : 10;
        std::cout << "Running big data instances (time-limit mode) with "
                  << numRuns << " runs each\n\n";
        solver.runBigData(numRuns);
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
    }
    else
    {
        std::cerr << "Unknown option: " << mode << "\n";
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}