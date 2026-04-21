#include <iostream>
#include <string>
#include <vector>
#include "Instance.h"
#include "Solver.h"

void printUsage(const char* programName)
{
    std::cout << "Usage:\n";
    std::cout << "  " << programName << " --single <file> [runs] [time]\n";
    std::cout << "  " << programName << " --batch [runs]\n";
    std::cout << "  " << programName << " --big [runs]\n";
    std::cout << "  " << programName << " --help\n";
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
        std::cout << "Running big data instances with " << numRuns << " runs each\n\n";
        solver.runBigData(numRuns);
    }
    else
    {
        std::cerr << "Unknown option: " << mode << "\n";
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}