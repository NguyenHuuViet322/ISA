#include "Solver_LNS.h"

int main()
{
    LNSParams p;
    p.destroy_ratio = 0.15; // Phá ~15% số job mỗi bước
    p.verbose = true;

    SolverLNS solver(p);
    solver.loadInstance("data/N_N_1.txt");
    // solver.runAllInstances(10);
    solver.solve();
    solver.printSolution();
}