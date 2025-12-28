#include <bits/stdc++.h>
using namespace std;

int main()
{
    const int MACH = 60;
    const int JOB = 700;
    const double CTRL_FACTOR = 0.8;

    // Seed random
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

    // Distributions
    uniform_int_distribution<int> proc_dist(0, 100); // proc_time: 0..100
    uniform_int_distribution<int> cost_dist(1, 10);  // unit_cost: 1..10

    ofstream fout("test.txt");
    if (!fout.is_open())
    {
        cerr << "Cannot open file!\n";
        return 1;
    }

    // Header
    fout << "mach\n"
         << MACH << "\n";
    fout << "job\n"
         << JOB << "\n";
    fout << "ctrl_factor\n"
         << CTRL_FACTOR << "\n";

    // proc_time
    fout << "proc_time\n";
    fout << "[ ";
    for (int i = 0; i < JOB; i++)
    {
        fout << proc_dist(rng);
        if (i < JOB - 1)
            fout << "  ";
    }
    fout << " ]\n";

    // unit_cost
    fout << "unit_cost\n";
    fout << "[ ";
    for (int i = 0; i < MACH; i++)
    {
        fout << cost_dist(rng);
        if (i < MACH - 1)
            fout << "  ";
    }
    fout << " ]\n";

    fout.close();
    cout << "Generated test.txt successfully!\n";

    return 0;
}
