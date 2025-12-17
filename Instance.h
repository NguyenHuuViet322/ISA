#ifndef INSTANCE_H
#define INSTANCE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

class Instance {
public:
    int mach;
    int job;
    double ctrl_factor;
    std::vector<int> proc_time;
    std::vector<double> unit_cost;

    Instance() : mach(0), job(0), ctrl_factor(0.0) {}

    bool readFromFile(const std::string& filename) {
        std::ifstream fin(filename);
        if (!fin.is_open()) {
            std::cerr << "Không thể mở file: " << filename << std::endl;
            return false;
        }

        std::string key;
        while (fin >> key) {
            if (key == "mach") {
                fin >> mach;
            } else if (key == "job") {
                fin >> job;
            } else if (key == "ctrl_factor") {
                fin >> ctrl_factor;
            } else if (key == "proc_time") {
                readArray(fin, proc_time);
            } else if (key == "unit_cost") {
                readArray(fin, unit_cost);
            }
        }

        sort(proc_time.begin(), proc_time.end());
        sort(unit_cost.begin(), unit_cost.end());

        fin.close();
        return true;
    }

    void print() const {
        std::cout << "mach = " << mach << "\n";
        std::cout << "job = " << job << "\n";
        std::cout << "ctrl_factor = " << ctrl_factor << "\n";
        std::cout << "proc_time = [ ";
        for (auto t : proc_time) std::cout << t << " ";
        std::cout << "]\nunit_cost = [ ";
        for (auto c : unit_cost) std::cout << c << " ";
        std::cout << "]\n";
    }

private:
    template <typename T>
    void readArray(std::ifstream& fin, std::vector<T>& outVec) {
        outVec.clear();
        std::string content, line;

        while (std::getline(fin, line)) {
            content += " " + line;
            if (line.find(']') != std::string::npos)
                break;
        }

        std::stringstream ss(content);
        char ch;
        T value;
        while (ss >> ch) {
            if (ch == '[' || ch == ']') continue;
            ss.putback(ch);
            if (ss >> value)
                outVec.push_back(value);
        }
    }
};

#endif 
