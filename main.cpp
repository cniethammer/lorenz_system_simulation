#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <numeric>
#include <sstream>


class InvalidParameterError : public std::runtime_error {
public:
    explicit InvalidParameterError(const std::string& message) 
        : std::runtime_error(message) {}
};

class JSONWriter {
public:
    static void writeArray(std::ostream& out, const std::vector<double>& arr) {
        out << "[";
        for (size_t i = 0; i < arr.size(); ++i) {
            out << arr[i];
            if (i < arr.size() - 1) out << ",";
        }
        out << "]";
    }

    static void write2DArray(std::ostream& out, const std::vector<std::vector<double>>& arr) {
        out << "[";
        for (size_t i = 0; i < arr.size(); ++i) {
            writeArray(out, arr[i]);
            if (i < arr.size() - 1) out << ",";
        }
        out << "]";
    }
};

class YAMLWriter {
public:
    static void writeArray(std::ostream& out, const std::vector<double>& arr, int indent = 0) {
        for (const auto& val : arr) {
            out << std::string(indent, ' ') << "- " << val << "\n";
        }
    }

    static void write2DArray(std::ostream& out, const std::vector<std::vector<double>>& arr, int indent = 0) {
        for (const auto& row : arr) {
            out << std::string(indent, ' ') << "-\n";
            writeArray(out, row, indent + 2);
        }
    }
};

struct Config {
    size_t k = 4;
    double x0 = 0.0;
    double xend = 10.0;
    std::vector<double> y0 = {1.0, 1.0, 1.0};
    double tol = 1e-6;
    
    
static Config fromFile(const std::string& filename) {
        Config cfg;
        std::string ext = filename.substr(filename.find_last_of(".") + 1);
        if (ext == "json") {
            cfg = fromJSON(filename);
        } else if (ext == "yaml" || ext == "yml") {
            cfg = fromYAML(filename);
        } else if (ext == "csv") {
            cfg = fromCSV(filename);
        } else if (ext == "txt") {
            cfg = fromTXT(filename);
        } else {
            throw std::runtime_error("Unsupported file format");
        }
        return cfg;
    }

    
    static Config fromTXT(const std::string& filename) {
    Config cfg;
    std::ifstream file(filename);
    file >> cfg.k >> cfg.x0 >> cfg.xend >> cfg.tol;
    cfg.y0.clear();
    double val;
    while (file >> val) {
        cfg.y0.push_back(val);
    }
    return cfg;
}

    static Config fromJSON(const std::string& filename) {
        Config cfg;
        std::ifstream file(filename);
        std::string line, content;
        while (std::getline(file, line)) {
            content += line;
        }
        
    
        size_t pos = 0;
        cfg.k = std::stoul(extractJSONValue(content, "k", pos));
        cfg.x0 = std::stod(extractJSONValue(content, "x0", pos));
        cfg.xend = std::stod(extractJSONValue(content, "xend", pos));
        cfg.tol = std::stod(extractJSONValue(content, "tol", pos));
        cfg.y0 = parseJSONArray(extractJSONValue(content, "y0", pos));
        
        return cfg;
    }

    static Config fromYAML(const std::string& filename) {
    Config cfg;
    std::ifstream file(filename);
    std::string line;
    bool in_y0_section = false;
    
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        if (line.empty()) continue;
        
        if (line.find("k:") == 0) {
            cfg.k = std::stoul(line.substr(line.find(":") + 1));
        }
        else if (line.find("x0:") == 0) {
            cfg.x0 = std::stod(line.substr(line.find(":") + 1));
        }
        else if (line.find("xend:") == 0) {
            cfg.xend = std::stod(line.substr(line.find(":") + 1));
        }
        else if (line.find("tol:") == 0) {
            cfg.tol = std::stod(line.substr(line.find(":") + 1));
        }
        else if (line.find("y0:") == 0) {
            in_y0_section = true;
            cfg.y0.clear();
            continue;
        }
        else if (in_y0_section && line[0] == '-') {
            size_t valueStart = line.find_first_not_of(" \t", 1);
            if (valueStart != std::string::npos) {
                std::string valueStr = line.substr(valueStart);
                try {
                    cfg.y0.push_back(std::stod(valueStr));
                } catch (const std::exception& e) {
                    throw std::runtime_error("Invalid number in y0 array: " + valueStr);
                }
            }
        }
    }
    
    if (cfg.y0.empty()) {
        throw std::runtime_error("No initial conditions (y0) found in YAML file");
    }
    
    return cfg;
}

    static Config fromCSV(const std::string& filename) {
        Config cfg;
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::string header, line;
        if (!std::getline(file, header)) {
            throw std::runtime_error("Empty CSV file");
        }

        if (!std::getline(file, line)) {
            throw std::runtime_error("No data in CSV file");
        }

        std::stringstream ss(line);
        std::string value;

        if (!std::getline(ss, value, ',')) throw std::runtime_error("Missing k value");
        cfg.k = std::stoul(value);

        if (!std::getline(ss, value, ',')) throw std::runtime_error("Missing x0 value");
        cfg.x0 = std::stod(value);

        if (!std::getline(ss, value, ',')) throw std::runtime_error("Missing xend value");
        cfg.xend = std::stod(value);

        if (!std::getline(ss, value, ',')) throw std::runtime_error("Missing tol value");
        cfg.tol = std::stod(value);

        cfg.y0.clear();
        while (std::getline(ss, value, ',')) {
            if (!value.empty()) {
                try {
                    cfg.y0.push_back(std::stod(value));
                } catch (const std::exception& e) {
                    throw std::runtime_error("Invalid y0 value: " + value);
                }
            }
        }

        if (cfg.y0.empty()) {
            throw std::runtime_error("No initial conditions (y0) found in CSV file");
        }

        return cfg;
    }

     
    void validate() const {
        if (k < 1) throw InvalidParameterError("Block size k must be positive");
        if (tol <= 0) throw InvalidParameterError("Tolerance must be positive");
        if (xend <= x0) throw InvalidParameterError("End time must be greater than start time");
        if (y0.empty()) throw InvalidParameterError("Initial conditions required");
        if (x0 < 0) throw InvalidParameterError("Start time must be non-negative");
        if (y0.size() != 3) throw InvalidParameterError("Exactly 3 initial conditions required for Lorenz system");
    }

private:
    static std::string extractJSONValue(const std::string& content, const std::string& key, size_t& pos) {
        pos = content.find("\"" + key + "\"", pos);
        if (pos == std::string::npos) throw std::runtime_error("Key not found: " + key);
        
        pos = content.find(':', pos);
        if (pos == std::string::npos) throw std::runtime_error("Invalid JSON format: missing colon after key " + key);
        pos++;
        
        while (pos < content.length() && std::isspace(content[pos])) pos++;
        
        if (content[pos] == '[') return extractJSONArray(content, pos);
        
        size_t end = pos;
        while (end < content.length() && (std::isdigit(content[end]) || content[end] == '.' || 
               content[end] == 'e' || content[end] == 'E' || content[end] == '-' || content[end] == '+')) {
            end++;
        }
        
        std::string value = content.substr(pos, end - pos);
        pos = end;
        return value;
    }

    static std::string extractJSONArray(const std::string& content, size_t& pos) {
        size_t start = pos;
        int bracketCount = 1;
        pos++; 
        
        while (pos < content.length() && bracketCount > 0) {
            if (content[pos] == '[') bracketCount++;
            else if (content[pos] == ']') bracketCount--;
            pos++;
        }
        
        if (bracketCount > 0) throw std::runtime_error("Invalid JSON format: unclosed array");
        return content.substr(start, pos - start);
    }

    static std::vector<double> parseJSONArray(const std::string& arrayStr) {
        std::vector<double> result;
        size_t pos = 1; 
        
        while (pos < arrayStr.length()) {
            while (pos < arrayStr.length() && (std::isspace(arrayStr[pos]) || arrayStr[pos] == ',')) pos++;
            
            if (arrayStr[pos] == ']') break;
            
            size_t end = pos;
            while (end < arrayStr.length() && (std::isdigit(arrayStr[end]) || arrayStr[end] == '.' || 
                   arrayStr[end] == 'e' || arrayStr[end] == 'E' || arrayStr[end] == '-' || arrayStr[end] == '+')) {
                end++;
            }
            
            if (pos == end) throw std::runtime_error("Invalid JSON format: expected number in array");
            
            std::string numStr = arrayStr.substr(pos, end - pos);
            try {
                result.push_back(std::stod(numStr));
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid number format in JSON array: " + numStr);
            }
            
            pos = end;
        }
        
        return result;
    }

};

class ODESystem {
public:
    virtual void operator()(const std::vector<double>& y, double x, std::vector<double>& dydt) const = 0;
    virtual size_t size() const = 0;
    virtual ~ODESystem() = default;
};

class BlockMethodSolver {
private:
    const ODESystem& ode;
    size_t k;
    size_t m;
    double tol;
    double fac_min, fac_max;
    std::vector<std::vector<double>> a;
    std::vector<double> b;
    MPI_Comm comm;
    int rank, size;

    void initializeCoefficients() {
        a.resize(k, std::vector<double>(k));
        b.resize(k, 1.0 / k);
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < i; ++j) {
                a[i][j] = 1.0 / (k * (k - 1));
            }
        }
    }

    double norm(const std::vector<double>& v) const {
        if (v.empty()) return 0.0;
        
        double local_max = 0.0;
        #pragma omp parallel for reduction(max:local_max)
        for (size_t i = 0; i < v.size(); ++i) {
            local_max = std::max(local_max, std::abs(v[i]));
        }
        
        double global_max;
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        return global_max;
    }

    void computeBlock(double x, const std::vector<double>& y, double h,
                     std::vector<std::vector<double>>& block) {
        const size_t total_points = k - 1;
        
        block.resize(k, std::vector<double>(m));
        block[0] = y;
        
        const size_t points_per_proc = (total_points + size - 1) / size;
        const size_t start_idx = rank * points_per_proc + 1;
        const size_t end_idx = std::min(start_idx + points_per_proc, k);
        
        std::vector<double> local_results((end_idx - start_idx) * m);
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = start_idx; i < end_idx; ++i) {
            std::vector<double> yi(y);
            std::vector<double> temp_f(m);
            
            for (size_t j = 0; j < i; ++j) {
                ode(block[j], x + j * h, temp_f);
                
                for (size_t l = 0; l < m; ++l) {
                    yi[l] += h * a[i][j] * temp_f[l];
                }
            }
            
            std::copy(yi.begin(), yi.end(), local_results.begin() + (i - start_idx) * m);
        }
        
        std::vector<int> recvcounts(size), displs(size);
        for (int i = 0; i < size; ++i) {
            size_t start = i * points_per_proc + 1;
            size_t end = std::min(start + points_per_proc, k);
            recvcounts[i] = (end - start) * m;
            displs[i] = (i > 0) ? displs[i-1] + recvcounts[i-1] : 0;
        }
        
        std::vector<double> all_results(total_points * m);
        
        MPI_Request gather_request;
        MPI_Iallgatherv(local_results.data(), local_results.size(), MPI_DOUBLE,
                        all_results.data(), recvcounts.data(), displs.data(),
                        MPI_DOUBLE, comm, &gather_request);
        
        
        MPI_Wait(&gather_request, MPI_STATUS_IGNORE);
        
        for (size_t i = 1; i < k; ++i) {
            std::copy_n(all_results.begin() + (i - 1) * m, m, block[i].begin());
        }
    }

    double estimateError(const std::vector<std::vector<double>>& block,
                        const std::vector<double>& y_next) {
        double max_error = 0.0;
        #pragma omp parallel for reduction(max:max_error)
        for (size_t j = 0; j < m; ++j) {
            max_error = std::max(max_error, std::abs(block.back()[j] - y_next[j]));
        }
        return max_error;
    }

    double adjustStepSize(double h, double err) {
        const double fac = std::min(fac_max, std::max(fac_min,
            std::pow(tol / std::max(err, 1e-10), 1.0 / (k + 1))));
        return h * fac;
    }

public:
    BlockMethodSolver(const ODESystem& ode, size_t k, double tol = 1e-6,
                     double fac_min = 0.1, double fac_max = 5.0)
        : ode(ode), k(k), m(ode.size()), tol(tol),
          fac_min(fac_min), fac_max(fac_max) {
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        initializeCoefficients();
    }

    ~BlockMethodSolver() {
        MPI_Comm_free(&comm);
    }

    void solve(double x0, const std::vector<double>& y0, double xend,
               std::vector<std::vector<double>>& solution) {
        double h = (xend - x0) / 100.0;
        double x = x0;
        std::vector<double> y = y0;
        
        solution = {y0};

        int step_count = 0;
        int accepted_steps = 0;
        bool solution_completed = false;
        MPI_Request bcast_request = MPI_REQUEST_NULL;

        if (rank == 0) {
            std::cout << "Starting solution...\n";
        }

        while (x < xend && !solution_completed) {
            std::vector<std::vector<double>> block(k, std::vector<double>(m));
            computeBlock(x, y, h, block);

            std::vector<double> y_next = block.back();
            #pragma omp parallel for
            for (size_t i = 0; i < m; ++i) {
                y_next[i] += h * h * 0.1;
            }

            double err = estimateError(block, y_next);
            double h_new = adjustStepSize(h, err);

            if (err <= tol) {
                x += h;
                y = block.back();
                
                if (rank == 0) {
                    for (const auto& step : block) {
                        solution.push_back(step);
                    }
                    if (++accepted_steps % 1000 == 0) {
                        std::cout << "Accepted step " << accepted_steps << ": x = " << x
                                  << ", h = " << h << "\n";
                    }
                }
                
                h = h_new;
                
                if (bcast_request != MPI_REQUEST_NULL) {
                    MPI_Wait(&bcast_request, MPI_STATUS_IGNORE);
                }
                MPI_Ibcast(&x, 1, MPI_DOUBLE, 0, comm, &bcast_request);
            } else {
                h = h_new;
            }

            if (x + h > xend) {
                h = xend - x;
            }

            solution_completed = (x >= xend);
            MPI_Bcast(&solution_completed, 1, MPI_C_BOOL, 0, comm);
            step_count++;
        }

        if (bcast_request != MPI_REQUEST_NULL) {
            MPI_Wait(&bcast_request, MPI_STATUS_IGNORE);
        }

        if (rank == 0) {
            std::cout << "Solution completed.\n";
            std::cout << "Total steps: " << step_count << "\n";
            std::cout << "Accepted steps: " << accepted_steps << "\n";
            std::cout << "Rejection ratio: " << (step_count - accepted_steps) / (double)step_count * 100 << "%\n";
        }
    }

};

class LorenzSystem : public ODESystem {
private:
    double sigma, rho, beta;

public:
    LorenzSystem(double sigma = 10.0, double rho = 28.0, double beta = 8.0 / 3.0)
        : sigma(sigma), rho(rho), beta(beta) {}

    void operator()(const std::vector<double>& y, double x,
                   std::vector<double>& dydt) const override {
        dydt = {
            sigma * (y[1] - y[0]),
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2]
        };
    }

    size_t size() const override { return 3; }
};

void writeJSON(const std::string& filename, const std::vector<std::vector<double>>& solution, double x0, double dx) {
    std::ofstream file(filename);
    file << "{\n  \"x0\": " << x0 << ",\n";
    file << "  \"dx\": " << dx << ",\n";
    file << "  \"solution\": ";
    JSONWriter::write2DArray(file, solution);
    file << "\n}";
}

void writeYAML(const std::string& filename, const std::vector<std::vector<double>>& solution, double x0, double dx) {
    std::ofstream file(filename);
    file << "x0: " << x0 << "\n";
    file << "dx: " << dx << "\n";
    file << "solution:\n";
    YAMLWriter::write2DArray(file, solution, 2);
}

void writeCSV(const std::string& filename, const std::vector<std::vector<double>>& solution, double x0, double dx) {
    std::ofstream file(filename);
    file << "x,y1,y2,y3\n";
    for (size_t i = 0; i < solution.size(); ++i) {
        file << x0 + i * dx << ","
             << solution[i][0] << ","
             << solution[i][1] << ","
             << solution[i][2] << "\n";
    }
}

void writeTXT(const std::string& filename, const std::vector<std::vector<double>>& solution, double x0, double dx) {
    std::ofstream file(filename);
    file << std::setprecision(15);
    for (size_t i = 0; i < solution.size(); ++i) {
        file << x0 + i * dx << " ";
        for (const auto& val : solution[i]) {
            file << val << " ";
        }
        file << "\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        Config cfg;
        if (argc > 1) {
            cfg = Config::fromFile(argv[1]);
        } else if (rank == 0) {
            std::cout << "Enter k, x0, xend, tol, y0 (3 values): ";
            std::cin >> cfg.k >> cfg.x0 >> cfg.xend >> cfg.tol;
            cfg.y0.clear();
            for (int i = 0; i < 3; ++i) {
                double val;
                std::cin >> val;
                cfg.y0.push_back(val);
            }
        }
        
        MPI_Bcast(&cfg.k, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cfg.x0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cfg.xend, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cfg.tol, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank != 0) cfg.y0.resize(3);
        MPI_Bcast(cfg.y0.data(), cfg.y0.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

 cfg.validate();

        LorenzSystem lorenz;
        BlockMethodSolver solver(lorenz, cfg.k, cfg.tol);
        std::vector<std::vector<double>> solution;

        auto start = std::chrono::high_resolution_clock::now();
        solver.solve(cfg.x0, cfg.y0, cfg.xend, solution);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::high_resolution_clock::now() - start);

        if (rank == 0) {
            std::cout << "Final solution:\n";
            for (size_t i = 0; i < solution.back().size(); ++i) {
                std::cout << "y[" << i << "] = " << std::setprecision(10)
                         << solution.back()[i] << "\n";
            }
            std::cout << "Time: " << duration.count() << " ms\n";
            
            double dx = (cfg.xend - cfg.x0) / (solution.size() - 1);
            
            writeCSV("solution.csv", solution, cfg.x0, dx);
            writeJSON("solution.json", solution, cfg.x0, dx);
            writeYAML("solution.yaml", solution, cfg.x0, dx);
            writeTXT("solution.txt", solution, cfg.x0, dx);
        }
    }
    catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Error: " << e.what() << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}