#include <iostream>
#include <fstream>

#include "../include/kmeans.hpp"
#include "../external/cmd_line_parser/include/parser.hpp"

using namespace kmeans;

int main(int argc, char** argv) {
    cmd_line_parser::parser parser(argc, argv);
    parser.add("byte_vectors_filename",
               "The filename of the file with the byte vectors to cluster.", "-i", true);
    parser.add("k", "Number of wanted clusters (for kmeans_lloyd).", "-k", false);
    parser.add("max_iter", "Number of maximum iterations (for kmeans_lloyd).", "-m", false);
    parser.add("min_delta", "Minimum difference in means (for kmeans_lloyd).", "-d", false);
    parser.add("seed", "Random seed for kmeans++ initialization (for kmeans_lloyd).", "-s", false);
    parser.add("batch_size",
               "Use batch mode: compute k clusters for batches of batch_size vectors.", "-b",
               false);
    parser.add("min_mse", "Minimum mean squared error (mse) (for kmeans_divisive).", "--mse",
               false);
    parser.add("min_cluster_size", "Minimum cluster size (for kmeans_divisive).", "--mcs", false);

    if (!parser.parse()) return 1;

    clustering_parameters params;

    if (parser.parsed("k")) {
        uint64_t k = parser.get<uint64_t>("k");
        if (k == 0) {
            std::cerr << "Error: k cannot be 0" << std::endl;
            return 1;
        }
        if (k > (uint64_t(1) << 32)) {
            std::cerr << "Error: number of clusters cannot be more than 2^32." << std::endl;
            return 1;
        }
        params.set_k(k);
    }
    if (parser.parsed("max_iter")) params.set_max_iteration(parser.get<uint64_t>("max_iter"));
    if (parser.parsed("min_delta")) {
        float_type d = parser.get<float_type>("min_delta");
        if (d < 0) {
            std::cerr << "Error: min_delta cannot be < 0" << std::endl;
            return 1;
        }
        params.set_min_delta(d);
    }
    if (parser.parsed("seed")) params.set_random_seed(parser.get<uint64_t>("seed"));

    if (parser.parsed("min_mse")) {
        double mse = parser.get<double>("min_mse");
        if (mse < 0) {
            std::cerr << "Error: mse cannot be < 0" << std::endl;
            return 1;
        }
        params.set_min_mse(mse);
    }

    if (parser.parsed("min_cluster_size")) {
        uint64_t min_cluster_size = parser.get<uint64_t>("min_cluster_size");
        if (min_cluster_size == 0) {
            std::cerr << "Error: min_cluster_size cannot be 0" << std::endl;
            return 1;
        }
        params.set_min_cluster_size(min_cluster_size);
    }

    uint64_t batch_size = 0;
    if (parser.parsed("batch_size")) {
        batch_size = parser.get<uint64_t>("batch_size");
        if (batch_size == 0) {
            std::cerr << "Error: batch_size cannot be 0" << std::endl;
            return 1;
        }
    }

    std::vector<point> points;

    std::ifstream in(parser.get<std::string>("byte_vectors_filename"), std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("error in opening input file");

    uint64_t num_bytes_per_point = 0;
    uint64_t num_points = 0;
    in.read(reinterpret_cast<char*>(&num_bytes_per_point), sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(&num_points), sizeof(uint64_t));

    if (batch_size == 0) batch_size = num_points;

    for (uint64_t batch = 0; num_points != 0; ++batch) {
        batch_size = std::min<uint64_t>(batch_size, num_points);
        points.resize(batch_size, point(num_bytes_per_point));
        for (uint64_t i = 0; i != batch_size; ++i) {
            in.read(reinterpret_cast<char*>(points[i].data()), num_bytes_per_point);
        }
        std::cerr << "batch-" << batch << ": running kmeans for " << points.size() << " points"
                  << std::endl;
        auto data = params.has_k() ? kmeans_lloyd(points, params) : kmeans_divisive(points, params);
        std::cerr << " == terminated after " << data.iterations << " iterations" << std::endl;
        for (auto c : data.clusters) std::cout << c << " ";
        num_points -= batch_size;
    }

    in.close();

    return 0;
}