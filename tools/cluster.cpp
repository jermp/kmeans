#include <iostream>
#include <fstream>

#include "../include/kmeans.hpp"
#include "../external/cmd_line_parser/include/parser.hpp"

using namespace kmeans;

int main(int argc, char** argv) {
    cmd_line_parser::parser parser(argc, argv);
    parser.add("byte_vectors_filename",
               "The filename of the file with the byte vectors to cluster.", "-i", true);
    parser.add("k", "Number of wanted clusters.", "-k", true);
    parser.add("max_iter", "Number of maximum iterations.", "-m", false);
    parser.add("min_delta", "Minimum difference in means.", "-d", false);
    parser.add("seed", "Random seed for kmeans++ initialization.", "-s", false);

    if (!parser.parse()) return 1;

    auto k = parser.get<uint64_t>("k");
    if (k > (uint64_t(1) << 32)) {
        std::cout << "Error: num. clusters cannot be more than 2^32." << std::endl;
        return 1;
    }

    clustering_parameters params(k);
    if (parser.parsed("max_iter")) params.set_max_iteration(parser.get<uint64_t>("max_iter"));
    if (parser.parsed("min_delta")) params.set_min_delta(parser.get<uint64_t>("min_delta"));
    if (parser.parsed("seed")) params.set_random_seed(parser.get<uint64_t>("seed"));

    std::vector<byte_vec> points;
    {
        std::ifstream in(parser.get<std::string>("byte_vectors_filename"), std::ios::binary);
        uint32_t num_bytes_per_point = 0;
        uint32_t num_points = 0;
        in.read(reinterpret_cast<char*>(&num_bytes_per_point), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&num_points), sizeof(uint32_t));
        points.resize(num_points, byte_vec(num_bytes_per_point));
        for (uint32_t i = 0; i != num_points; ++i) {
            in.read(reinterpret_cast<char*>(points[i].data()), num_bytes_per_point);
        }
        in.close();
    }
    auto labels = kmeans_lloyd(points, params);

    for (auto l : labels) { std::cout << l << " "; }
    std::cout << std::endl;

    return 0;
}