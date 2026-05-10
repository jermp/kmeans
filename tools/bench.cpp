/*
    Benchmark for kmeans_divisive on synthetic data.

    Generates N uint8 points drawn from K isotropic gaussian centers and
    runs the bisective algorithm with min_cluster_size = N/K so it tries
    to "discover" those K clusters. Reports wall-clock time, the number
    of clusters actually found, total iterations and final MSE.

    Usage: bench N d K threads [sigma] [seed]
*/

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../include/kmeans.hpp"

using namespace kmeans;

using clock_type = std::chrono::steady_clock;

static double seconds_since(clock_type::time_point t0) {
    return std::chrono::duration<double>(clock_type::now() - t0).count();
}

/*
    Generate N uint8 points of dimension d, drawn from G isotropic gaussian
    centers placed at random uint8 positions. Values are clamped to [0,255].
    Returns the points and the ground-truth assignments (point i -> center).
*/
static std::pair<std::vector<point>, std::vector<uint32_t>>
generate_synthetic(uint64_t N, uint64_t d, uint64_t G, double sigma, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> center_dist(0, 255);
    std::vector<std::vector<uint8_t>> centers(G, std::vector<uint8_t>(d));
    for (uint64_t g = 0; g != G; ++g) {
        for (uint64_t j = 0; j != d; ++j) centers[g][j] = uint8_t(center_dist(rng));
    }
    std::normal_distribution<double> noise(0.0, sigma);
    std::uniform_int_distribution<uint64_t> assign(0, G - 1);
    std::vector<point> points(N, point(d));
    std::vector<uint32_t> truth(N);
    for (uint64_t i = 0; i != N; ++i) {
        uint64_t g = assign(rng);
        truth[i] = uint32_t(g);
        auto const& c = centers[g];
        for (uint64_t j = 0; j != d; ++j) {
            double v = double(c[j]) + noise(rng);
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            points[i][j] = uint8_t(v);
        }
    }
    return {std::move(points), std::move(truth)};
}

/*
    Compute mean within-cluster squared distance from points to their
    cluster's centroid.
*/
static double compute_mse(std::vector<point> const& points,
                          std::vector<uint32_t> const& clusters,
                          uint64_t num_clusters) {
    const uint64_t d = points.front().size();
    std::vector<std::vector<double>> centroids(num_clusters, std::vector<double>(d, 0.0));
    std::vector<uint64_t> counts(num_clusters, 0);
    for (uint64_t i = 0; i != points.size(); ++i) {
        uint32_t c = clusters[i];
        counts[c] += 1;
        for (uint64_t j = 0; j != d; ++j) centroids[c][j] += double(points[i][j]);
    }
    for (uint64_t c = 0; c != num_clusters; ++c) {
        if (counts[c] == 0) continue;
        for (uint64_t j = 0; j != d; ++j) centroids[c][j] /= counts[c];
    }
    double sse = 0.0;
    for (uint64_t i = 0; i != points.size(); ++i) {
        uint32_t c = clusters[i];
        double s = 0.0;
        for (uint64_t j = 0; j != d; ++j) {
            double delta = double(points[i][j]) - centroids[c][j];
            s += delta * delta;
        }
        sse += s;
    }
    return sse / points.size();
}

static void usage(const char* prog) {
    std::cerr << "usage: " << prog << " N d K threads [sigma=15] [seed=42]\n"
              << "  N        number of points\n"
              << "  d        size (dimension) of each point vector\n"
              << "  K        target number of clusters (also # of true gaussian centers)\n"
              << "  threads  number of worker threads\n";
}

int main(int argc, char** argv) {
    if (argc < 5) {
        usage(argv[0]);
        return 1;
    }
    const uint64_t N = std::strtoull(argv[1], nullptr, 10);
    const uint64_t d = std::strtoull(argv[2], nullptr, 10);
    const uint64_t K = std::strtoull(argv[3], nullptr, 10);
    const uint64_t num_threads = std::strtoull(argv[4], nullptr, 10);
    const double sigma = (argc > 5) ? std::strtod(argv[5], nullptr) : 15.0;
    const uint64_t seed = (argc > 6) ? std::strtoull(argv[6], nullptr, 10) : 42;

    if (N == 0 || num_threads == 0 || K == 0 || d == 0) {
        usage(argv[0]);
        return 1;
    }
    if (K > N) {
        std::cerr << "error: K (" << K << ") cannot exceed N (" << N << ")\n";
        return 1;
    }

    std::cerr << "config: N=" << N << " d=" << d << " K=" << K
              << " threads=" << num_threads << " sigma=" << sigma << " seed=" << seed << "\n";

    std::cerr << "generating synthetic data ... " << std::flush;
    auto t_gen = clock_type::now();
    auto [points, truth] = generate_synthetic(N, d, K, sigma, seed);
    (void)truth;
    std::cerr << "done in " << seconds_since(t_gen) << "s\n";

    /*
        Set min_cluster_size = N/K so binary splits stop at roughly K leaves.
        min_mse is left unset; kmeans_divisive auto-derives it from the root
        cluster's MSE on the first iteration.
    */
    clustering_parameters params;
    params.set_num_threads(num_threads);
    params.set_random_seed(seed);
    params.set_max_iteration(20);
    params.set_min_cluster_size(std::max<uint64_t>(1, N / K));

    std::cerr << "running kmeans_divisive (min_cluster_size=" << (N / K) << ") ... " << std::flush;
    auto t_run = clock_type::now();
    auto data = kmeans_divisive(points.begin(), points.end(), params);
    double total_s = seconds_since(t_run);
    std::cerr << "done in " << total_s << "s\n";

    double mse = compute_mse(points, data.clusters, data.num_clusters);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "N=" << N << " d=" << d << " target_K=" << K
              << " threads=" << num_threads << "\n"
              << "  found_clusters = " << data.num_clusters << "\n"
              << "  iterations     = " << data.iterations << "\n"
              << "  total_time_s   = " << total_s << "\n"
              << "  mse            = " << std::setprecision(2) << mse << "\n";

    return 0;
}
