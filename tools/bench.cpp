/*
    Benchmark for kmeans Lloyd's algorithm on synthetic data.

    Reports per-phase timings (kmeans++ init, calculate_clusters,
    calculate_means) plus total time, final MSE, and iteration count
    so we can compare optimizations without changing solution quality.
*/

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../include/kmeans.hpp"

using namespace kmeans;

using clock_type = std::chrono::steady_clock;

static double seconds_since(clock_type::time_point t0) {
    auto t1 = clock_type::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

/*
    Generate N uint8 points of dimension d, drawn from G isotropic Gaussian
    centers placed at random uint8 means. Values are clamped to [0,255].
*/
static std::vector<point> generate_synthetic(uint64_t N, uint64_t d, uint64_t G,
                                             double sigma, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> center_dist(0, 255);
    std::vector<std::vector<uint8_t>> centers(G, std::vector<uint8_t>(d));
    for (uint64_t g = 0; g != G; ++g) {
        for (uint64_t j = 0; j != d; ++j) centers[g][j] = uint8_t(center_dist(rng));
    }
    std::normal_distribution<double> noise(0.0, sigma);
    std::uniform_int_distribution<uint64_t> assign(0, G - 1);
    std::vector<point> points(N, point(d));
    for (uint64_t i = 0; i != N; ++i) {
        uint64_t g = assign(rng);
        auto const& c = centers[g];
        for (uint64_t j = 0; j != d; ++j) {
            double v = double(c[j]) + noise(rng);
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            points[i][j] = uint8_t(v);
        }
    }
    return points;
}

/*
    Compute total within-cluster sum of squared distances divided by N.
*/
static double compute_mse(std::vector<point> const& points,
                          std::vector<uint32_t> const& clusters,
                          std::vector<mean> const& means) {
    double sse = 0.0;
    for (uint64_t i = 0; i != points.size(); ++i) {
        sse += details::distance_squared(points[i], means[clusters[i]]);
    }
    return sse / points.size();
}

struct phase_times {
    double init = 0;
    double assign = 0;  // calculate_clusters
    double update = 0;  // calculate_means
    double total = 0;
    uint64_t iters = 0;
    double mse = 0;
};

/*
    Re-implements kmeans_lloyd with per-phase timing so we can attribute
    cost. Behaviour matches kmeans.hpp's kmeans_lloyd: kmeans++ init,
    then alternate calculate_clusters / calculate_means until max_iter
    or min_delta.
*/
static phase_times run_lloyd_timed(std::vector<point> const& points, uint32_t k,
                                   uint64_t max_iter, float_type min_delta,
                                   uint64_t seed, uint64_t num_threads) {
    phase_times pt;
    thread_pool threads(num_threads);

    auto t_total = clock_type::now();

    auto t0 = clock_type::now();
    auto means = details::random_plusplus(points.begin(), points.end(), k, seed, threads);
    pt.init = seconds_since(t0);

    std::vector<uint32_t> clusters;
    std::vector<mean> old_means;

    for (uint64_t it = 0; it != max_iter; ++it) {
        auto t1 = clock_type::now();
        clusters = details::calculate_clusters(points.begin(), points.end(), means, threads);
        pt.assign += seconds_since(t1);

        auto t2 = clock_type::now();
        old_means = std::move(means);
        means = details::calculate_means(points.begin(), points.end(), clusters, old_means, k);
        pt.update += seconds_since(t2);

        pt.iters += 1;

        if (min_delta > 0) {
            auto deltas = details::deltas(old_means, means);
            if (details::deltas_below_limit(deltas, min_delta)) break;
        }
    }

    pt.total = seconds_since(t_total);
    pt.mse = compute_mse(points, clusters, means);
    return pt;
}

static void print_header() {
    std::cout << "N,d,k,threads,iters,init_s,assign_s,update_s,total_s,mse\n";
}

static void print_row(uint64_t N, uint64_t d, uint32_t k, uint64_t threads, phase_times const& pt) {
    std::cout << N << ',' << d << ',' << k << ',' << threads << ',' << pt.iters << ','
              << std::fixed << std::setprecision(4) << pt.init << ',' << pt.assign << ','
              << pt.update << ',' << pt.total << ',' << std::setprecision(2) << pt.mse << '\n'
              << std::flush;
}

int main(int argc, char** argv) {
    /* Defaults: a small grid that runs in ~a minute on a 4-core box. */
    std::vector<uint64_t> Ns = {10000, 100000};
    std::vector<uint64_t> ds = {16, 64};
    std::vector<uint32_t> ks = {16, 64};
    std::vector<uint64_t> ts = {1, 4};
    uint64_t max_iter = 20;
    uint64_t G = 32;     // number of true gaussian centers in the data
    double sigma = 20.0; // per-dim stddev of the gaussian noise
    uint64_t seed = 42;
    int repeats = 1;

    /* Optional argv override: bench [grid|big|huge] [repeats] */
    if (argc >= 2) {
        std::string mode = argv[1];
        if (mode == "big") {
            Ns = {100000, 1000000};
            ds = {16, 64, 128};
            ks = {16, 64, 256};
            ts = {1, 4};
            max_iter = 20;
        } else if (mode == "huge") {
            Ns = {1000000};
            ds = {128};
            ks = {256, 1024};
            ts = {1, 4};
            max_iter = 10;
        } else if (mode == "small") {
            Ns = {10000};
            ds = {16};
            ks = {16};
            ts = {1};
            max_iter = 10;
        }
    }
    if (argc >= 3) repeats = std::atoi(argv[2]);
    if (repeats < 1) repeats = 1;

    std::cerr << "bench: repeats=" << repeats << " max_iter=" << max_iter
              << " G=" << G << " sigma=" << sigma << "\n";

    print_header();

    for (uint64_t N : Ns) {
        for (uint64_t d : ds) {
            std::cerr << "generating N=" << N << " d=" << d << " ..." << std::flush;
            auto t_gen = clock_type::now();
            auto points = generate_synthetic(N, d, G, sigma, seed);
            std::cerr << " done in " << seconds_since(t_gen) << "s\n";

            for (uint32_t k : ks) {
                if (k > N) continue;
                for (uint64_t nt : ts) {
                    /* Average phase times over `repeats` runs. Sum then divide. */
                    phase_times agg;
                    for (int r = 0; r != repeats; ++r) {
                        auto pt = run_lloyd_timed(points, k, max_iter, /*min_delta*/ 0,
                                                  seed + r, nt);
                        agg.init += pt.init;
                        agg.assign += pt.assign;
                        agg.update += pt.update;
                        agg.total += pt.total;
                        agg.iters += pt.iters;
                        agg.mse += pt.mse;
                    }
                    agg.init /= repeats;
                    agg.assign /= repeats;
                    agg.update /= repeats;
                    agg.total /= repeats;
                    agg.iters /= repeats;
                    agg.mse /= repeats;
                    print_row(N, d, k, nt, agg);
                }
            }
        }
    }

    return 0;
}
