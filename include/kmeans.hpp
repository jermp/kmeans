#pragma once

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

namespace kmeans {

typedef std::vector<uint8_t> byte_vec;
typedef float float_type;

namespace details {

/*
    Calculate the square of the distance between two points.
*/
uint64_t distance_squared(byte_vec const& x, byte_vec const& y) {
    assert(x.size() == y.size());
    uint64_t d_squared = 0;
    for (uint64_t i = 0; i != x.size(); ++i) {
        int64_t delta = int64_t(x[i]) - int64_t(y[i]);
        d_squared += delta * delta;
    }
    return d_squared;
}

float_type distance(byte_vec const& x, byte_vec const& y) {
    return std::sqrt(distance_squared(x, y));
}

/*
    Calculate the smallest distance between each of the data points and any of the input means.
*/
std::vector<float_type> closest_distance(std::vector<byte_vec> const& means,
                                         std::vector<byte_vec> const& points) {
    std::vector<float_type> distances;
    distances.resize(points.size());
#pragma omp parallel for
    for (uint64_t i = 0; i != points.size(); ++i) {
        float_type closest = distance_squared(points[i], means.front());
        for (auto const& mean : means) {
            float_type distance = distance_squared(points[i], mean);
            if (distance < closest) closest = distance;
        }
        distances[i] = closest;
    }
    return distances;
}

/*
    This is an alternate initialization method based on the
    [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) initialization algorithm.
*/
std::vector<byte_vec> random_plusplus(std::vector<byte_vec> const& points, uint32_t k,
                                      uint64_t seed) {
    assert(k > 0);
    assert(points.size() > 0);

    std::vector<byte_vec> means;
    means.reserve(k);

    // Using a very simple PRBS generator, parameters selected according to
    // https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
    std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX>
        rand_engine(seed);

    // Select first mean at random from the set
    {
        std::uniform_int_distribution<uint64_t> uniform_generator(0, points.size() - 1);
        uint64_t index = uniform_generator(rand_engine);
        means.push_back(points[index]);
    }

    for (uint32_t i = 1; i != k; ++i) {
        // Calculate the distance to the closest mean for each data point
        auto distances = details::closest_distance(means, points);

        // Pick a random point weighted by the distance from existing means
        std::discrete_distribution<float_type> generator(distances.begin(), distances.end());
        uint64_t index = generator(rand_engine);
        means.push_back(points[index]);
    }

    // std::cout << "means are:\n";
    // for (auto const& mean : means) {
    //     for (auto x : mean) { std::cout << x << ' '; }
    //     std::cout << std::endl;
    // }

    return means;
}

/*
    Calculate the index of the mean a particular data point is closest to (euclidean distance)
*/
uint64_t closest_mean(byte_vec const& point, std::vector<byte_vec> const& means) {
    assert(!means.empty());
    float_type closest = distance_squared(point, means.front());
    uint64_t index = 0;
    for (uint64_t i = 1; i != means.size(); ++i) {
        float_type distance = distance_squared(point, means[i]);
        if (distance < closest) {
            closest = distance;
            index = i;
        }
    }
    return index;
}

/*
    Calculate the index of the mean each data point is closest to (euclidean distance).
    We assume there are less than 2^32 clusters.
*/
std::vector<uint32_t> calculate_clusters(std::vector<byte_vec> const& points,
                                         std::vector<byte_vec> const& means) {
    std::vector<uint32_t> clusters;
    clusters.resize(points.size());
#pragma omp parallel for
    for (uint64_t i = 0; i != points.size(); ++i) {
        uint32_t cluster_id = closest_mean(points[i], means);
        clusters[i] = cluster_id;
    }
    return clusters;
}

/*
Calculate means based on data points and their cluster assignments.
*/
std::vector<byte_vec> calculate_means(std::vector<byte_vec> const& points,
                                      std::vector<uint32_t> const& clusters,
                                      std::vector<byte_vec> const& old_means, uint32_t k) {
    assert(clusters.size() == points.size());

    const uint64_t point_size = points.front().size();
    std::vector<byte_vec> means(k, byte_vec(point_size, 0));
    std::vector<uint32_t> count(k, 0);

    for (uint64_t i = 0; i != clusters.size(); ++i) {
        assert(clusters[i] < k);
        count[clusters[i]] += 1;
        uint32_t count_value = count[clusters[i]];
        auto& mean = means[clusters[i]];
        auto const& point = points[i];
        for (uint64_t j = 0; j != point_size; ++j) {
            float_type val = std::round(
                (double(mean[j]) * (count_value > 1 ? count_value - 1 : 1) + double(point[j])) /
                count_value);
            assert(val >= 0.0 and val <= double(uint64_t(1) << 8 * sizeof(byte_vec::value_type)));
            mean[j] = val;
        }
    }

    for (size_t i = 0; i != k; ++i) {
        if (count[i] == 0) means[i] = old_means[i];
    }

    // 3 7 10 20
    // 3 + 0 = 3 / 1 = 3
    // 3*1 + 7 = 10 / 2 = 5
    // 5*2 + 10 = 20 / 3 = 7
    // 7*3 + 20 = 41 / 4 = 10

    // 17 8 34
    // 17 + 0 = 17 / 1 = 17
    // 17* 1 + 8 = 25 / 2 = 12
    // 12*2 + 34 = 19

    // for (size_t i = 0; i < clusters.size(); ++i) {
    //     auto& mean = means[clusters[i]];
    //     count[clusters[i]] += 1;
    //     for (size_t j = 0; j != point_size; ++j) { mean[j] += points[i][j]; }
    // }
    // for (size_t i = 0; i < k; ++i) {
    //     std::cout << "count[" << i << "]=" << count[i] << std::endl;
    //     if (count[i] == 0) {
    //         means[i] = old_means[i];
    //     } else {
    //         for (size_t j = 0; j != point_size; ++j) { means[i][j] /= count[i]; }
    //     }
    // }

    // std::cout << "means are:\n";
    // for (auto const& mean : means) {
    //     for (auto x : mean) { std::cout << x << ' '; }
    //     std::cout << std::endl;
    // }

    return means;
}

std::vector<float_type> deltas(std::vector<byte_vec> const& old_means,
                               std::vector<byte_vec> const& means) {
    std::vector<float_type> distances;
    distances.reserve(means.size());
    assert(old_means.size() == means.size());
    for (uint64_t i = 0; i != means.size(); ++i) {
        distances.push_back(distance(means[i], old_means[i]));
    }
    return distances;
}

bool deltas_below_limit(std::vector<float_type> const& deltas, float_type min_delta) {
    for (float_type d : deltas) {
        if (d > min_delta) return false;
    }
    return true;
}

}  // namespace details

class clustering_parameters {
public:
    explicit clustering_parameters(uint32_t k)
        : m_k(k)
        , m_has_max_iter(false)
        , m_max_iter()
        , m_has_min_delta(false)
        , m_min_delta()
        , m_has_rand_seed(false)
        , m_rand_seed() {}

    void set_max_iteration(uint64_t max_iter) {
        m_max_iter = max_iter;
        m_has_max_iter = true;
    }

    void set_min_delta(float_type min_delta) {
        m_min_delta = min_delta;
        m_has_min_delta = true;
    }

    void set_random_seed(uint64_t rand_seed) {
        m_rand_seed = rand_seed;
        m_has_rand_seed = true;
    }

    bool has_max_iteration() const { return m_has_max_iter; }
    bool has_min_delta() const { return m_has_min_delta; }
    bool has_random_seed() const { return m_has_rand_seed; }

    uint32_t get_k() const { return m_k; };
    uint64_t get_max_iteration() const { return m_max_iter; }
    float_type get_min_delta() const { return m_min_delta; }
    uint64_t get_random_seed() const { return m_rand_seed; }

private:
    uint32_t m_k;
    bool m_has_max_iter;
    uint64_t m_max_iter;
    bool m_has_min_delta;
    float_type m_min_delta;
    bool m_has_rand_seed;
    uint64_t m_rand_seed;
};

std::vector<uint32_t> kmeans_lloyd(std::vector<byte_vec> const& points,
                                   clustering_parameters const& parameters) {
    assert(parameters.get_k() > 0);               // k must be greater than zero
    assert(points.size() >= parameters.get_k());  // there must be at least k points

    std::random_device rand_device;
    uint64_t seed = parameters.has_random_seed() ? parameters.get_random_seed() : rand_device();

    std::vector<byte_vec> old_means;
    std::vector<byte_vec> means = details::random_plusplus(points, parameters.get_k(), seed);

    std::vector<uint32_t> clusters;

    // Calculate new means until convergence is reached or we hit the maximum iteration count
    uint64_t iteration = 0;
    do {
        clusters = details::calculate_clusters(points, means);

        // std::cout << "clusters: " << std::endl;
        // for (auto c : clusters) { std::cout << c << " "; }
        // std::cout << std::endl;

        old_means = std::move(means);
        means = details::calculate_means(points, clusters, old_means, parameters.get_k());
        ++iteration;
    } while (!(parameters.has_max_iteration() and iteration == parameters.get_max_iteration()) and
             !(parameters.has_min_delta() and
               details::deltas_below_limit(details::deltas(old_means, means),
                                           parameters.get_min_delta())));

    return clusters;
}

}  // namespace kmeans
