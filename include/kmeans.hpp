#pragma once

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

namespace kmeans {

typedef std::vector<uint8_t> byte_vec;
typedef float float_type;

/*
These functions are all private implementation details and shouldn't be referenced outside of this
file.
*/
namespace details {

/*
Calculate the square of the distance between two points.
*/
uint64_t distance_squared(byte_vec const& point_a, byte_vec const& point_b) {
    assert(point_a.size() == point_b.size());
    uint64_t d_squared = 0;
    for (uint64_t i = 0; i != point_a.size(); ++i) {
        int64_t delta = int64_t(point_a[i]) - int64_t(point_b[i]);
        d_squared += delta * delta;
    }
    return d_squared;
}

float_type distance(byte_vec const& point_a, byte_vec const& point_b) {
    return std::sqrt(distance_squared(point_a, point_b));
}

/*
Calculate the smallest distance between each of the data points and any of the input means.
*/
std::vector<float_type> closest_distance(std::vector<byte_vec> const& means,
                                         std::vector<byte_vec> const& points) {
    std::vector<float_type> distances;
    distances.reserve(points.size());
    for (auto const& point : points) {
        float_type closest = distance_squared(point, means.front());
        for (auto& m : means) {
            float_type distance = distance_squared(point, m);
            if (distance < closest) closest = distance;
        }
        distances.push_back(closest);
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
        // std::cout << "choosing first mean-0 as point of index " << index << std::endl;
        means.push_back(points[index]);
    }

    for (uint32_t i = 1; i != k; ++i) {
        // Calculate the distance to the closest mean for each data point
        auto distances = details::closest_distance(means, points);
        // Pick a random point weighted by the distance from existing means

        // TODO: This might convert floating point weights to ints, distorting the distribution for
        // small weights
        // #if !defined(_MSC_VER) || _MSC_VER >= 1900
        std::discrete_distribution<float_type> generator(distances.begin(), distances.end());
        // #else  // MSVC++ older than 14.0
        //         input_size_t i = 0;
        //         std::discrete_distribution<float_type> generator(
        //             distances.size(), 0.0, 0.0, [&distances, &i](double) { return distances[i++];
        //             });
        // #endif
        uint64_t index = generator(rand_engine);
        // std::cout << "choosing mean-" << i << " as point of index " << index << std::endl;
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
    clusters.reserve(points.size());
    uint64_t count = 0;
    for (auto const& point : points) {
        uint32_t cluster_id = closest_mean(point, means);
        // std::cout << "point-" << count << " assigned to cluster " << cluster_id << std::endl;
        clusters.push_back(cluster_id);
        ++count;
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
            assert(val >= 0.0 and val <= 255.0);
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

/*
clustering_parameters is the configuration used for running the kmeans_lloyd algorithm.

It requires a k value for initialization, and can subsequently be configured with your choice
of optional parameters, including:
* Maximum iteration count; the algorithm will terminate if it reaches this iteration count
  before converging on a solution. The results returned are the means and cluster assignments
  calculated in the last iteration before termination.
* Minimum delta; the algorithm will terminate if the change in position of all means is
  smaller than the specified distance.
* Random seed; if present, this will be used in place of `std::random_device` for kmeans++
  initialization. This can be used to ensure reproducible/deterministic behavior.
*/
class clustering_parameters {
public:
    explicit clustering_parameters(uint32_t k)
        : _k(k)
        , _has_max_iter(false)
        , _max_iter()
        , _has_min_delta(false)
        , _min_delta()
        , _has_rand_seed(false)
        , _rand_seed() {}

    void set_max_iteration(uint64_t max_iter) {
        _max_iter = max_iter;
        _has_max_iter = true;
    }

    void set_min_delta(float_type min_delta) {
        _min_delta = min_delta;
        _has_min_delta = true;
    }

    void set_random_seed(uint64_t rand_seed) {
        _rand_seed = rand_seed;
        _has_rand_seed = true;
    }

    bool has_max_iteration() const { return _has_max_iter; }
    bool has_min_delta() const { return _has_min_delta; }
    bool has_random_seed() const { return _has_rand_seed; }

    uint32_t get_k() const { return _k; };
    uint64_t get_max_iteration() const { return _max_iter; }
    float_type get_min_delta() const { return _min_delta; }
    uint64_t get_random_seed() const { return _rand_seed; }

private:
    uint32_t _k;
    bool _has_max_iter;
    uint64_t _max_iter;
    bool _has_min_delta;
    float_type _min_delta;
    bool _has_rand_seed;
    uint64_t _rand_seed;
};

/*
Implementation of k-means generic across the data type and the dimension of each data item. Expects
the data to be a vector of fixed-size arrays. Generic parameters are the type of the base data (T)
and the dimensionality of each data point (N). All points must have the same dimensionality.

e.g. points of the form (X, Y, Z) would be N = 3.

Takes a `clustering_parameters` struct for algorithm configuration. See the comments for the
`clustering_parameters` struct for more information about the configuration values and how they
affect the algorithm.

Returns a vector containing the cluster number (0 to k-1) for each corresponding element of the
input data vector.

Implementation details:
This implementation of k-means uses [Lloyd's
Algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) with the
[kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) used for initializing the means.

*/

std::vector<uint32_t> kmeans_lloyd(std::vector<byte_vec> const& points,
                                   clustering_parameters const& parameters) {
    assert(parameters.get_k() > 0);               // k must be greater than zero
    assert(points.size() >= parameters.get_k());  // there must be at least k points

    std::random_device rand_device;
    uint64_t seed = parameters.has_random_seed() ? parameters.get_random_seed() : rand_device();

    // std::vector<byte_vec> old_old_means;
    std::vector<byte_vec> old_means;
    std::vector<byte_vec> means = details::random_plusplus(points, parameters.get_k(), seed);

    std::vector<uint32_t> clusters;

    // Calculate new means until convergence is reached or we hit the maximum iteration count
    uint64_t iteration = 0;
    do {
        clusters = details::calculate_clusters(points, means);
        std::cout << "clusters: " << std::endl;
        for (auto c : clusters) { std::cout << c << " "; }
        std::cout << std::endl;

        // old_old_means = old_means;
        old_means = means;
        means = details::calculate_means(points, clusters, old_means, parameters.get_k());
        ++iteration;
    } while (means != old_means /* and means != old_old_means */ and
             !(parameters.has_max_iteration() and iteration == parameters.get_max_iteration()) and
             !(parameters.has_min_delta() and
               details::deltas_below_limit(details::deltas(old_means, means),
                                           parameters.get_min_delta())));

    return clusters;
}

}  // namespace kmeans
