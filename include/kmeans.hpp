#pragma once

#include <cassert>
#include <random>
#include <limits>

#include "thread_pool.hpp"

namespace kmeans {

typedef float float_type;
typedef std::vector<uint8_t> point;
typedef std::vector<float_type> mean;

namespace details {

/*
    Calculate the square of the distance between two points.
*/
template <typename T, typename Q>
float_type distance_squared(std::vector<T> const& x, std::vector<Q> const& y) {
    assert(x.size() == y.size());
    float_type d_squared = 0;
    for (uint64_t i = 0; i != x.size(); ++i) {
        float_type delta = float_type(x[i]) - float_type(y[i]);
        d_squared += delta * delta;
    }
    return d_squared;
}

float_type distance(mean const& x, mean const& y) { return std::sqrt(distance_squared(x, y)); }

/*
    Calculate the smallest distance between each of the data points and any of the input means.
*/
template <typename RandomAccessIterator>
std::vector<float_type> closest_distance(std::vector<mean> const& means, RandomAccessIterator begin,
                                         RandomAccessIterator end, thread_pool& threads) {
    assert(end > begin);
    const uint64_t num_points = end - begin;
    const uint64_t num_threads = threads.num_threads();
    std::vector<float_type> distances;
    distances.resize(num_points);

    auto worker = [&](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i != end; ++i) {
            auto const& point = *(begin + i);
            float_type closest = distance_squared(point, means.front());
            for (auto const& mean : means) {
                float_type distance = distance_squared(point, mean);
                if (distance < closest) closest = distance;
            }
            distances[i] = closest;
        }
    };

    const uint64_t block_size = (num_points + num_threads - 1) / num_threads;
    for (uint64_t t = 0; t != num_threads; ++t) {
        uint64_t start = t * block_size;
        uint64_t end = std::min(start + block_size, num_points);
        if (start < end) {  // avoid empty range
            threads.enqueue([&, start, end] { worker(start, end); });
        }
    }

    threads.wait();

    return distances;
}

/*
    This is an alternate initialization method based on the
    [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) initialization algorithm.
*/
template <typename RandomAccessIterator>
std::vector<mean> random_plusplus(RandomAccessIterator begin, RandomAccessIterator end,  //
                                  const uint32_t k, const uint64_t seed,                 //
                                  thread_pool& threads)                                  //
{
    assert(end > begin);
    const uint64_t num_points = end - begin;
    assert(k > 0);
    assert(num_points > 0);

    std::vector<mean> means;
    means.reserve(k);
    mean m;
    m.reserve((*begin).size());

    /*
        Using a very simple PRBS generator, parameters selected according to
        https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
    */
    std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX>
        rand_engine(seed);

    /* Select first mean at random from the set */
    {
        std::uniform_int_distribution<uint64_t> uniform_generator(0, num_points - 1);
        uint64_t index = uniform_generator(rand_engine);
        auto const& point = *(begin + index);
        for (auto x : point) m.push_back(float_type(x));
        means.push_back(m);
    }

    for (uint32_t i = 1; i != k; ++i) {
        /* Calculate the distance to the closest mean for each data point */
        auto distances = details::closest_distance(means, begin, end, threads);
        /* Pick a random point weighted by the distance from existing means */
        std::discrete_distribution<uint64_t> generator(distances.begin(), distances.end());
        uint64_t index = generator(rand_engine);
        m.clear();
        auto const& point = *(begin + index);
        for (auto x : point) m.push_back(float_type(x));
        means.push_back(m);
    }

    return means;
}

/*
    Calculate the index of the mean a particular data point is closest to (euclidean distance)
*/
std::pair<uint64_t, float_type> closest_mean(point const& point, std::vector<mean> const& means) {
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
    return {index, closest};
}

/*
    Calculate the index of the mean each data point is closest to (euclidean distance).
    We assume there are less than 2^32 clusters.
*/
template <typename RandomAccessIterator>
std::vector<uint32_t> calculate_clusters(RandomAccessIterator begin, RandomAccessIterator end,
                                         std::vector<mean> const& means, thread_pool& threads) {
    assert(end > begin);
    const uint64_t num_points = end - begin;
    const uint64_t num_threads = threads.num_threads();
    std::vector<uint32_t> clusters;
    clusters.resize(num_points);

    auto worker = [&](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i != end; ++i) {
            auto const& point = *(begin + i);
            uint32_t cluster_id = closest_mean(point, means).first;
            clusters[i] = cluster_id;
        }
    };

    const uint64_t block_size = (num_points + num_threads - 1) / num_threads;
    for (uint64_t t = 0; t != num_threads; ++t) {
        uint64_t start = t * block_size;
        uint64_t end = std::min(start + block_size, num_points);
        if (start < end) {  // avoid empty range
            threads.enqueue([&, start, end] { worker(start, end); });
        }
    }

    threads.wait();

    return clusters;
}

/*
    Calculate means based on data points and their cluster assignments.
*/
template <typename RandomAccessIterator>
std::vector<mean> calculate_means(RandomAccessIterator begin, RandomAccessIterator end,
                                  std::vector<uint32_t> const& clusters,
                                  std::vector<mean> const& old_means, uint32_t k) {
    assert(end > begin);
    assert(clusters.size() == static_cast<uint64_t>(end - begin));
    (void)end;  // silence, please!

    const uint64_t point_size = (*begin).size();
    std::vector<mean> means(k, mean(point_size, 0.0));
    std::vector<uint32_t> count(k, 0);

    for (size_t i = 0; i != clusters.size(); ++i) {
        auto& mean = means[clusters[i]];
        count[clusters[i]] += 1;
        auto const& point = *(begin + i);
        for (size_t j = 0; j != point_size; ++j) mean[j] += point[j];
    }

    for (size_t i = 0; i != k; ++i) {
        if (count[i] == 0) {
            means[i] = old_means[i];
        } else {
            for (size_t j = 0; j != point_size; ++j) means[i][j] /= count[i];
        }
    }

    return means;
}

std::vector<float_type> deltas(std::vector<mean> const& old_means, std::vector<mean> const& means) {
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

struct clustering_parameters {
    clustering_parameters()
        : m_has_k(false)
        , m_k(0)
        , m_has_max_iter(false)
        , m_max_iter(0)
        , m_has_min_delta(false)
        , m_min_delta(0)
        , m_has_rand_seed(false)
        , m_rand_seed(0)
        , m_has_min_mse(false)
        , m_min_mse(0)
        , m_has_min_cluster_size(false)
        , m_min_cluster_size(0)
        , m_num_threads(1) {}

    void set_k(uint64_t k) {
        m_k = k;
        m_has_k = true;
    }

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

    void set_min_mse(double min_mse) {
        m_min_mse = min_mse;
        m_has_min_mse = true;
    }

    void set_min_cluster_size(uint64_t min_cluster_size) {
        m_min_cluster_size = min_cluster_size;
        m_has_min_cluster_size = true;
    }

    void set_num_threads(uint64_t num_threads) { m_num_threads = num_threads; }

    bool has_k() const { return m_has_k; }
    bool has_max_iteration() const { return m_has_max_iter; }
    bool has_min_delta() const { return m_has_min_delta; }
    bool has_random_seed() const { return m_has_rand_seed; }

    uint32_t get_k() const { return m_k; };
    uint64_t get_max_iteration() const { return m_max_iter; }
    float_type get_min_delta() const { return m_min_delta; }
    uint64_t get_random_seed() const { return m_rand_seed; }
    double get_min_mse() const { return m_min_mse; }
    uint64_t get_min_cluster_size() const { return m_min_cluster_size; }
    uint64_t get_num_threads() const { return m_num_threads; }

private:
    bool m_has_k;
    uint32_t m_k;
    bool m_has_max_iter;
    uint64_t m_max_iter;
    bool m_has_min_delta;
    float_type m_min_delta;
    bool m_has_rand_seed;
    uint64_t m_rand_seed;
    bool m_has_min_mse;
    double m_min_mse;
    bool m_has_min_cluster_size;
    uint64_t m_min_cluster_size;
    uint64_t m_num_threads;
};

struct cluster_data {
    cluster_data() : iterations(0) {}
    uint64_t iterations;
    uint64_t num_clusters;
    std::vector<mean> means;
    std::vector<uint32_t> clusters;
};

template <typename RandomAccessIterator>
cluster_data kmeans_lloyd(RandomAccessIterator begin, RandomAccessIterator end,
                          clustering_parameters const& parameters, thread_pool& threads) {
    assert(end > begin);
    assert(parameters.get_k() > 0);
    assert(end - begin >= parameters.get_k());

    cluster_data data;
    data.num_clusters = parameters.get_k();
    std::random_device rand_device;
    uint64_t seed = parameters.has_random_seed() ? parameters.get_random_seed() : rand_device();

    std::vector<mean> old_means;
    data.means = details::random_plusplus(begin, end, parameters.get_k(), seed,  //
                                          threads);

    /* calculate new means until convergence is reached or we hit the maximum iteration count */
    do {
        data.clusters = details::calculate_clusters(begin, end, data.means,  //
                                                    threads);
        old_means = std::move(data.means);
        data.means =
            details::calculate_means(begin, end, data.clusters, old_means, parameters.get_k());
        data.iterations += 1;
    } while (
        !(parameters.has_max_iteration() and data.iterations == parameters.get_max_iteration()) and
        !(parameters.has_min_delta() and
          details::deltas_below_limit(details::deltas(old_means, data.means),
                                      parameters.get_min_delta())));

    return data;
}

template <typename RandomAccessIterator>
cluster_data kmeans_divisive(RandomAccessIterator begin, RandomAccessIterator end,
                             clustering_parameters& parameters) {
    assert(end > begin);
    typedef uint32_t index_type;
    const uint64_t num_points = end - begin;
    assert(num_points > 0);

    if (sizeof(index_type) * 8 < 64 and num_points > (1ULL << sizeof(index_type) * 8)) {
        throw std::runtime_error("number of points does not fit in a " +
                                 std::to_string(sizeof(index_type)) + "-byte integer");
    }

    cluster_data data;

    if (num_points == 1) {
        data.num_clusters = 1;
        data.clusters.push_back(0);
        return data;
    }

    struct cluster {
        cluster() {}
        cluster(uint64_t size) { indexes.reserve(size); }
        std::vector<index_type> indexes;
        mean centroid;
        uint64_t id = -1;
        double mse = std::numeric_limits<double>::infinity();  // mean squared error
    };

    struct iterator_adaptor {
        iterator_adaptor(const uint64_t i, RandomAccessIterator begin,
                         std::vector<index_type> const& indexes)
            : m_i(i), m_begin(begin), m_indexes(indexes) {
            assert(i <= indexes.size());
        }

        point const& operator*() const {
            assert(m_i < m_indexes.size());
            return *(m_begin + m_indexes[m_i]);
        }

        iterator_adaptor operator+(const uint64_t i) const {
            return iterator_adaptor(i, m_begin, m_indexes);
        }

        uint64_t operator-(iterator_adaptor const& other) {
            assert(m_i >= other.m_i);
            return m_i - other.m_i;
        }

        bool operator>(iterator_adaptor const& other) const { return m_i > other.m_i; }

    private:
        uint64_t m_i;
        RandomAccessIterator m_begin;
        std::vector<index_type> const& m_indexes;
    };

    std::queue<cluster> Q;

    {
        cluster c(num_points);
        for (uint64_t i = 0; i != num_points; ++i) c.indexes.push_back(i);

        /* calculate centroid as mean of all points */
        std::vector<uint64_t> sum((*begin).size(), 0);  // to avoid overflow
        for (auto index : c.indexes) {
            auto const& point = *(begin + index);
            for (uint64_t i = 0; i != point.size(); ++i) sum[i] += point[i];
        }

        c.centroid.resize((*begin).size(), 0.0);
        for (uint64_t i = 0; i != c.centroid.size(); ++i) {
            c.centroid[i] = static_cast<double>(sum[i]) / c.indexes.size();
        }

        Q.push(c);
    }

    uint64_t id = 0;
    bool first = true;  // first iteration
    std::vector<cluster> atomic_clusters;

    std::cerr << " == min_cluster_size = " << parameters.get_min_cluster_size() << std::endl;
    thread_pool threads(parameters.get_num_threads());

    while (!Q.empty()) {
        auto& c = Q.front();
        const uint64_t num_points_in_cluster = c.indexes.size();

        /* compute mean squared error for the cluster c */
        double mse = 0.0;
        for (auto index : c.indexes) {
            auto const& point = *(begin + index);
            mse += details::distance_squared(point, c.centroid);
        }
        mse /= num_points_in_cluster;

        if (first) {
            parameters.set_min_mse(mse * 0.1);
            std::cerr << " == min_mse = " << parameters.get_min_mse() << std::endl;
            first = false;
        }

        if (mse < parameters.get_min_mse() or
            num_points_in_cluster <= parameters.get_min_cluster_size()) {
            /* finalize cluster */
            cluster ac;
            ac.id = id;
            id += 1;
            ac.indexes.swap(c.indexes);
            ac.centroid.swap(c.centroid);
            ac.mse = mse;
            atomic_clusters.push_back(std::move(ac));
        } else {
            clustering_parameters kmeans_lloyd_params = parameters;
            kmeans_lloyd_params.set_k(2);
            auto data = kmeans_lloyd(iterator_adaptor(0, begin, c.indexes),
                                     iterator_adaptor(num_points_in_cluster, begin, c.indexes),
                                     kmeans_lloyd_params, threads);
            cluster c0(data.clusters.size());
            cluster c1(data.clusters.size());
            c0.centroid.swap(data.means[0]);
            c1.centroid.swap(data.means[1]);
            for (uint64_t i = 0; i != data.clusters.size(); ++i) {
                assert(data.clusters[i] <= 1);
                if (data.clusters[i] == 0) {
                    c0.indexes.push_back(c.indexes[i]);
                } else {
                    c1.indexes.push_back(c.indexes[i]);
                }
            }
            Q.push(c0);
            Q.push(c1);
        }
        Q.pop();
        data.iterations += 1;
    }

    assert(atomic_clusters.size() == id);

    std::vector<cluster> final_clusters;
    final_clusters.reserve(atomic_clusters.size());

    /* take the final clusters */
    for (auto& fc : atomic_clusters) {
        if (fc.mse <= parameters.get_min_mse() and
            fc.indexes.size() >= parameters.get_min_cluster_size()) {
            final_clusters.push_back(std::move(fc));
            fc.mse = -1.0;
        }
    }

    if (final_clusters.size() > 0) {
        /* re-assign the other points */
        std::vector<mean> means(final_clusters.size());
        for (uint64_t i = 0; i != final_clusters.size(); ++i) {
            means[i] = std::move(final_clusters[i].centroid);
        }
        for (auto const& fc : atomic_clusters) {
            if (fc.mse == -1.0) {  // cluster was moved, hence it was final
                continue;
            }
            if (fc.mse > parameters.get_min_mse() or
                fc.indexes.size() < parameters.get_min_cluster_size()) {
                /* re-assign to best cluster */
                for (uint64_t i = 0; i != fc.indexes.size(); ++i) {
                    auto [cluster_id, closest_distance] =
                        details::closest_mean(*(begin + fc.indexes[i]), means);
                    assert(cluster_id < final_clusters.size());
                    final_clusters[cluster_id].indexes.push_back(fc.indexes[i]);
                }
            }
        }
    } else {
        final_clusters.swap(atomic_clusters);
    }

    /* sort by non-increasing cluster size */
    std::sort(final_clusters.begin(), final_clusters.end(), [](auto const& c0, auto const& c1) {
        if (c0.indexes.size() == c1.indexes.size()) return c0.mse < c1.mse;
        return c0.indexes.size() > c1.indexes.size();
    });

    /* re-assign ids */
    for (uint32_t cluster_id = 0; cluster_id != final_clusters.size(); ++cluster_id) {
        final_clusters[cluster_id].id = cluster_id;
    }

    data.num_clusters = final_clusters.size();
    data.clusters.resize(num_points);
    for (auto const& fc : final_clusters) {
        for (auto index : fc.indexes) data.clusters[index] = fc.id;
    }

    return data;
}

}  // namespace kmeans
