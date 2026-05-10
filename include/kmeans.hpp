#pragma once

#include <cassert>
#include <cmath>
#include <queue>
#include <random>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "thread_pool.hpp"

namespace kmeans {

typedef float float_type;
typedef std::vector<uint8_t> point;
typedef std::vector<float_type> mean;

namespace details {

/*
    Calculate the square of the distance between two points. The pointer
    overload is the hot one: __restrict__ + a runtime size lets the compiler
    auto-vectorize cleanly with -march=native, which the std::vector form
    sometimes blocks via aliasing concerns.
*/
template <typename T, typename Q>
float_type distance_squared(T const* __restrict__ x, Q const* __restrict__ y, std::size_t n) {
    float_type d_squared = 0;
    for (std::size_t i = 0; i < n; ++i) {
        float_type delta = float_type(x[i]) - float_type(y[i]);
        d_squared += delta * delta;
    }
    return d_squared;
}

template <typename T, typename Q>
float_type distance_squared(std::vector<T> const& x, std::vector<Q> const& y) {
    assert(x.size() == y.size());
    return distance_squared(x.data(), y.data(), x.size());
}

float_type distance(mean const& x, mean const& y) { return std::sqrt(distance_squared(x, y)); }

/*
    Compute the squared distance from each point to a single mean, in parallel.
*/
template <typename RandomAccessIterator>
std::vector<float_type> distances_to_mean(mean const& m, RandomAccessIterator begin,
                                          RandomAccessIterator end, thread_pool& threads) {
    assert(end > begin);
    const uint64_t num_points = end - begin;
    const uint64_t num_threads = threads.num_threads();
    std::vector<float_type> distances(num_points);

    auto worker = [&](uint64_t start, uint64_t stop) {
        for (uint64_t i = start; i != stop; ++i) {
            distances[i] = distance_squared(*(begin + i), m);
        }
    };

    const uint64_t block_size = (num_points + num_threads - 1) / num_threads;
    for (uint64_t t = 0; t != num_threads; ++t) {
        uint64_t start = t * block_size;
        uint64_t stop = std::min(start + block_size, num_points);
        if (start < stop) threads.enqueue([&, start, stop] { worker(start, stop); });
    }
    threads.wait();
    return distances;
}

/*
    For each point, update its stored "distance to closest mean so far" by
    taking the min with its squared distance to a newly added mean.
    Used by k-means++ to avoid recomputing distances to every prior mean.
*/
template <typename RandomAccessIterator>
void update_min_distances(std::vector<float_type>& distances, mean const& new_mean,
                          RandomAccessIterator begin, RandomAccessIterator end,
                          thread_pool& threads) {
    assert(end > begin);
    const uint64_t num_points = end - begin;
    assert(distances.size() == num_points);
    const uint64_t num_threads = threads.num_threads();

    auto worker = [&](uint64_t start, uint64_t stop) {
        for (uint64_t i = start; i != stop; ++i) {
            float_type d = distance_squared(*(begin + i), new_mean);
            if (d < distances[i]) distances[i] = d;
        }
    };

    const uint64_t block_size = (num_points + num_threads - 1) / num_threads;
    for (uint64_t t = 0; t != num_threads; ++t) {
        uint64_t start = t * block_size;
        uint64_t stop = std::min(start + block_size, num_points);
        if (start < stop) threads.enqueue([&, start, stop] { worker(start, stop); });
    }
    threads.wait();
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

    /*
        Maintain a running vector of squared distances from each point to its
        closest mean so far. After adding a new mean we only need to compare
        against that one mean, dropping init from O(N*k^2*d) to O(N*k*d).
    */
    auto distances = details::distances_to_mean(means.front(), begin, end, threads);

    for (uint32_t i = 1; i != k; ++i) {
        /* Pick a random point weighted by squared distance from existing means */
        std::discrete_distribution<uint64_t> generator(distances.begin(), distances.end());
        uint64_t index = generator(rand_engine);
        m.clear();
        auto const& point = *(begin + index);
        for (auto x : point) m.push_back(float_type(x));
        means.push_back(m);

        if (i + 1 != k) {
            details::update_min_distances(distances, means.back(), begin, end, threads);
        }
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

    Parallelized via per-thread (k * point_size) accumulators and a final
    reduction. Falls back to a serial path for small inputs / single-thread
    pools, where the fork/join overhead would dominate (notably the
    binary splits inside kmeans_divisive).
*/
template <typename RandomAccessIterator>
std::vector<mean> calculate_means(RandomAccessIterator begin, RandomAccessIterator end,
                                  std::vector<uint32_t> const& clusters,
                                  std::vector<mean> const& old_means, uint32_t k,
                                  thread_pool& threads) {
    assert(end > begin);
    assert(clusters.size() == static_cast<uint64_t>(end - begin));

    const uint64_t num_points = end - begin;
    const uint64_t point_size = (*begin).size();
    const uint64_t num_threads = threads.num_threads();

    /* Serial fallback: small N or single-threaded pool. */
    if (num_threads <= 1 || num_points < 4096) {
        std::vector<mean> means(k, mean(point_size, 0.0));
        std::vector<uint64_t> count(k, 0);
        for (uint64_t i = 0; i != clusters.size(); ++i) {
            uint32_t c = clusters[i];
            count[c] += 1;
            auto const& point = *(begin + i);
            float_type* m = means[c].data();
            for (uint64_t j = 0; j != point_size; ++j) m[j] += point[j];
        }
        for (uint32_t c = 0; c != k; ++c) {
            if (count[c] == 0) {
                means[c] = old_means[c];
            } else {
                float_type inv = float_type(1) / float_type(count[c]);
                for (uint64_t j = 0; j != point_size; ++j) means[c][j] *= inv;
            }
        }
        return means;
    }

    /* Per-thread (k * point_size) flat sum buffers + per-thread k counts. */
    std::vector<std::vector<float_type>> tl_sums(num_threads,
                                                 std::vector<float_type>(k * point_size, 0));
    std::vector<std::vector<uint64_t>> tl_counts(num_threads, std::vector<uint64_t>(k, 0));

    auto worker = [&](uint64_t t, uint64_t start, uint64_t stop) {
        float_type* sums = tl_sums[t].data();
        uint64_t* counts = tl_counts[t].data();
        for (uint64_t i = start; i != stop; ++i) {
            uint32_t c = clusters[i];
            counts[c] += 1;
            auto const& point = *(begin + i);
            float_type* row = sums + uint64_t(c) * point_size;
            for (uint64_t j = 0; j != point_size; ++j) row[j] += point[j];
        }
    };

    const uint64_t block_size = (num_points + num_threads - 1) / num_threads;
    for (uint64_t t = 0; t != num_threads; ++t) {
        uint64_t start = t * block_size;
        uint64_t stop = std::min(start + block_size, num_points);
        if (start < stop) threads.enqueue([&, t, start, stop] { worker(t, start, stop); });
    }
    threads.wait();

    /* Reduce per-thread accumulators. */
    std::vector<mean> means(k, mean(point_size, 0.0));
    std::vector<uint64_t> count(k, 0);
    for (uint64_t t = 0; t != num_threads; ++t) {
        for (uint32_t c = 0; c != k; ++c) count[c] += tl_counts[t][c];
        const float_type* sums = tl_sums[t].data();
        for (uint32_t c = 0; c != k; ++c) {
            float_type* dst = means[c].data();
            const float_type* src = sums + uint64_t(c) * point_size;
            for (uint64_t j = 0; j != point_size; ++j) dst[j] += src[j];
        }
    }

    for (uint32_t c = 0; c != k; ++c) {
        if (count[c] == 0) {
            means[c] = old_means[c];
        } else {
            float_type inv = float_type(1) / float_type(count[c]);
            for (uint64_t j = 0; j != point_size; ++j) means[c][j] *= inv;
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

/*
    Lloyd's algorithm with Hamerly's triangle-inequality bounds.

    Per point we keep an upper bound u[i] on its distance to its assigned
    centre and a lower bound l[i] on its distance to the next-closest
    centre. Per iteration we compute p[c] (how far each centre moved) and
    s[c] (half the distance from c to its closest other centre); for each
    point u[i] is loosened by p[a] and l[i] is tightened by max p[c'!=a],
    and if u[i] <= max(s[a], l[i]) the point cannot have changed cluster
    so we skip the full distance scan. When that fails we tighten u[i] to
    the exact distance and try again, only computing all k distances when
    necessary.
*/
template <typename RandomAccessIterator>
cluster_data kmeans_lloyd(RandomAccessIterator begin, RandomAccessIterator end,
                          clustering_parameters const& parameters, thread_pool& threads) {
    assert(end > begin);
    assert(parameters.get_k() > 0);
    assert(end - begin >= parameters.get_k());

    cluster_data data;
    const uint32_t k = parameters.get_k();
    const uint64_t num_points = end - begin;
    data.num_clusters = k;

    std::random_device rand_device;
    uint64_t seed = parameters.has_random_seed() ? parameters.get_random_seed() : rand_device();
    data.means = details::random_plusplus(begin, end, k, seed, threads);

    std::vector<float_type> upper(num_points);
    std::vector<float_type> lower(num_points);
    data.clusters.assign(num_points, 0);

    auto parallel_for = [&](auto&& body) {
        const uint64_t nt = threads.num_threads();
        const uint64_t block = (num_points + nt - 1) / nt;
        for (uint64_t t = 0; t != nt; ++t) {
            uint64_t s_idx = t * block;
            uint64_t e_idx = std::min(s_idx + block, num_points);
            if (s_idx < e_idx) threads.enqueue([&, s_idx, e_idx] { body(s_idx, e_idx); });
        }
        threads.wait();
    };

    /* Initial assignment: full scan for every point, populating clusters[],
       upper[] (closest distance) and lower[] (second-closest distance). */
    parallel_for([&](uint64_t start, uint64_t stop) {
        for (uint64_t i = start; i != stop; ++i) {
            auto const& point = *(begin + i);
            float_type best = std::sqrt(details::distance_squared(point, data.means[0]));
            uint32_t best_c = 0;
            float_type second = std::numeric_limits<float_type>::infinity();
            for (uint32_t c = 1; c != k; ++c) {
                float_type d = std::sqrt(details::distance_squared(point, data.means[c]));
                if (d < best) {
                    second = best;
                    best = d;
                    best_c = c;
                } else if (d < second) {
                    second = d;
                }
            }
            data.clusters[i] = best_c;
            upper[i] = best;
            lower[i] = second;
        }
    });

    std::vector<mean> old_means;
    std::vector<float_type> p(k);
    std::vector<float_type> s(k);

    while (true) {
        old_means = std::move(data.means);
        data.means = details::calculate_means(begin, end, data.clusters, old_means, k, threads);
        data.iterations += 1;

        for (uint32_t c = 0; c != k; ++c) {
            p[c] = std::sqrt(details::distance_squared(old_means[c], data.means[c]));
        }

        if ((parameters.has_max_iteration() and
             data.iterations == parameters.get_max_iteration()) or
            (parameters.has_min_delta() and
             details::deltas_below_limit(p, parameters.get_min_delta()))) {
            break;
        }

        /* Top two of p so we can use the second-largest movement when the
           argmax happens to be the point's own cluster. */
        uint32_t p_argmax = 0;
        for (uint32_t c = 1; c != k; ++c)
            if (p[c] > p[p_argmax]) p_argmax = c;
        float_type p_max = p[p_argmax];
        float_type p_max2 = 0;
        for (uint32_t c = 0; c != k; ++c)
            if (c != p_argmax and p[c] > p_max2) p_max2 = p[c];

        /* s[c] = (1/2) * min over c' != c of ||means[c] - means[c']||. */
        std::fill(s.begin(), s.end(), std::numeric_limits<float_type>::infinity());
        for (uint32_t c0 = 0; c0 != k; ++c0) {
            for (uint32_t c1 = c0 + 1; c1 != k; ++c1) {
                float_type d = std::sqrt(details::distance_squared(data.means[c0], data.means[c1]));
                if (d < s[c0]) s[c0] = d;
                if (d < s[c1]) s[c1] = d;
            }
        }
        for (uint32_t c = 0; c != k; ++c) s[c] *= 0.5f;

        /* Bound update + reassignment. Most points hit the early-out and
           skip distance computations entirely. */
        parallel_for([&](uint64_t start, uint64_t stop) {
            for (uint64_t i = start; i != stop; ++i) {
                uint32_t a = data.clusters[i];
                upper[i] += p[a];
                lower[i] -= (a == p_argmax) ? p_max2 : p_max;

                float_type m = std::max(s[a], lower[i]);
                if (upper[i] <= m) continue;

                auto const& point = *(begin + i);
                upper[i] = std::sqrt(details::distance_squared(point, data.means[a]));
                if (upper[i] <= m) continue;

                float_type best = upper[i];
                uint32_t best_c = a;
                float_type second = std::numeric_limits<float_type>::infinity();
                for (uint32_t c = 0; c != k; ++c) {
                    if (c == a) continue;
                    float_type d = std::sqrt(details::distance_squared(point, data.means[c]));
                    if (d < best) {
                        second = best;
                        best = d;
                        best_c = c;
                    } else if (d < second) {
                        second = d;
                    }
                }
                data.clusters[i] = best_c;
                upper[i] = best;
                lower[i] = second;
            }
        });
    }

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

        Q.push(std::move(c));
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
            Q.push(std::move(c0));
            Q.push(std::move(c1));
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
