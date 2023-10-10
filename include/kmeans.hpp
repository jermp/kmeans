#pragma once

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>
#include <queue>
#include <limits>

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
std::vector<float_type> closest_distance(std::vector<mean> const& means,
                                         std::vector<point> const& points) {
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
std::vector<mean> random_plusplus(std::vector<point> const& points, uint32_t k, uint64_t seed) {
    assert(k > 0);
    assert(points.size() > 0);

    std::vector<mean> means;
    means.reserve(k);
    mean m;
    m.reserve(points.front().size());

    /*
        Using a very simple PRBS generator, parameters selected according to
        https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
    */
    std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX>
        rand_engine(seed);

    /* Select first mean at random from the set */
    {
        std::uniform_int_distribution<uint64_t> uniform_generator(0, points.size() - 1);
        uint64_t index = uniform_generator(rand_engine);
        auto const& point = points[index];
        for (auto x : point) m.push_back(float_type(x));
        means.push_back(m);
    }

    for (uint32_t i = 1; i != k; ++i) {
        /* Calculate the distance to the closest mean for each data point */
        auto distances = details::closest_distance(means, points);
        /* Pick a random point weighted by the distance from existing means */
        std::discrete_distribution<uint64_t> generator(distances.begin(), distances.end());
        uint64_t index = generator(rand_engine);
        m.clear();
        auto const& point = points[index];
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
std::vector<uint32_t> calculate_clusters(std::vector<point> const& points,
                                         std::vector<mean> const& means) {
    std::vector<uint32_t> clusters;
    clusters.resize(points.size());
#pragma omp parallel for
    for (uint64_t i = 0; i != points.size(); ++i) {
        uint32_t cluster_id = closest_mean(points[i], means).first;
        clusters[i] = cluster_id;
    }
    return clusters;
}

/*
    Calculate means based on data points and their cluster assignments.
*/
std::vector<mean> calculate_means(std::vector<point> const& points,
                                  std::vector<uint32_t> const& clusters,
                                  std::vector<mean> const& old_means, uint32_t k) {
    assert(clusters.size() == points.size());

    const uint64_t point_size = points.front().size();
    std::vector<mean> means(k, mean(point_size, 0.0));
    std::vector<uint32_t> count(k, 0);

    for (size_t i = 0; i != clusters.size(); ++i) {
        auto& mean = means[clusters[i]];
        count[clusters[i]] += 1;
        for (size_t j = 0; j != point_size; ++j) mean[j] += points[i][j];
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
        , m_k()
        , m_has_max_iter(false)
        , m_max_iter()
        , m_has_min_delta(false)
        , m_min_delta()
        , m_has_rand_seed(false)
        , m_rand_seed()
        , m_has_min_mse(false)
        , m_min_mse()
        , m_has_min_cluster_size(false)
        , m_min_cluster_size() {}

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
};

struct cluster_data {
    cluster_data() : iterations(0) {}
    uint64_t iterations;
    uint64_t num_clusters;
    std::vector<mean> means;
    std::vector<uint32_t> clusters;
};

cluster_data kmeans_lloyd(std::vector<point> const& points,
                          clustering_parameters const& parameters) {
    assert(parameters.get_k() > 0);
    assert(points.size() >= parameters.get_k());

    cluster_data data;
    data.num_clusters = parameters.get_k();
    std::random_device rand_device;
    uint64_t seed = parameters.has_random_seed() ? parameters.get_random_seed() : rand_device();

    std::vector<mean> old_means;
    data.means = details::random_plusplus(points, parameters.get_k(), seed);

    /* calculate new means until convergence is reached or we hit the maximum iteration count */
    do {
        data.clusters = details::calculate_clusters(points, data.means);
        old_means = std::move(data.means);
        data.means = details::calculate_means(points, data.clusters, old_means, parameters.get_k());
        data.iterations += 1;
    } while (
        !(parameters.has_max_iteration() and data.iterations == parameters.get_max_iteration()) and
        !(parameters.has_min_delta() and
          details::deltas_below_limit(details::deltas(old_means, data.means),
                                      parameters.get_min_delta())));

    return data;
}

cluster_data kmeans_divisive(std::vector<point> const& points, clustering_parameters& parameters) {
    typedef uint32_t index_type;

    assert(points.size() > 0);
    if (sizeof(index_type) * 8 < 64 and points.size() > (1ULL << sizeof(index_type) * 8)) {
        throw std::runtime_error("number of points does not fit in a " +
                                 std::to_string(sizeof(index_type)) + "-byte integer");
    }

    cluster_data data;

    struct cluster {
        cluster() {}
        cluster(uint64_t size) { indexes.reserve(size); }
        std::vector<index_type> indexes;
        mean centroid;
        uint64_t id = -1;
        double mse = std::numeric_limits<double>::infinity();  // mean squared error
    };

    std::queue<cluster> Q;

    {
        cluster c(points.size());
        for (uint64_t i = 0; i != points.size(); ++i) c.indexes.push_back(i);

        /* calculate centroid as mean of all points */
        std::vector<uint64_t> sum(points.front().size(), 0);  // to avoid overflow
        for (auto index : c.indexes) {
            auto const& point = points[index];
            for (uint64_t i = 0; i != point.size(); ++i) sum[i] += point[i];
        }
        c.centroid.resize(points.front().size(), 0.0);
        for (uint64_t i = 0; i != c.centroid.size(); ++i) {
            c.centroid[i] = static_cast<double>(sum[i]) / c.indexes.size();
        }

        Q.push(c);
    }

    uint64_t id = 0;
    bool first = true;  // first iteration
    std::vector<cluster> final_clusters;

    std::cerr << " == min_cluster_size = " << parameters.get_min_cluster_size() << std::endl;

    while (!Q.empty()) {
        auto& c = Q.front();

        /* compute mean squared error for the cluster c */
        double mse = 0.0;
        for (auto index : c.indexes) {
            mse += details::distance_squared(points[index], c.centroid);
        }
        mse /= c.indexes.size();

        if (first) {
            parameters.set_min_mse(mse * 0.1);
            std::cerr << " == min_mse = " << parameters.get_min_mse() << std::endl;
            first = false;
        }

        if (mse < parameters.get_min_mse() or
            c.indexes.size() <= parameters.get_min_cluster_size()) {
            /* finalize cluster */
            cluster final;
            final.id = id;
            id += 1;
            final.indexes.swap(c.indexes);
            final.centroid.swap(c.centroid);
            final.mse = mse;
            final_clusters.push_back(std::move(final));
        } else {
            std::vector<point> tmp_points;
            tmp_points.reserve(c.indexes.size());
            for (auto index : c.indexes) tmp_points.push_back(points[index]);
            clustering_parameters tmp_params = parameters;
            tmp_params.set_k(2);
            auto data = kmeans_lloyd(tmp_points, tmp_params);
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

    assert(final_clusters.size() == id);

    std::vector<cluster> final;
    final.reserve(final_clusters.size());

    /* take the final clusters */
    for (auto& fc : final_clusters) {
        if (fc.mse <= parameters.get_min_mse() and
            fc.indexes.size() >= parameters.get_min_cluster_size()) {
            final.push_back(std::move(fc));
            fc.mse = -1.0;
        }
    }

    /* re-assign the other points */
    std::vector<mean> means(final.size());
    for (uint64_t i = 0; i != final.size(); ++i) means[i] = std::move(final[i].centroid);
    for (auto const& fc : final_clusters) {
        if (fc.mse == -1.0) {  // cluster was moved, hence it was final
            continue;
        }
        if (fc.mse > parameters.get_min_mse() or
            fc.indexes.size() < parameters.get_min_cluster_size()) {
            /* re-assign to best cluster */
            for (uint64_t i = 0; i != fc.indexes.size(); ++i) {
                assert(points[fc.indexes[i]].size());
                auto [cluster_id, closest_distance] =
                    details::closest_mean(points[fc.indexes[i]], means);
                assert(cluster_id < final.size());
                final[cluster_id].indexes.push_back(fc.indexes[i]);
            }
        }
    }

    /* sort by non-increasing cluster size */
    std::sort(final.begin(), final.end(), [](auto const& c0, auto const& c1) {
        if (c0.indexes.size() == c1.indexes.size()) return c0.mse < c1.mse;
        return c0.indexes.size() > c1.indexes.size();
    });

    /* re-assign ids */
    for (uint32_t cluster_id = 0; cluster_id != final.size(); ++cluster_id) {
        final[cluster_id].id = cluster_id;
    }

    // uint64_t sum = 0;
    // for (auto const& fc : final) {
    //     sum += fc.indexes.size();
    //     std::cerr << "cluster-" << fc.id << ": size = " << fc.indexes.size() << " ("
    //               << (fc.indexes.size() * 100.0) / points.size() << "%); mse = " << fc.mse
    //               << std::endl;
    // }
    // assert(sum == points.size());
    // std::cout << sum << " / " << points.size() << std::endl;

    data.num_clusters = final.size();
    data.clusters.resize(points.size());
    for (auto const& fc : final) {
        for (auto index : fc.indexes) data.clusters[index] = fc.id;
    }

    return data;
}

}  // namespace kmeans
