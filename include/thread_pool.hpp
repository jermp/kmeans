#pragma once

#include <algorithm>
#include <atomic>
#include <vector>
#include <queue>
#include <thread>
#include <condition_variable>
#include <functional>
#include <mutex>

namespace kmeans {

struct thread_pool {
    thread_pool(const uint64_t num_threads) : m_working(0) {
        for (uint64_t i = 0; i != num_threads; ++i) {
            m_threads.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(m_queue_mutex);
                        m_cv.wait(lock, [this] { return !m_tasks.empty() || m_stop; });
                        if (m_stop && m_tasks.empty()) return;
                        task = std::move(m_tasks.front());
                        m_tasks.pop();
                    }

                    task();

                    if (m_working.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        std::lock_guard<std::mutex> lock(m_done_mutex);
                        m_done_cv.notify_all();
                    }
                }
            });
        }
    }

    ~thread_pool() {
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        for (auto& thread : m_threads) thread.join();
    }

    void wait() {
        /*
            Hybrid wait: spin briefly, then block on a cv. Most fork-join
            phases here finish in microseconds, where the spin succeeds and
            we avoid cv wake-up latency. Once the spin budget is exhausted
            we fall back to cv-block so the main thread doesn't starve a
            worker when num_threads >= num_cores (a pure spin in that case
            costs ~12% on 4 worker threads on a 4-core box).
        */
        constexpr int kSpinIters = 4096;
        for (int i = 0; i < kSpinIters; ++i) {
            if (m_working.load(std::memory_order_acquire) == 0) return;
        }
        std::unique_lock<std::mutex> lock(m_done_mutex);
        m_done_cv.wait(lock, [this] { return m_working.load(std::memory_order_acquire) == 0; });
    }

    uint64_t num_threads() const { return m_threads.size(); }

    void enqueue(std::function<void()> task) {
        m_working.fetch_add(1, std::memory_order_acq_rel);
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_tasks.emplace(std::move(task));
        }
        m_cv.notify_one();
    }

private:
    std::vector<std::thread> m_threads;
    std::queue<std::function<void()>> m_tasks;
    std::mutex m_queue_mutex;
    std::condition_variable m_cv;
    std::mutex m_done_mutex;
    std::condition_variable m_done_cv;
    bool m_stop = false;
    std::atomic<uint32_t> m_working;
};

}  // namespace kmeans
