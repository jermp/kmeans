#pragma once

#include <algorithm>
#include <vector>
#include <queue>
#include <thread>
#include <condition_variable>
#include <functional>
#include <mutex>

namespace kmeans {

struct thread_pool {
    thread_pool(const uint64_t num_threads)
      : m_working(0)
    {
        for (uint64_t i = 0; i != num_threads; ++i) {
            m_threads.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(m_queue_mutex);
                        m_cv.wait(lock, [this] { return !m_tasks.empty() || m_stop; });
                        if (m_stop && m_tasks.empty()) return;
                        task = move(m_tasks.front());
                        m_tasks.pop();
                    }

                    task();
                    m_working--;
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

    bool working() const { return m_working != 0; }

    uint64_t num_threads() const { return m_threads.size(); }

    void enqueue(std::function<void()> task) {
        m_working++;
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_tasks.emplace(move(task));
        }
        m_cv.notify_one();
    }

private:
    std::vector<std::thread> m_threads;
    std::queue<std::function<void()>> m_tasks;
    std::mutex m_queue_mutex;
    std::condition_variable m_cv;
    bool m_stop = false;
    std::atomic<uint32_t> m_working;
};

}  // namespace kmeans
