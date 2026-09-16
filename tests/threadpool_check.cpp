#include <RLGymCPP/ThreadPool.h>

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    RLGC::ThreadPool pool;
    std::atomic<int> completed{0};

    // Worker failures must reach the caller after all sibling jobs complete.
    pool.StartJobAsync([] { throw std::runtime_error("intentional worker failure"); });
    for (int i = 0; i < 16; ++i)
        pool.StartJobAsync([&] { ++completed; });
    bool caught = false;
    try {
        pool.WaitUntilDone();
    } catch (const std::runtime_error& error) {
        caught = std::string(error.what()) == "intentional worker failure";
    }
    if (!caught || completed != 16) {
        std::cerr << "Asynchronous failure was not propagated or sibling jobs were lost\n";
        return 1;
    }

    // A consumed failure must not poison subsequent successful work.
    pool.StartJobAsync([&] { ++completed; });
    pool.WaitUntilDone();
    if (completed != 17) return 2;

    caught = false;
    try {
        pool.StartBatchedJobs([&](int index) {
            if (index == 3) throw std::runtime_error("intentional batched failure");
            ++completed;
        }, 8, false);
    } catch (const std::runtime_error& error) {
        caught = std::string(error.what()) == "intentional batched failure";
    }
    if (!caught || completed != 24) {
        std::cerr << "Batched failure was not propagated or sibling jobs were lost\n";
        return 3;
    }
    pool.WaitUntilDone();
    std::cout << "THREADPOOL_CHECK_PASSED async_failure=propagated "
                 "batched_failure=propagated completed=24 recovery=pass\n";
}
