#include "utils.h"
#include "tracker.h"

int main()
{
    // utilities::visualizeInputs();

    // std::vector<std::vector<int>> groundtruths = utilities::readGroundtruths();
    // utilities::visualizeGroundtruths(groundtruths);

    auto start = std::chrono::steady_clock::now();

    performTracking();

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Wall clock time: %.2f s\n", duration / 1.0e3);

    return 0;
}