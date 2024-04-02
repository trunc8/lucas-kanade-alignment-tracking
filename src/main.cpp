#include "utils.h"
#include "tracker.h"

int main(int argc, char **argv)
{
    bool to_viz_input = false;
    bool to_viz_groundtruth = false;
    bool to_viz_groundtruth_on_tracking = false;

    for (int i = 0; i < argc; i++)
    {
        if (std::string(argv[i]) == "viz_input")
            to_viz_input = true;
        else if (std::string(argv[i]) == "viz_groundtruth")
            to_viz_groundtruth = true;
        else if (std::string(argv[i]) == "viz_groundtruth_on_tracking")
            to_viz_groundtruth_on_tracking = true;
    }

    /*
     * Visualize input frames
     */
    if (to_viz_input)
    {
        utilities::visualizeInputs();
    }

    std::vector<std::vector<int>> groundtruths = utilities::readGroundtruths();

    /*
     * Visualize groundtruth bounding boxes
     */
    if (to_viz_groundtruth)
    {
        utilities::visualizeGroundtruths(groundtruths);
    }

    /*
     * Create tracker object and perform tracking
    */
    Tracker tracker_obj(to_viz_groundtruth_on_tracking, groundtruths);

    auto start = std::chrono::steady_clock::now();

    tracker_obj.performTracking();

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    /*
     * Report on statistics
    */
    printf("\nTotal Time for %ld Frames (Wall Clock): %.4f s\n",
           constants::num_frames, duration / 1.0e3);
    printf("\nAverage Tracking Time Per Frame (Wall Clock): %.4f s\n\n",
           duration / 1.0e3 / constants::num_frames);

    return 0;
}