// Helper: detect landmark from LaserScan
// Returns true if landmark is detected, and fills r_out + phi_out
// with the measured range and bearing in robot frame.
//
// Strategy: compute expected bearing + distance to the known landmark
// from the current robot pose, then check the LiDAR beam closest to
// that bearing. If the reading is within DISTANCE_TOLERANCE → detected.

#pragma once
#include <sensor_msgs/msg/laser_scan.hpp>
#include <cmath>
#include <limits>

inline bool detectLandmark(
    const sensor_msgs::msg::LaserScan & scan,
    double robot_x, double robot_y, double robot_theta,
    double landmark_x, double landmark_y,
    double & r_out, double & phi_out)
{
    constexpr double LM_DISTANCE_TOLERANCE = 0.25;
    constexpr double LM_ANGLE_TOLERANCE    = 0.15;

    // Expected distance and bearing to landmark
    const double dx          = landmark_x - robot_x;
    const double dy          = landmark_y - robot_y;
    const double r_expected  = std::sqrt(dx * dx + dy * dy);
    const double bear_world  = std::atan2(dy, dx);
    // bearing in robot frame
    const double phi_expected =
        std::atan2(std::sin(bear_world - robot_theta),
                   std::cos(bear_world - robot_theta));

    const int num_beams = static_cast<int>(scan.ranges.size());
    const int beam_idx  = static_cast<int>(
        std::round((phi_expected - scan.angle_min) / scan.angle_increment));

    if (beam_idx < 0 || beam_idx >= num_beams) {
        return false;
    }

    const int window = static_cast<int>(
        std::ceil(LM_ANGLE_TOLERANCE / scan.angle_increment));

    double best_r     = -1.0;
    double best_phi   = 0.0;
    double best_error = std::numeric_limits<double>::max();

    for (int i = beam_idx - window; i <= beam_idx + window; ++i) {
        if (i < 0 || i >= num_beams) { continue; }
        const double r = scan.ranges[i];
        if (!std::isfinite(r) || r < scan.range_min || r > scan.range_max) { continue; }
        const double error = std::abs(r - r_expected);
        if (error < best_error) {
            best_error = error;
            best_r     = r;
            best_phi   = scan.angle_min + i * scan.angle_increment;
        }
    }

    if (best_r < 0.0 || best_error > LM_DISTANCE_TOLERANCE) {
        return false;
    }

    r_out   = best_r;
    phi_out = best_phi;
    return true;
}