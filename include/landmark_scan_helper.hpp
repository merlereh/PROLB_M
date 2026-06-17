#pragma once

// landmark_scan_helper.hpp
//
// Detects a cylindrical landmark from a 2D LiDAR scan.
//
// Strategy (cylinder-aware):
//   1. Compute expected bearing + distance to the known landmark from
//      the robot's current pose estimate.
//   2. Collect all beams in an angular window around that bearing whose
//      measured range is close to r_expected (candidate cluster).
//   3. Centroid of cluster → actual bearing phi.
//      Minimum range in cluster → actual range r (closest surface point).
//   4. Return (r, phi) as measurement z = [r, phi].

#include <sensor_msgs/msg/laser_scan.hpp>
#include <cmath>
#include <limits>
#include <vector>

inline bool detectLandmark(
    const sensor_msgs::msg::LaserScan & scan,
    double robot_x,
    double robot_y,
    double robot_theta,
    double landmark_x,
    double landmark_y,
    double & r_out,
    double & phi_out)
{
    // Wider tolerance for a thick cylinder (radius ~0.15 m)
    constexpr double RANGE_TOLERANCE   = 0.50;   // war 0.40
constexpr double ANGLE_HALF_WIN    = 0.25;   // war 0.20  
constexpr int    MIN_CLUSTER_BEAMS = 2;       // war 3

    const int num_beams = static_cast<int>(scan.ranges.size());
    if (num_beams == 0) { return false; }

    const double dx         = landmark_x - robot_x;
    const double dy         = landmark_y - robot_y;
    const double r_expected = std::sqrt(dx * dx + dy * dy);
    const double bear_world = std::atan2(dy, dx);

    const double phi_expected =
        std::atan2(std::sin(bear_world - robot_theta),
                   std::cos(bear_world - robot_theta));

    const int center_idx = static_cast<int>(
        std::round((phi_expected - scan.angle_min) / scan.angle_increment));

    if (center_idx < 0 || center_idx >= num_beams) { return false; }

    const int half_win = static_cast<int>(
        std::ceil(ANGLE_HALF_WIN / scan.angle_increment));

    struct Beam { double r; double phi; };
    std::vector<Beam> cluster;
    cluster.reserve(2 * half_win + 1);

    double min_r = std::numeric_limits<double>::max();

    for (int i = center_idx - half_win; i <= center_idx + half_win; ++i) {
        if (i < 0 || i >= num_beams) { continue; }
        const double r = scan.ranges[i];
        if (!std::isfinite(r) || r < scan.range_min || r > scan.range_max) { continue; }
        if (std::abs(r - r_expected) <= RANGE_TOLERANCE) {
            const double phi = scan.angle_min + i * scan.angle_increment;
            cluster.push_back({r, phi});
            if (r < min_r) { min_r = r; }
        }
    }

    if (static_cast<int>(cluster.size()) < MIN_CLUSTER_BEAMS) { return false; }

    double phi_sum = 0.0;
    for (const auto & b : cluster) { phi_sum += b.phi; }

    r_out   = min_r;
    phi_out = phi_sum / static_cast<double>(cluster.size());
    return true;
}