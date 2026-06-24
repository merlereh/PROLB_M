#pragma once

#include <sensor_msgs/msg/laser_scan.hpp>
#include <cmath>
#include <limits>
#include <vector>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────
// Landmark geometry constants
//
// The landmark is a thin pillar (radius 0.05 m) placed at a known position.
// The regular environment cylinders are 0.15 m — three times as wide — which
// is what lets us tell them apart in the scan using the cluster width check.
// ─────────────────────────────────────────────────────────────────────────────
static constexpr double LANDMARK_PILLAR_RADIUS = 0.05;   // m

// Acceptable cluster arc-width range.
// Too narrow  → probably a noise spike, not a real pillar.
// Too wide    → matches a regular cylinder (arc ~ 0.30 m) → reject.
static constexpr double CLUSTER_WIDTH_MIN = 0.04;   // m
static constexpr double CLUSTER_WIDTH_MAX = 0.22;   // m

// ─────────────────────────────────────────────────────────────────────────────
// detectLandmark
//
// Pose-guided single-scan landmark detection.
//
// The idea: instead of scanning the whole laser ring, we know roughly where
// the landmark should be given our current pose estimate.  So we open a small
// angular search window around that expected direction and only look there.
// This makes detection robust to clutter elsewhere in the scan.
//
// The five steps:
//   1. Compute expected range and bearing from the current pose.
//   2. Convert that bearing to a scan index and open a window around it.
//   3. Collect all beams whose range is within `range_tolerance` of the
//      expected range  →  these form a "cluster".
//   4. Check the arc width of the cluster against the landmark's known radius.
//      Regular cylinders are ~3× wider and get rejected here.
//   5. Output:
//        r   = closest range in the cluster + pillar radius
//              (closest surface point → pillar centre)
//        phi = angular centroid of the cluster beams
//              (better estimate than using just the centre beam)
//
// Returns true when a valid detection was made; r_out and phi_out are then
// the measurement z = [r, phi] in the robot's sensor frame.
// ─────────────────────────────────────────────────────────────────────────────

inline bool detectLandmark(
    const sensor_msgs::msg::LaserScan & scan,
    double robot_x,
    double robot_y,
    double robot_theta,
    double landmark_x,
    double landmark_y,
    double & r_out,
    double & phi_out,
    double range_tolerance    = 0.25,   // ± m around the expected range
    double angle_half_win     = 0.35,   // rad, half-width of the search cone
    int    min_cluster_beams  = 2)      // need at least this many hits
{
    const int num_beams = static_cast<int>(scan.ranges.size());
    if (num_beams == 0) return false;

    // ── Step 1: expected measurement from the current pose ────────────────
    //
    // Work out how far away the landmark is and in which direction, then
    // subtract the robot's heading to get the bearing in the sensor frame.

    const double dx          = landmark_x - robot_x;
    const double dy          = landmark_y - robot_y;
    const double r_expected  = std::sqrt(dx * dx + dy * dy);
    const double bear_world  = std::atan2(dy, dx);   // direction in world frame

    // Wrap the bearing into the sensor frame so it matches scan angles.
    const double phi_expected = std::atan2(
        std::sin(bear_world - robot_theta),
        std::cos(bear_world - robot_theta));

    // Find the scan index that points closest to the expected direction.
    const int center_idx = static_cast<int>(
        std::round((phi_expected - scan.angle_min) / scan.angle_increment));

    if (center_idx < 0 || center_idx >= num_beams) {
        // Landmark is behind the robot or outside the scan field of view.
        fprintf(stderr, "[landmark] expected bearing %.2f rad is outside scan range\n",
                phi_expected);
        return false;
    }

    // ── Step 2: collect beams in the search window ────────────────────────
    //
    // Walk left and right from center_idx.  Only keep beams that are also
    // close to the expected range — this filters out walls or other objects
    // that happen to be in the same angular direction.

    const int half_win = static_cast<int>(
        std::ceil(angle_half_win / scan.angle_increment));

    struct Beam { double r; double phi; };
    std::vector<Beam> cluster;
    cluster.reserve(2 * half_win + 1);

    double min_r = std::numeric_limits<double>::max();

    for (int i = center_idx - half_win; i <= center_idx + half_win; ++i) {
        if (i < 0 || i >= num_beams) continue;

        const double r = scan.ranges[i];
        if (!std::isfinite(r) || r < scan.range_min || r > scan.range_max) continue;

        if (std::abs(r - r_expected) <= range_tolerance) {
            const double phi = scan.angle_min + i * scan.angle_increment;
            cluster.push_back({r, phi});
            if (r < min_r) min_r = r;   // track the closest surface point
        }
    }

    if (static_cast<int>(cluster.size()) < min_cluster_beams) {
        fprintf(stderr, "[landmark] only %zu beams in cluster (need %d), "
                        "r_expected=%.2f\n",
                cluster.size(), min_cluster_beams, r_expected);
        return false;
    }

    // ── Step 3: signature check — arc width ───────────────────────────────
    //
    // The cluster spans an arc from phi_min to phi_max.  Multiplying by the
    // expected range converts that angular span to a physical arc length.
    //
    // A regular environment cylinder (radius 0.15 m) produces an arc of
    // roughly 0.30 m, which is above CLUSTER_WIDTH_MAX and gets rejected.
    // Our landmark (radius 0.05 m) gives ~0.10 m, which passes.

    const double phi_min       = cluster.front().phi;
    const double phi_max       = cluster.back().phi;
    const double cluster_width = r_expected * (phi_max - phi_min);   // arc length in m

    if (cluster_width < CLUSTER_WIDTH_MIN || cluster_width > CLUSTER_WIDTH_MAX) {
        fprintf(stderr, "[landmark] cluster width %.3f m is outside [%.2f, %.2f] m — "
                        "probably a regular cylinder or noise\n",
                cluster_width, CLUSTER_WIDTH_MIN, CLUSTER_WIDTH_MAX);
        return false;
    }

    // ── Step 4: compute the output measurement z = [r, phi] ──────────────
    //
    // Range: closest beam gives us the surface of the pillar; add the radius
    // to get the distance to the pillar's centre (which is what the EKF
    // measurement model h(x) predicts).
    //
    // Bearing: use the angular centroid of all cluster beams rather than just
    // the centre beam — slightly more accurate when only a few beams hit.

    double phi_sum = 0.0;
    for (const auto & b : cluster) phi_sum += b.phi;

    r_out   = min_r + LANDMARK_PILLAR_RADIUS;
    phi_out = phi_sum / static_cast<double>(cluster.size());

    fprintf(stderr,
        "[landmark] detected: r=%.3f m, phi=%.3f rad, "
        "cluster=%zu beams, width=%.3f m\n",
        r_out, phi_out, cluster.size(), cluster_width);

    return true;
}