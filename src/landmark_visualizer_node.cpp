#include <memory>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "landmark_scan_helper.hpp"

// Known landmark position — must match the SDF world file and the values in
// the filter nodes.  If you move the pillar in Gazebo, change it here too.
static constexpr double LANDMARK_X   = 1.8;
static constexpr double LANDMARK_Y   = 0.0;

// Association gate threshold (same value as in ekf_node.cpp).
// A detection is only accepted if the projected world position is within
// this distance of the known landmark position.
static constexpr double ASSOC_GATE_M = 0.60;

// ─────────────────────────────────────────────────────────────────────────────
// LandmarkVisualizerNode
//
// This node doesn't do any filtering itself — it's purely for visualization.
// It runs the same detection pipeline as the filter nodes so that RViz can
// show you in real time whether the robot can currently "see" the landmark.
//
// Published to /landmark_markers (MarkerArray):
//   marker id 0 — blue cylinder at the known landmark position (static)
//   marker id 1 — "Landmark" text label above it (static)
//   marker id 2 — floor disc:  green = detected + gate OK
//                              red   = not detected or gate rejected
// ─────────────────────────────────────────────────────────────────────────────

class LandmarkVisualizerNode : public rclcpp::Node
{
public:
  LandmarkVisualizerNode()
  : Node("landmark_visualizer_node")
  {
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/landmark_markers", 10);

    // Republish the static landmark marker every 500 ms so it survives
    // an RViz restart without needing a new scan.
    static_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      [this]() { publishStaticMarker(); });

    // We need the robot's current pose to compute the expected bearing to the
    // landmark (pose-guided detection).  Using /ekf_pose here — it's the most
    // accurate estimate available and also the first one to start up.
    pose_sub_ = this->create_subscription<
      geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/ekf_pose", 10,
        [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) {
          robot_x_     = msg->pose.pose.position.x;
          robot_y_     = msg->pose.pose.position.y;
          robot_theta_ = getYaw(msg->pose.pose.orientation);
          has_pose_    = true;
        });

    // Each new scan triggers a fresh detection attempt.
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10,
      [this](sensor_msgs::msg::LaserScan::UniquePtr msg) {
        // Skip until we have at least one pose estimate.
        if (!has_pose_) return;

        double r_meas, phi_meas;
        const bool detected = detectLandmark(
          *msg,
          robot_x_, robot_y_, robot_theta_,
          LANDMARK_X, LANDMARK_Y,
          r_meas, phi_meas);

        // Even when the scan cluster looks right, check that it projects to
        // where the landmark actually is in the world.  Bad pose estimates can
        // cause false positives without this gate.
        bool accepted = false;
        if (detected) {
          const double wx = robot_x_ + r_meas * std::cos(robot_theta_ + phi_meas);
          const double wy = robot_y_ + r_meas * std::sin(robot_theta_ + phi_meas);
          const double assoc_err =
            std::sqrt((wx - LANDMARK_X) * (wx - LANDMARK_X) +
                      (wy - LANDMARK_Y) * (wy - LANDMARK_Y));
          accepted = (assoc_err < ASSOC_GATE_M);

          if (detected && !accepted) {
            RCLCPP_WARN(this->get_logger(),
              "Landmark cluster found but association gate failed "
              "(error=%.2f m > %.2f m)", assoc_err, ASSOC_GATE_M);
          }
        }

        // Only log state transitions so the console doesn't get spammed.
        if (accepted != last_detected_) {
          last_detected_ = accepted;
          if (accepted) {
            RCLCPP_INFO(this->get_logger(),
              "LANDMARK DETECTED — r=%.3f m, phi=%.3f rad", r_meas, phi_meas);
          } else {
            RCLCPP_INFO(this->get_logger(), "Landmark lost.");
          }
        }

        publishDetectionMarker(accepted, msg->header.stamp);
      });

    RCLCPP_INFO(this->get_logger(),
      "Landmark visualizer started. Landmark at (%.2f, %.2f) in map frame. "
      "Listening to /ekf_pose for robot position.",
      LANDMARK_X, LANDMARK_Y);
  }

private:
  // Extract yaw from a quaternion message.
  double getYaw(const geometry_msgs::msg::Quaternion & q_msg)
  {
    tf2::Quaternion q(q_msg.x, q_msg.y, q_msg.z, q_msg.w);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    return yaw;
  }

  // Publish the blue cylinder and text label at the known landmark position.
  // These never move, so we just keep sending them to keep RViz happy.
  void publishStaticMarker()
  {
    visualization_msgs::msg::MarkerArray arr;

    // Blue cylinder — represents the physical pillar in the world.
    // Height and radius match the Gazebo SDF so it lines up visually.
    visualization_msgs::msg::Marker sphere;
    sphere.header.frame_id = "map";
    sphere.header.stamp    = this->now();
    sphere.ns              = "landmark";
    sphere.id              = 0;
    sphere.type            = visualization_msgs::msg::Marker::CYLINDER;
    sphere.action          = visualization_msgs::msg::Marker::ADD;
    sphere.pose.position.x = LANDMARK_X;
    sphere.pose.position.y = LANDMARK_Y;
    sphere.pose.position.z = 0.3;     // vertical centre of the 0.5 m tall pillar
    sphere.pose.orientation.w = 1.0;
    sphere.scale.x = 0.10;   // diameter = 2 × LANDMARK_PILLAR_RADIUS
    sphere.scale.y = 0.10;
    sphere.scale.z = 0.50;
    sphere.color.r = 0.0f;
    sphere.color.g = 0.4f;
    sphere.color.b = 1.0f;
    sphere.color.a = 0.9f;
    sphere.lifetime = rclcpp::Duration(0, 0);  // 0 = never auto-delete
    arr.markers.push_back(sphere);

    // Text label floating above the pillar.
    visualization_msgs::msg::Marker label;
    label.header.frame_id = "map";
    label.header.stamp    = this->now();
    label.ns              = "landmark";
    label.id              = 1;
    label.type            = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action          = visualization_msgs::msg::Marker::ADD;
    label.pose.position.x = LANDMARK_X;
    label.pose.position.y = LANDMARK_Y;
    label.pose.position.z = 0.75;
    label.pose.orientation.w = 1.0;
    label.scale.z = 0.15;
    label.color.r = 1.0f;
    label.color.g = 1.0f;
    label.color.b = 1.0f;
    label.color.a = 1.0f;
    label.text    = "Landmark";
    label.lifetime = rclcpp::Duration(0, 0);
    arr.markers.push_back(label);

    marker_pub_->publish(arr);
  }

  // Flat disc on the floor below the landmark, coloured by detection status.
  // Green = in view + association gate passed, Red = not currently visible.
  // Short lifetime (0.6 s) so it disappears cleanly if the node stops.
  void publishDetectionMarker(bool accepted, const rclcpp::Time & stamp)
  {
    visualization_msgs::msg::MarkerArray arr;

    visualization_msgs::msg::Marker ind;
    ind.header.frame_id = "map";
    ind.header.stamp    = stamp;
    ind.ns              = "landmark";
    ind.id              = 2;
    ind.type            = visualization_msgs::msg::Marker::CYLINDER;
    ind.action          = visualization_msgs::msg::Marker::ADD;
    ind.pose.position.x = LANDMARK_X;
    ind.pose.position.y = LANDMARK_Y;
    ind.pose.position.z = 0.0;
    ind.pose.orientation.w = 1.0;
    ind.scale.x = 0.35;
    ind.scale.y = 0.35;
    ind.scale.z = 0.05;
    if (accepted) {
      ind.color.r = 0.0f; ind.color.g = 1.0f; ind.color.b = 0.0f;
    } else {
      ind.color.r = 1.0f; ind.color.g = 0.0f; ind.color.b = 0.0f;
    }
    ind.color.a  = 0.7f;
    ind.lifetime = rclcpp::Duration(0, 600'000'000);  // 0.6 s
    arr.markers.push_back(ind);

    marker_pub_->publish(arr);
  }

  // ── Members ──────────────────────────────────────────────────────────────

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::TimerBase::SharedPtr static_timer_;

  double robot_x_{0.0}, robot_y_{0.0}, robot_theta_{0.0};
  bool has_pose_{false};
  bool last_detected_{false};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LandmarkVisualizerNode>());
  rclcpp::shutdown();
  return 0;
}