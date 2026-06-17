#include <memory>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "landmark_scan_helper.hpp"

static constexpr double LANDMARK_X = 2.35;
static constexpr double LANDMARK_Y = 0.0;

class LandmarkVisualizerNode : public rclcpp::Node
{
public:
  LandmarkVisualizerNode()
  : Node("landmark_visualizer_node")
  {
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/landmark_markers", 10);

    // Timer to continuously publish the static landmark sphere
    static_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      [this]() { publishStaticMarker(); });

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      [this](nav_msgs::msg::Odometry::UniquePtr msg) {
        robot_x_     = msg->pose.pose.position.x;
        robot_y_     = msg->pose.pose.position.y;
        robot_theta_ = getYaw(msg->pose.pose.orientation);
        has_odom_    = true;
      });

    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10,
      [this](sensor_msgs::msg::LaserScan::UniquePtr msg) {
        if (!has_odom_) { return; }

        double r_meas, phi_meas;
        bool detected = detectLandmark(
          *msg,
          robot_x_, robot_y_, robot_theta_,
          LANDMARK_X, LANDMARK_Y,
          r_meas, phi_meas);

        if (detected != last_detected_) {
          last_detected_ = detected;
          if (detected) {
            RCLCPP_INFO(this->get_logger(),
              "LANDMARK DETECTED — r=%.3f m, phi=%.3f rad", r_meas, phi_meas);
          } else {
            RCLCPP_INFO(this->get_logger(), "Landmark lost.");
          }
        }

        publishDetectionMarker(detected, msg->header.stamp);
      });

    RCLCPP_INFO(this->get_logger(),
      "Landmark visualizer started. Landmark at (%.2f, %.2f). "
      "Add MarkerArray '/landmark_markers' in RViz.",
      LANDMARK_X, LANDMARK_Y);
  }

private:
  double getYaw(const geometry_msgs::msg::Quaternion & q_msg)
  {
    tf2::Quaternion q(q_msg.x, q_msg.y, q_msg.z, q_msg.w);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    return yaw;
  }

  void publishStaticMarker()
  {
    visualization_msgs::msg::MarkerArray arr;

    // Blue sphere at the known landmark position
    visualization_msgs::msg::Marker sphere;
    sphere.header.frame_id = "map";
    sphere.header.stamp    = this->now();
    sphere.ns              = "landmark";
    sphere.id              = 0;
    sphere.type            = visualization_msgs::msg::Marker::SPHERE;
    sphere.action          = visualization_msgs::msg::Marker::ADD;
    sphere.pose.position.x = LANDMARK_X;
    sphere.pose.position.y = LANDMARK_Y;
    sphere.pose.position.z = 0.3;
    sphere.pose.orientation.w = 1.0;
    sphere.scale.x = 0.2;
    sphere.scale.y = 0.2;
    sphere.scale.z = 0.2;
    sphere.color.r = 0.0f;
    sphere.color.g = 0.4f;
    sphere.color.b = 1.0f;
    sphere.color.a = 0.9f;
    sphere.lifetime = rclcpp::Duration(0, 0);
    arr.markers.push_back(sphere);

    // Text label above the sphere
    visualization_msgs::msg::Marker label;
    label.header.frame_id = "map";
    label.header.stamp    = this->now();
    label.ns              = "landmark";
    label.id              = 1;
    label.type            = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action          = visualization_msgs::msg::Marker::ADD;
    label.pose.position.x = LANDMARK_X;
    label.pose.position.y = LANDMARK_Y;
    label.pose.position.z = 0.65;
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

  void publishDetectionMarker(bool detected, const rclcpp::Time & stamp)
  {
    visualization_msgs::msg::MarkerArray arr;

    // Small indicator sphere that turns green when detected
    visualization_msgs::msg::Marker ind;
    ind.header.frame_id = "map";
    ind.header.stamp    = stamp;
    ind.ns              = "landmark";
    ind.id              = 2;
    ind.type            = visualization_msgs::msg::Marker::SPHERE;
    ind.action          = visualization_msgs::msg::Marker::ADD;
    ind.pose.position.x = LANDMARK_X;
    ind.pose.position.y = LANDMARK_Y;
    ind.pose.position.z = 0.0;
    ind.pose.orientation.w = 1.0;
    ind.scale.x = 0.35;
    ind.scale.y = 0.35;
    ind.scale.z = 0.1;
    if (detected) {
      ind.color.r = 0.0f; ind.color.g = 1.0f; ind.color.b = 0.0f;
    } else {
      ind.color.r = 1.0f; ind.color.g = 0.0f; ind.color.b = 0.0f;
    }
    ind.color.a  = 0.6f;
    ind.lifetime = rclcpp::Duration(0, 600'000'000);  // 0.6 s
    arr.markers.push_back(ind);

    marker_pub_->publish(arr);
  }

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::TimerBase::SharedPtr static_timer_;

  double robot_x_{0.0}, robot_y_{0.0}, robot_theta_{0.0};
  bool has_odom_{false};
  bool last_detected_{false};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LandmarkVisualizerNode>());
  rclcpp::shutdown();
  return 0;
}