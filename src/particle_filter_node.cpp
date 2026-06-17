#include <memory>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"

#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "particle_filter.hpp"
#include "landmark_scan_helper.hpp"

// Landmark position in map frame
static constexpr double LANDMARK_X = 1.1;
static constexpr double LANDMARK_Y = 1.1;

class ParticleFilterNode : public rclcpp::Node
{
public:
  ParticleFilterNode()
  : Node("pf_node")
  {
    this->declare_parameter<std::string>("resampling_method", "multinomial");
    std::string resampling_method =
      this->get_parameter("resampling_method").as_string();
    filter_.setResamplingMethod(resampling_method);
    RCLCPP_INFO(this->get_logger(),
      "Using resampling method: %s", resampling_method.c_str());

    // /initialpose subscriber — initializes filter in map frame
    // exactly like clicking "2D Pose Estimate" in RViz
    initialpose_subscription_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 10,
        [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) {
          const double x     = msg->pose.pose.position.x;
          const double y     = msg->pose.pose.position.y;
          const double theta = getYaw(msg->pose.pose.orientation);

          Eigen::Vector3d pose;
          pose << x, y, theta;
          filter_.initializeParticlesAroundState(pose);

          odom_initialized_ = false;
          initialized_       = true;

          RCLCPP_INFO(this->get_logger(),
            "pf_node initialized from /initialpose (map frame): "
            "x=%.3f, y=%.3f, theta=%.3f", x, y, theta);
        });

    // /cmd_vel subscriber
    cmd_vel_subscription_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10,
        [this](geometry_msgs::msg::Twist::UniquePtr msg) {
          last_v_        = msg->linear.x;
          last_omega_    = msg->angular.z;
          last_cmd_time_ = this->now();
          has_cmd_vel_   = true;
        });

    // /odom subscriber — used only as DELTA measurement, not absolute pose
    odom_subscription_ =
      this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10,
        [this](nav_msgs::msg::Odometry::UniquePtr msg) {
          if (!initialized_) { return; }

          rclcpp::Time current_time = msg->header.stamp;

          const double x     = msg->pose.pose.position.x;
          const double y     = msg->pose.pose.position.y;
          const double theta = getYaw(msg->pose.pose.orientation);

          // First odom after init: store as reference
          if (!odom_initialized_) {
            last_odom_x_     = x;
            last_odom_y_     = y;
            last_odom_theta_ = theta;
            last_time_       = current_time;
            odom_initialized_ = true;
            return;
          }

          // Compute odom delta
          const double dx     = x - last_odom_x_;
          const double dy     = y - last_odom_y_;
          const double dtheta = correctAngle(theta - last_odom_theta_);

          last_odom_x_     = x;
          last_odom_y_     = y;
          last_odom_theta_ = theta;

          // Rotate delta from odom frame into map frame using predicted heading
          const Vector6d & pred = filter_.predictedState();
          const double map_theta = pred(2);

          const double dx_map = dx * std::cos(map_theta) - dy * std::sin(map_theta);
          const double dy_map = dx * std::sin(map_theta) + dy * std::cos(map_theta);

          // Build absolute measurement in map frame
          Eigen::Vector3d z_odom_map;
          z_odom_map(0) = pred(0) + dx_map;
          z_odom_map(1) = pred(1) + dy_map;
          z_odom_map(2) = correctAngle(pred(2) + dtheta);

          double dt = (current_time - last_time_).seconds();
          last_time_ = current_time;

          if (dt <= 0.0 || dt > 1.0) { return; }
          if (!has_cmd_vel_) { return; }

          double cmd_dt = std::abs((current_time - last_cmd_time_).seconds());
          if (cmd_dt > 0.10) { return; }

          Eigen::Vector2d u;
          u << last_v_, last_omega_;

          Vector6d estimate = filter_.update(u, z_odom_map, dt);

          publishPose(estimate, msg->header.stamp, "map");
        });

    // /scan subscriber — landmark update in map frame
    scan_subscription_ =
      this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 10,
        [this](sensor_msgs::msg::LaserScan::UniquePtr msg) {
          if (!initialized_ || !odom_initialized_) { return; }

          // Use predicted state (map frame) for landmark search
          const Vector6d & pred = filter_.predictedState();

          double r_meas, phi_meas;
          const bool detected = detectLandmark(
            *msg,
            pred(0), pred(1), pred(2),
            LANDMARK_X, LANDMARK_Y,
            r_meas, phi_meas);

          if (!detected) { return; }

          RCLCPP_INFO(this->get_logger(),
            "Landmark detected! r=%.3f, phi=%.3f", r_meas, phi_meas);

          Eigen::Vector2d z_lm;
          z_lm(0) = r_meas;
          z_lm(1) = phi_meas;

          Eigen::Vector2d landmark;
          landmark(0) = LANDMARK_X;
          landmark(1) = LANDMARK_Y;

          Vector6d estimate = filter_.updateLandmark(z_lm, landmark);

          RCLCPP_INFO(this->get_logger(),
            "After landmark correction: x=%.3f, y=%.3f, theta=%.3f",
            estimate(0), estimate(1), estimate(2));

          publishPose(estimate, msg->header.stamp, "map");
        });

    pose_publisher_ =
      this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/pf_pose", 10);

    particle_publisher_ =
      this->create_publisher<geometry_msgs::msg::PoseArray>(
        "/my_particle_cloud", 10);

    RCLCPP_INFO(this->get_logger(),
      "pf_node started. Waiting for /initialpose. Landmark at (%.2f, %.2f)",
      LANDMARK_X, LANDMARK_Y);
  }

private:
  static double correctAngle(double angle)
  {
    return std::atan2(std::sin(angle), std::cos(angle));
  }

  double getYaw(const geometry_msgs::msg::Quaternion & q_msg)
  {
    tf2::Quaternion q(q_msg.x, q_msg.y, q_msg.z, q_msg.w);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    return yaw;
  }

  void publishPose(
    const Vector6d & state,
    const rclcpp::Time & stamp,
    const std::string & frame_id)
  {
    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp    = stamp;
    pose_msg.header.frame_id = frame_id;

    pose_msg.pose.pose.position.x = state(0);
    pose_msg.pose.pose.position.y = state(1);
    pose_msg.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, state(2));
    pose_msg.pose.pose.orientation = tf2::toMsg(q);

    for (int i = 0; i < 36; ++i) { pose_msg.pose.covariance[i] = 0.0; }

    pose_publisher_->publish(pose_msg);

    // Publish particle cloud for RViz
    geometry_msgs::msg::PoseArray pa;
    pa.header = pose_msg.header;
    for (const auto & p : filter_.particles()) {
      geometry_msgs::msg::Pose pose;
      pose.position.x = p(0);
      pose.position.y = p(1);
      pose.position.z = 0.0;
      tf2::Quaternion pq;
      pq.setRPY(0.0, 0.0, p(2));
      pose.orientation = tf2::toMsg(pq);
      pa.poses.push_back(pose);
    }
    particle_publisher_->publish(pa);
  }

  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr                     cmd_vel_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr                       odom_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr                   scan_subscription_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr    pose_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr                    particle_publisher_;

  ParticleFilter filter_;

  double last_odom_x_{0.0};
  double last_odom_y_{0.0};
  double last_odom_theta_{0.0};

  double last_v_{0.0};
  double last_omega_{0.0};

  rclcpp::Time last_cmd_time_;
  rclcpp::Time last_time_;

  bool has_cmd_vel_{false};
  bool initialized_{false};
  bool odom_initialized_{false};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ParticleFilterNode>());
  rclcpp::shutdown();
  return 0;
}