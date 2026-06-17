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

#include "landmark_scan_helper.hpp"
#include "kalman_filter.hpp"

// Known landmark position in map frame (arena corner)
static constexpr double LANDMARK_X = 2.35;
static constexpr double LANDMARK_Y = 0.0;

class KalmanFilterNode : public rclcpp::Node
{
public:
  KalmanFilterNode()
  : Node("kf_node")
  {
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

    // /odom subscriber — measurement z = [x, y, theta]
    odom_subscription_ =
      this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10,
        [this](nav_msgs::msg::Odometry::UniquePtr msg) {
          rclcpp::Time current_time = msg->header.stamp;

          const double x     = msg->pose.pose.position.x;
          const double y     = msg->pose.pose.position.y;
          const double theta = getYaw(msg->pose.pose.orientation);

          // Cache current pose for landmark detection
          robot_x_     = x;
          robot_y_     = y;
          robot_theta_ = theta;

          Eigen::Vector3d z_odom;
          z_odom << x, y, theta;

          if (!initialized_) {
            filter_.setState(z_odom);
            last_time_   = current_time;
            initialized_ = true;
            RCLCPP_INFO(this->get_logger(),
              "kf_node initialized: x=%.3f, y=%.3f, theta=%.3f", x, y, theta);
            return;
          }

          double dt = (current_time - last_time_).seconds();
          last_time_ = current_time;

          if (dt <= 0.0 || dt > 1.0) { return; }
          if (!has_cmd_vel_) { return; }

          double cmd_dt = std::abs((current_time - last_cmd_time_).seconds());
          if (cmd_dt > 0.10) { return; }

          Eigen::Vector2d u;
          u << last_v_, last_omega_;

          Vector6d estimate = filter_.update(u, z_odom, dt);

          publishPose(estimate, msg->header.stamp, msg->header.frame_id);
        });

    // /scan subscriber — landmark detection integrated here
    scan_subscription_ =
      this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 10,
        [this](sensor_msgs::msg::LaserScan::UniquePtr msg) {
          if (!initialized_) { return; }

          double r_meas, phi_meas;
          const bool detected = detectLandmark(
            *msg,
            robot_x_, robot_y_, robot_theta_,
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

          Vector6d estimate = filter_.correctLandmark(z_lm, landmark);

          RCLCPP_INFO(this->get_logger(),
            "After landmark correction: x=%.3f, y=%.3f, theta=%.3f",
            estimate(0), estimate(1), estimate(2));

          publishPose(estimate, msg->header.stamp, "odom");
        });

    pose_publisher_ =
      this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/kf_pose", 10);

    RCLCPP_INFO(this->get_logger(),
      "kf_node started. State: [x, y, theta, vx, vy, theta_dot]");
    RCLCPP_INFO(this->get_logger(),
      "Landmark at (%.2f, %.2f)", LANDMARK_X, LANDMARK_Y);
  }

private:
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
    pose_msg.header.frame_id = frame_id.empty() ? "odom" : frame_id;

    pose_msg.pose.pose.position.x = state(0);
    pose_msg.pose.pose.position.y = state(1);
    pose_msg.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, state(2));
    pose_msg.pose.pose.orientation = tf2::toMsg(q);

    for (int i = 0; i < 36; ++i) { pose_msg.pose.covariance[i] = 0.0; }

    const auto & cov = filter_.covariance();
    pose_msg.pose.covariance[0]  = cov(0, 0);
    pose_msg.pose.covariance[7]  = cov(1, 1);
    pose_msg.pose.covariance[35] = cov(2, 2);

    pose_publisher_->publish(pose_msg);

  }

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr    cmd_vel_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      odom_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr  scan_subscription_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;

  KalmanFilter filter_;

  double robot_x_{0.0};
  double robot_y_{0.0};
  double robot_theta_{0.0};

  double last_v_{0.0};
  double last_omega_{0.0};

  rclcpp::Time last_cmd_time_;
  bool has_cmd_vel_{false};

  rclcpp::Time last_time_;
  bool initialized_{false};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KalmanFilterNode>());
  rclcpp::shutdown();
  return 0;
}