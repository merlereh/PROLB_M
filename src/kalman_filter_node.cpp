#include <memory>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"

#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"


#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "kalman_filter.hpp"

class KalmanFilterNode : public rclcpp::Node
{
public:
  // Constructor
  KalmanFilterNode()
  : Node("kf_node")
  {
    // Subscriber for velocity commands.
    // This is the normal /cmd_vel topic.
    auto cmd_vel_callback =
      [this](geometry_msgs::msg::Twist::UniquePtr msg) -> void {
        // Save latest control input
        // u = [v, omega]
        last_v_ = msg->linear.x;
        last_omega_ = msg->angular.z;

        // store the time
        last_cmd_time_ = this->now();
        has_cmd_vel_ = true;

        RCLCPP_INFO(
          this->get_logger(),
          "cmd_vel: linear.x = %.3f, angular.z = %.3f",
          last_v_,
          last_omega_
        );
      };

    cmd_vel_subscription_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel",
        10,
        cmd_vel_callback
      );

    // Subscriber for odometry.
    // Odometry contains pose and twist.
    // We use the pose as measurement z = [x, y, theta].
    auto odom_callback =
      [this](nav_msgs::msg::Odometry::UniquePtr msg) -> void {
        // Current timestamp from odometry message
        rclcpp::Time current_time = msg->header.stamp;

        // Read measured x and y directly from odometry pose
        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;

        // Read orientation from odometry pose.
        // ROS stores orientation as quaternion, but our filter uses theta/yaw.
        double theta = getYawFromQuaternion(msg->pose.pose.orientation);

        // Measurement z = [x, y, theta]
        Eigen::Vector3d measurement;
        measurement << x, y, theta;

        // Initialize filter with first odometry measurement
        if (!initialized_) {
          filter_.setState(measurement);
          last_time_ = current_time;
          initialized_ = true;

          RCLCPP_INFO(
            this->get_logger(),
            "Filter initialized: x=%.3f, y=%.3f, theta=%.3f",
            x,
            y,
            theta
          );

          return;
        }

        // Calculate time difference dt
        double dt = (current_time - last_time_).seconds();
        last_time_ = current_time;

        // Skip invalid time steps
        if (dt <= 0.0 || dt > 1.0) {
          RCLCPP_WARN(this->get_logger(), "Skipping update because dt=%.3f", dt);
          return;
        }

        // Check if a velocity command was received before
        if (!has_cmd_vel_) {
          RCLCPP_WARN(
            this->get_logger(),
            "Skipping update because no cmd_vel was received yet."
          );
          return;
        }

        // Check if latest /cmd_vel and current /odom are close enough in time
        double cmd_odom_dt = std::abs((current_time - last_cmd_time_).seconds());

        if (cmd_odom_dt > max_time_difference_) {
          RCLCPP_WARN(
            this->get_logger(),
            "Skipping update because cmd_vel and odom are not synchronized. Difference: %.3f s",
            cmd_odom_dt
          );
          return;
        }

        // Control input u = [v, omega]
        // Values come from the latest /cmd_vel message.
        Eigen::Vector2d control;
        control << last_v_, last_omega_;

        // predict + correct
        Eigen::Vector3d estimate = filter_.update(control, measurement, dt);
        RCLCPP_INFO(
          this->get_logger(),
          "KF estimate: x=%.3f, y=%.3f, theta=%.3f",
          estimate(0),
          estimate(1),
          estimate(2)
        );

        // Publish estimated pose
        publishPose(estimate, msg->header.stamp, msg->header.frame_id);
      };

    odom_subscription_ =
      this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom",
        10,
        odom_callback
      );

    // Publisher for the estimated pose from the Kalman Filter
    pose_publisher_ =
      this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/kf_pose",
        10
      );

    RCLCPP_INFO(this->get_logger(), "Kalman filter node started.");
    RCLCPP_INFO(this->get_logger(), "Subscribing to /cmd_vel");
    RCLCPP_INFO(this->get_logger(), "Subscribing to /odom");
    RCLCPP_INFO(this->get_logger(), "Publishing to /kf_pose");
  }

private:
  // Convert quaternion orientation to yaw angle.
  // Odometry stores orientation as quaternion [x, y, z, w].
  // The filter needs theta, which is yaw in 2D navigation.
  double getYawFromQuaternion(const geometry_msgs::msg::Quaternion & q_msg)
  {
    tf2::Quaternion q(
      q_msg.x,
      q_msg.y,
      q_msg.z,
      q_msg.w
    );

    double roll;
    double pitch;
    double yaw;

    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    return yaw;
  }

  // Publish estimated pose as PoseWithCovarianceStamped
  void publishPose(
    const Eigen::Vector3d & state,
    const rclcpp::Time & stamp,
    const std::string & frame_id)
  {
    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;

    // Header
    pose_msg.header.stamp = stamp;
    if (frame_id.empty()) {
      pose_msg.header.frame_id = "odom";
    } else {
      pose_msg.header.frame_id = frame_id;
    }

    // Position
    pose_msg.pose.pose.position.x = state(0);
    pose_msg.pose.pose.position.y = state(1);
    pose_msg.pose.pose.position.z = 0.0;

    // Orientation from theta/yaw.
    // ROS messages need quaternion again, so we convert yaw back to quaternion.
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, state(2));
    pose_msg.pose.pose.orientation = tf2::toMsg(q);

    // Initialize 6x6 covariance matrix with zeros
    for (int i = 0; i < 36; ++i) {
      pose_msg.pose.covariance[i] = 0.0;
    }

    // Fill only x, y and yaw covariance
    const Eigen::Matrix3d & covariance = filter_.covariance();

    pose_msg.pose.covariance[0] = covariance(0, 0);    // x
    pose_msg.pose.covariance[7] = covariance(1, 1);    // y
    pose_msg.pose.covariance[35] = covariance(2, 2);   // yaw

    pose_publisher_->publish(pose_msg);
  }

  // Subscriber for normal velocity command /cmd_vel
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscription_;

  // Subscriber for odometry /odom
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;

  // Publisher for filtered pose /kf_pose
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;

  // Kalman Filter object
  KalmanFilter filter_;

  // Latest velocity command from /cmd_vel
  double last_v_{0.0};
  double last_omega_{0.0};

  // Time of latest velocity command
  rclcpp::Time last_cmd_time_;
  bool has_cmd_vel_{false};

  // Maximum allowed time difference between cmd_vel and odom
  double max_time_difference_{0.10};

  // Time handling
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