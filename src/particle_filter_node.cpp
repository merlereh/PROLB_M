#include <memory>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"

#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "particle_filter.hpp"

class ParticleFilterNode : public rclcpp::Node
{
public:
  ParticleFilterNode()
  : Node("particle_filter_node")
  {
    this->declare_parameter<std::string>("resampling_method", "multinomial");
    std::string resampling_method =
      this->get_parameter("resampling_method").as_string();
    filter_.setResamplingMethod(resampling_method);

    RCLCPP_INFO(this->get_logger(),
      "Using resampling method: %s", resampling_method.c_str());

    // Subscriber for velocity commands (/cmd_vel)
    // u = [v, omega]
    auto cmd_vel_callback =
      [this](geometry_msgs::msg::Twist::UniquePtr msg) -> void {
        last_v_        = msg->linear.x;
        last_omega_    = msg->angular.z;
        last_cmd_time_ = this->now();
        has_cmd_vel_   = true;

        RCLCPP_DEBUG(this->get_logger(),
          "cmd_vel: v=%.3f, omega=%.3f", last_v_, last_omega_);
      };

    cmd_vel_subscription_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10, cmd_vel_callback);

    // Subscriber for odometry (/odom)
    // Measurement z = [x, y, theta]  (absolute pose)
    auto odom_callback =
      [this](nav_msgs::msg::Odometry::UniquePtr msg) -> void {
        rclcpp::Time current_time = msg->header.stamp;

        double x     = msg->pose.pose.position.x;
        double y     = msg->pose.pose.position.y;
        double theta = getYawFromQuaternion(msg->pose.pose.orientation);

        Eigen::Vector3d measurement;
        measurement << x, y, theta;

        if (!initialized_) {
          filter_.initializeParticlesAroundState(measurement);
          last_time_   = current_time;
          initialized_ = true;

          RCLCPP_INFO(this->get_logger(),
            "PF initialized: x=%.3f, y=%.3f, theta=%.3f", x, y, theta);
          return;
        }

        double dt = (current_time - last_time_).seconds();
        last_time_ = current_time;

        if (dt <= 0.0 || dt > 1.0) {
          RCLCPP_WARN(this->get_logger(), "Skipping: dt=%.3f", dt);
          return;
        }

        if (!has_cmd_vel_) {
          RCLCPP_WARN(this->get_logger(), "Skipping: no cmd_vel yet.");
          return;
        }

        double cmd_odom_dt = std::abs((current_time - last_cmd_time_).seconds());
        if (cmd_odom_dt > max_time_difference_) {
          RCLCPP_WARN(this->get_logger(),
            "Skipping: cmd_vel/odom time diff=%.3f s", cmd_odom_dt);
          return;
        }

        Eigen::Vector2d control;
        control << last_v_, last_omega_;

        Vector6d estimate = filter_.update(control, measurement, dt);

        RCLCPP_DEBUG(this->get_logger(),
          "PF estimate: x=%.3f, y=%.3f, theta=%.3f, vx=%.3f, vy=%.3f, dtheta=%.3f",
          estimate(0), estimate(1), estimate(2),
          estimate(3), estimate(4), estimate(5));

        publishPose(estimate, msg->header.stamp, msg->header.frame_id);
        publishParticles(msg->header.stamp, msg->header.frame_id);
      };

    odom_subscription_ =
      this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10, odom_callback);

    pose_publisher_ =
      this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/pf_pose", 10);

    particle_publisher_ =
      this->create_publisher<geometry_msgs::msg::PoseArray>(
        "/my_particle_cloud", 10);

    RCLCPP_INFO(this->get_logger(), "Particle filter node started (6D state).");
    RCLCPP_INFO(this->get_logger(), "State: [x, y, theta, vx, vy, theta_dot]");
    RCLCPP_INFO(this->get_logger(), "Subscribing to /cmd_vel and /odom");
    RCLCPP_INFO(this->get_logger(), "Publishing to /pf_pose and /my_particle_cloud");
  }

private:
  double getYawFromQuaternion(const geometry_msgs::msg::Quaternion & q_msg)
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

    for (int i = 0; i < 36; ++i) {
      pose_msg.pose.covariance[i] = 0.0;
    }

    pose_publisher_->publish(pose_msg);
  }

  // Publish only the pose part [x, y, theta] of each particle for RViz
  void publishParticles(
    const rclcpp::Time & stamp,
    const std::string & frame_id)
  {
    geometry_msgs::msg::PoseArray particle_msg;
    particle_msg.header.stamp    = stamp;
    particle_msg.header.frame_id = frame_id.empty() ? "odom" : frame_id;

    for (const auto & p : filter_.particles()) {
      geometry_msgs::msg::Pose pose;
      pose.position.x = p(0);
      pose.position.y = p(1);
      pose.position.z = 0.0;

      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, p(2));
      pose.orientation = tf2::toMsg(q);

      particle_msg.poses.push_back(pose);
    }

    particle_publisher_->publish(particle_msg);
  }

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr   odom_subscription_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_publisher_;

  ParticleFilter filter_;

  double last_v_{0.0};
  double last_omega_{0.0};

  rclcpp::Time last_cmd_time_;
  bool has_cmd_vel_{false};

  double max_time_difference_{0.10};

  rclcpp::Time last_time_;
  bool initialized_{false};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ParticleFilterNode>());
  rclcpp::shutdown();
  return 0;
}