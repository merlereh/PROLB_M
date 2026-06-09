#include <memory>
#include <fstream>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

class EvaluatorNode : public rclcpp::Node
{
public:
  // Constructor
  EvaluatorNode()
  : Node("evaluator_node")
  {
    // Open CSV file
    file_.open("trajectory_log.csv");

    // Write CSV header
    file_ << "time,source,x,y,theta\n";

    // Subscriber for odometry
    auto odom_callback =
      [this](nav_msgs::msg::Odometry::UniquePtr msg) -> void {
        double time = msg->header.stamp.sec +
          msg->header.stamp.nanosec * 1e-9;

        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;
        double theta = getYawFromQuaternion(msg->pose.pose.orientation);

        writeRow(time, "odom", x, y, theta);
      };

    odom_subscription_ =
      this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom",
        10,
        odom_callback
      );

    // Subscriber for Kalman Filter pose
    auto kf_callback =
      [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) -> void {
        double time = msg->header.stamp.sec +
          msg->header.stamp.nanosec * 1e-9;

        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;
        double theta = getYawFromQuaternion(msg->pose.pose.orientation);

        writeRow(time, "kf", x, y, theta);
      };

    kf_subscription_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/kf_pose",
        10,
        kf_callback
      );

    // Subscriber for Extended Kalman Filter pose
    auto ekf_callback =
      [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) -> void {
        double time = msg->header.stamp.sec +
          msg->header.stamp.nanosec * 1e-9;

        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;
        double theta = getYawFromQuaternion(msg->pose.pose.orientation);

        writeRow(time, "ekf", x, y, theta);
      };

    ekf_subscription_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/ekf_pose",
        10,
        ekf_callback
      );

    // Subscriber for Particle Filter pose
    auto pf_callback =
      [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) -> void {
        double time = msg->header.stamp.sec +
          msg->header.stamp.nanosec * 1e-9;

        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;
        double theta = getYawFromQuaternion(msg->pose.pose.orientation);

        writeRow(time, "pf", x, y, theta);
      };

    pf_subscription_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/pf_pose",
        10,
        pf_callback
      );

    // Subscriber for AMCL pose from Nav2
    auto amcl_callback =
      [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) -> void {
        double time = msg->header.stamp.sec +
          msg->header.stamp.nanosec * 1e-9;

        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;
        double theta = getYawFromQuaternion(msg->pose.pose.orientation);

        writeRow(time, "amcl", x, y, theta);
      };

    amcl_subscription_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/amcl_pose",
        10,
        amcl_callback
      );

    RCLCPP_INFO(this->get_logger(), "Evaluator node started.");
    RCLCPP_INFO(this->get_logger(), "Writing data to trajectory_log.csv");
  }

  // Destructor
  ~EvaluatorNode()
  {
    if (file_.is_open()) {
      file_.close();
    }
  }

private:
  // Convert quaternion orientation to yaw angle
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

  // Write one row into the CSV file
  void writeRow(
    double time,
    const std::string & source,
    double x,
    double y,
    double theta)
  {
    if (!file_.is_open()) {
      RCLCPP_WARN(this->get_logger(), "CSV file is not open.");
      return;
    }

    file_ << time << ","
          << source << ","
          << x << ","
          << y << ","
          << theta << "\n";

    file_.flush();
  }

  // CSV file
  std::ofstream file_;

  // Subscribers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr kf_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr ekf_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pf_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EvaluatorNode>());
  rclcpp::shutdown();
  return 0;
}