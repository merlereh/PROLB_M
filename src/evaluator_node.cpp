#include <memory>
#include <fstream>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/utils.h"

class EvaluatorNode : public rclcpp::Node
{
public:
  // Constructor
  EvaluatorNode()
  : Node("evaluator_node")
  {
    // Create TF buffer and listener.
    // This allows the evaluator to transform poses from odom frame to map frame.
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Open CSV file
    file_.open("trajectory_log.csv");

    // Write CSV header
    // All poses written into this file should be in the map frame.
    file_ << "time,source,frame,x,y,theta\n";

    // Subscriber for odometry
    auto odom_callback =
      [this](nav_msgs::msg::Odometry::UniquePtr msg) -> void {
        logPoseInMapFrame(
          msg->header.stamp,
          "odom",
          msg->header.frame_id,
          msg->pose.pose
        );
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
        logPoseInMapFrame(
          msg->header.stamp,
          "kf",
          msg->header.frame_id,
          msg->pose.pose
        );
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
        logPoseInMapFrame(
          msg->header.stamp,
          "ekf",
          msg->header.frame_id,
          msg->pose.pose
        );
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
        logPoseInMapFrame(
          msg->header.stamp,
          "pf",
          msg->header.frame_id,
          msg->pose.pose
        );
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
        logPoseInMapFrame(
          msg->header.stamp,
          "amcl",
          msg->header.frame_id,
          msg->pose.pose
        );
      };

    amcl_subscription_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/amcl_pose",
        10,
        amcl_callback
      );

    RCLCPP_INFO(this->get_logger(), "Evaluator node started.");
    RCLCPP_INFO(this->get_logger(), "Writing data to trajectory_log.csv in map frame.");
  }

  // Destructor
  ~EvaluatorNode()
  {
    if (file_.is_open()) {
      file_.close();
    }
  }

private:
  // Transform a PoseStamped message into the map frame
  bool transformPoseToMap(
    const geometry_msgs::msg::PoseStamped & input_pose,
    geometry_msgs::msg::PoseStamped & output_pose)
  {
    try {
      output_pose = tf_buffer_->transform(
        input_pose,
        "map",
        tf2::durationFromSec(0.1)
      );

      return true;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(
        this->get_logger(),
        "Could not transform pose from '%s' to 'map': %s",
        input_pose.header.frame_id.c_str(),
        ex.what()
      );

      return false;
    }
  }

  // Convert incoming pose to map frame and write it to CSV
  void logPoseInMapFrame(
    const rclcpp::Time & stamp,
    const std::string & source,
    const std::string & frame_id,
    const geometry_msgs::msg::Pose & pose)
  {
    // Create PoseStamped from incoming pose
    geometry_msgs::msg::PoseStamped input_pose;
    input_pose.header.stamp = stamp;
    input_pose.header.frame_id = frame_id;
    input_pose.pose = pose;

    // Transform pose into map frame
    geometry_msgs::msg::PoseStamped map_pose;

    if (!transformPoseToMap(input_pose, map_pose)) {
      return;
    }

    // Time
    double time = map_pose.header.stamp.sec +
      map_pose.header.stamp.nanosec * 1e-9;

    // Position in map frame
    double x = map_pose.pose.position.x;
    double y = map_pose.pose.position.y;

    // Orientation in map frame
    double theta = tf2::getYaw(map_pose.pose.orientation);

    // Write transformed pose
    writeRow(time, source, "map", x, y, theta);
  }

  // Write one row into the CSV file
  void writeRow(
    double time,
    const std::string & source,
    const std::string & frame,
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
          << frame << ","
          << x << ","
          << y << ","
          << theta << "\n";

    file_.flush();
  }

  // CSV file
  std::ofstream file_;

  // TF buffer and listener
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

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