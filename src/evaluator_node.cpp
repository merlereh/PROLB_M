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

// CSV columns:
//   time, source, frame, x, y, theta, cov_xx, cov_yy, cov_xy
//
// cov_xx / cov_yy / cov_xy come straight from the PoseWithCovarianceStamped
// message (covariance[0], covariance[7], covariance[1]).
// For /odom and /amcl the same fields are used.
// The PF node already publishes these (filled from particle spread),
// KF and EKF fill them from their Sigma matrix.

class EvaluatorNode : public rclcpp::Node
{
public:
  EvaluatorNode()
  : Node("evaluator_node")
  {
    tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    file_.open("trajectory_log.csv");
    // Header — cov_xx / cov_yy / cov_xy added
    file_ << "time,source,frame,x,y,theta,cov_xx,cov_yy,cov_xy\n";

    // ── /odom ────────────────────────────────────────────────────────────────
    odom_subscription_ =
      this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10,
        [this](nav_msgs::msg::Odometry::UniquePtr msg) {
          // odom has no filter covariance → write 0s
          logPoseInMapFrame(
            msg->header.stamp, "odom",
            msg->header.frame_id, msg->pose.pose,
            0.0, 0.0, 0.0);
        });

    // ── /kf_pose ─────────────────────────────────────────────────────────────
    kf_subscription_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/kf_pose", 10,
        [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) {
          logPoseInMapFrame(
            msg->header.stamp, "kf",
            msg->header.frame_id, msg->pose.pose,
            msg->pose.covariance[0],   // cov_xx
            msg->pose.covariance[7],   // cov_yy
            msg->pose.covariance[1]);  // cov_xy
        });

    // ── /ekf_pose ────────────────────────────────────────────────────────────
    ekf_subscription_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/ekf_pose", 10,
        [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) {
          logPoseInMapFrame(
            msg->header.stamp, "ekf",
            msg->header.frame_id, msg->pose.pose,
            msg->pose.covariance[0],
            msg->pose.covariance[7],
            msg->pose.covariance[1]);
        });

    // ── /pf_pose ─────────────────────────────────────────────────────────────
    pf_subscription_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/pf_pose", 10,
        [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) {
          logPoseInMapFrame(
            msg->header.stamp, "pf",
            msg->header.frame_id, msg->pose.pose,
            msg->pose.covariance[0],
            msg->pose.covariance[7],
            msg->pose.covariance[1]);
        });

    // ── /amcl_pose ───────────────────────────────────────────────────────────
    amcl_subscription_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/amcl_pose", 10,
        [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) {
          logPoseInMapFrame(
            msg->header.stamp, "amcl",
            msg->header.frame_id, msg->pose.pose,
            msg->pose.covariance[0],
            msg->pose.covariance[7],
            msg->pose.covariance[1]);
        });

    // ── /ekf_pose_predicted (Sigma_bar_ — pre-correction) ────────────────────
    ekf_predict_only_sub_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/ekf_predict_only_pose", 10,
        [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) {
          logPoseInMapFrame(
            msg->header.stamp, "ekf_predict_only",
            msg->header.frame_id, msg->pose.pose,
            msg->pose.covariance[0],
            msg->pose.covariance[7],
            msg->pose.covariance[1]);
        });

    RCLCPP_INFO(this->get_logger(), "Evaluator node started.");
    RCLCPP_INFO(this->get_logger(),
      "Writing trajectory_log.csv (x, y, theta, cov_xx, cov_yy, cov_xy) in map frame.");
  }

  ~EvaluatorNode()
  {
    if (file_.is_open()) { file_.close(); }
  }

private:
  // ---------------------------------------------------------------------------
  // Transform pose to map frame, then write one CSV row
  // ---------------------------------------------------------------------------
  void logPoseInMapFrame(
    const rclcpp::Time & stamp,
    const std::string  & source,
    const std::string  & frame_id,
    const geometry_msgs::msg::Pose & pose,
    double cov_xx, double cov_yy, double cov_xy)
  {
    geometry_msgs::msg::PoseStamped input_pose;
    input_pose.header.stamp    = stamp;
    input_pose.header.frame_id = frame_id;
    input_pose.pose            = pose;

    geometry_msgs::msg::PoseStamped map_pose;
    try {
      map_pose = tf_buffer_->transform(input_pose, "map", tf2::durationFromSec(0.1));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(this->get_logger(),
        "Could not transform '%s'→'map': %s", frame_id.c_str(), ex.what());
      return;
    }

    const double time  = map_pose.header.stamp.sec
                       + map_pose.header.stamp.nanosec * 1e-9;
    const double x     = map_pose.pose.position.x;
    const double y     = map_pose.pose.position.y;
    const double theta = tf2::getYaw(map_pose.pose.orientation);

    if (!file_.is_open()) { return; }

    file_ << time   << ","
          << source << ","
          << "map"  << ","
          << x      << ","
          << y      << ","
          << theta  << ","
          << cov_xx << ","
          << cov_yy << ","
          << cov_xy << "\n";

    file_.flush();
  }

  // ---------------------------------------------------------------------------
  // Members
  // ---------------------------------------------------------------------------
  std::ofstream file_;

  std::unique_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr                       odom_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr kf_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr ekf_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pf_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr ekf_predict_only_sub_;};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EvaluatorNode>());
  rclcpp::shutdown();
  return 0;
}