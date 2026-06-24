#include <memory>
#include <string>
#include <cmath>
#include <random>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"

#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "landmark_scan_helper.hpp"
#include "extended_kalman_filter.hpp"

// Landmark position — must match the SDF world file and landmark_visualizer_node.cpp.
static constexpr double LANDMARK_X   = 1.8;
static constexpr double LANDMARK_Y   = 0.0;

// Association gate threshold: maximum distance between the projected detection
// and the known landmark position in the world frame.
// Too large → false detections get accepted.
// Too small → real detections near a bad pose estimate get rejected.
static constexpr double ASSOC_GATE_M = 0.60;

// ---------------------------------------------------------------------------
// ExtendedKalmanFilterNode
//
// Trigger:  /odom + /scan     → time-synchronized (ApproximateTime, 100 ms)
// Input:    /cmd_vel          → control u = [v, omega], cached, max 0.2 s old
//           /odom             → z = [x_map, y_map, theta_map]
//           /scan             → laser scan for landmark detection
//           /initialpose      → sets the filter's starting state
// Output:   /ekf_pose         → estimated pose as PoseWithCovarianceStamped
// ---------------------------------------------------------------------------

using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::LaserScan>;

class ExtendedKalmanFilterNode : public rclcpp::Node
{
public:
  ExtendedKalmanFilterNode() : Node("ekf_node")
  {
    // ROS parameters for noise matrices R and Q (loaded from filter_params.yaml)
    this->declare_parameter("r_x",      0.05);
    this->declare_parameter("r_y",      0.05);
    this->declare_parameter("r_theta",  0.02);
    this->declare_parameter("r_vx",     0.10);
    this->declare_parameter("r_vy",     0.05);
    this->declare_parameter("r_omega",  0.05);
    this->declare_parameter("q_x",      0.10);
    this->declare_parameter("q_y",      0.10);
    this->declare_parameter("q_theta",  0.05);
    this->declare_parameter("q_lm_r",   0.05);
    this->declare_parameter("q_lm_phi", 0.01);

    // Optional artificial odometry noise (std dev) — set to 0 to disable
    this->declare_parameter("odom_noise_x",     0.05);
    this->declare_parameter("odom_noise_y",     0.05);
    this->declare_parameter("odom_noise_theta", 0.02);
    odom_noise_x_     = this->get_parameter("odom_noise_x").as_double();
    odom_noise_y_     = this->get_parameter("odom_noise_y").as_double();
    odom_noise_theta_ = this->get_parameter("odom_noise_theta").as_double();

    // Initialize the filter with the loaded noise parameters
    filter_.setNoiseParams(
      this->get_parameter("r_x").as_double(),
      this->get_parameter("r_y").as_double(),
      this->get_parameter("r_theta").as_double(),
      this->get_parameter("r_vx").as_double(),
      this->get_parameter("r_vy").as_double(),
      this->get_parameter("r_omega").as_double(),
      this->get_parameter("q_x").as_double(),
      this->get_parameter("q_y").as_double(),
      this->get_parameter("q_theta").as_double(),
      this->get_parameter("q_lm_r").as_double(),
      this->get_parameter("q_lm_phi").as_double());

    // /initialpose sets the filter's starting state (sent by initial_pose_node)
    initialpose_sub_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 10,
        [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) {
          Eigen::Vector3d pose;
          pose << msg->pose.pose.position.x,
                  msg->pose.pose.position.y,
                  getYaw(msg->pose.pose.orientation);
          filter_.setState(pose);
          odom_initialized_ = false;
          initialized_      = true;
          RCLCPP_INFO(this->get_logger(),
            "EKF init: x=%.3f y=%.3f th=%.3f", pose(0), pose(1), pose(2));
        });

    // Cache the latest cmd_vel as the control input u = [v, omega]
    cmd_vel_sub_ =
      this->create_subscription<geometry_msgs::msg::Twist>("/cmd_vel", 10,
        [this](geometry_msgs::msg::Twist::UniquePtr msg) {
          last_v_        = msg->linear.x;
          last_omega_    = msg->angular.z;
          last_cmd_time_ = this->now();
          has_cmd_vel_   = true;
        });

    // Time-synchronize /odom and /scan so the callback always gets a matching pair
    odom_sub_.subscribe(this, "/odom");
    scan_sub_.subscribe(this, "/scan");

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(20), odom_sub_, scan_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));
    sync_->registerCallback(
        std::bind(&ExtendedKalmanFilterNode::syncCallback, this,
                  std::placeholders::_1, std::placeholders::_2));

    pose_pub_ =
      this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/ekf_pose", 10);

    RCLCPP_INFO(this->get_logger(),
      "EKF node started. Waiting for /initialpose. Landmark at (%.2f, %.2f).",
      LANDMARK_X, LANDMARK_Y);
  }

private:
  void syncCallback(
    const nav_msgs::msg::Odometry::ConstSharedPtr      & odom_msg,
    const sensor_msgs::msg::LaserScan::ConstSharedPtr  & scan_msg)
  {
    if (!initialized_)  return;
    if (!has_cmd_vel_)  return;
    if (std::abs((this->now() - last_cmd_time_).seconds()) > 0.2) return;

    rclcpp::Time current_time = odom_msg->header.stamp;

    const double x_odom     = odom_msg->pose.pose.position.x     + noiseX();
    const double y_odom     = odom_msg->pose.pose.position.y     + noiseY();
    const double theta_odom = correctAngle(
        getYaw(odom_msg->pose.pose.orientation) + noiseTheta());

    // On the first message after initialization, compute the odom→map offset.
    if (!odom_initialized_) {
      const Vector6d & s = filter_.state();
      offset_theta_ = correctAngle(s(2) - theta_odom);
      const double co = std::cos(offset_theta_), so = std::sin(offset_theta_);
      offset_x_ = s(0) - (co * x_odom - so * y_odom);
      offset_y_ = s(1) - (so * x_odom + co * y_odom);
      last_time_        = current_time;
      odom_initialized_ = true;
      RCLCPP_INFO(this->get_logger(),
        "Odom offset computed: dx=%.3f dy=%.3f dth=%.3f",
        offset_x_, offset_y_, offset_theta_);
      return;
    }

    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;
    if (dt <= 0.0 || dt > 1.0) return;

    // z = [x_map, y_map, theta_map] from odometry (after offset transform)
    const double co        = std::cos(offset_theta_), so = std::sin(offset_theta_);
    const double x_map     = co * x_odom - so * y_odom + offset_x_;
    const double y_map     = so * x_odom + co * y_odom + offset_y_;
    const double theta_map = correctAngle(theta_odom + offset_theta_);

    Eigen::Vector3d z_full;
    z_full(0) = x_map;
    z_full(1) = y_map;
    z_full(2) = theta_map;

    // --- 1) EKF Predict ---
    Eigen::Vector2d u;
    u << last_v_, last_omega_;

    const auto t_start = std::chrono::steady_clock::now();

    filter_.predict(u, dt);

    // --- 2) EKF Correct: x, y, theta from odometry ---
    Vector6d estimate = filter_.correctFull(z_full);

    // --- 3) EKF Correct: landmark pillar ---
    const Vector6d & state = filter_.state();
    double r_meas, phi_meas;

    if (detectLandmark(*scan_msg,
                       state(0), state(1), state(2),
                       LANDMARK_X, LANDMARK_Y,
                       r_meas, phi_meas))
    {
      // Association gate: project the detection into world coordinates and
      // compare with the known landmark position.
      // This prevents corrections from false detections.
      const double wx = state(0) + r_meas * std::cos(state(2) + phi_meas);
      const double wy = state(1) + r_meas * std::sin(state(2) + phi_meas);
      const double assoc_err =
        std::sqrt((wx - LANDMARK_X) * (wx - LANDMARK_X) +
                  (wy - LANDMARK_Y) * (wy - LANDMARK_Y));

      if (assoc_err < ASSOC_GATE_M) {
        Eigen::Vector2d z_lm; z_lm << r_meas, phi_meas;
        Eigen::Vector2d lm;   lm   << LANDMARK_X, LANDMARK_Y;
        estimate = filter_.correctLandmark(z_lm, lm);
        RCLCPP_INFO(this->get_logger(),
          "Landmark accepted: r=%.3f phi=%.3f assoc_err=%.3f → x=%.3f y=%.3f",
          r_meas, phi_meas, assoc_err, estimate(0), estimate(1));
      } else {
        RCLCPP_WARN(this->get_logger(),
          "Landmark rejected (assoc_err=%.2f > %.2f)", assoc_err, ASSOC_GATE_M);
      }
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double runtime_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // --- 4) Publish pose estimate ---
    publishPose(estimate, odom_msg->header.stamp, "map", runtime_ms);
  }

  // ── Helper functions ──────────────────────────────────────────────────────

  static double correctAngle(double a)
  { return std::atan2(std::sin(a), std::cos(a)); }

  double noiseX()
  { return std::normal_distribution<double>(0.0, odom_noise_x_)(rng_); }
  double noiseY()
  { return std::normal_distribution<double>(0.0, odom_noise_y_)(rng_); }
  double noiseTheta()
  { return std::normal_distribution<double>(0.0, odom_noise_theta_)(rng_); }

  double getYaw(const geometry_msgs::msg::Quaternion & q_msg)
  {
    tf2::Quaternion q(q_msg.x, q_msg.y, q_msg.z, q_msg.w);
    double r, p, y;
    tf2::Matrix3x3(q).getRPY(r, p, y);
    return y;
  }

  void publishPose(const Vector6d & s, const rclcpp::Time & stamp,
                   const std::string & fid, double runtime_ms)
  {
    geometry_msgs::msg::PoseWithCovarianceStamped msg;
    msg.header.stamp    = stamp;
    msg.header.frame_id = fid;
    msg.pose.pose.position.x = s(0);
    msg.pose.pose.position.y = s(1);
    tf2::Quaternion q;
    q.setRPY(0, 0, s(2));
    msg.pose.pose.orientation = tf2::toMsg(q);
    for (int i = 0; i < 36; ++i) msg.pose.covariance[i] = 0.0;
    const auto & cov = filter_.covariance();
    msg.pose.covariance[0]  = cov(0, 0);
    msg.pose.covariance[7]  = cov(1, 1);
    msg.pose.covariance[35] = cov(2, 2);
    msg.pose.covariance[1]  = cov(0, 1);
    msg.pose.covariance[6]  = cov(1, 0);
    msg.pose.covariance[14] = runtime_ms;   // update runtime in ms (for evaluator)
    pose_pub_->publish(msg);
  }

  // ── Members ──────────────────────────────────────────────────────────────

  message_filters::Subscriber<nav_msgs::msg::Odometry>      odom_sub_;
  message_filters::Subscriber<sensor_msgs::msg::LaserScan>  scan_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    initialpose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    pose_pub_;

  ExtendedKalmanFilter filter_;

  double offset_x_{0.0}, offset_y_{0.0}, offset_theta_{0.0};
  double last_v_{0.0}, last_omega_{0.0};
  rclcpp::Time last_cmd_time_, last_time_;
  bool has_cmd_vel_{false}, initialized_{false}, odom_initialized_{false};

  double odom_noise_x_{0.0}, odom_noise_y_{0.0}, odom_noise_theta_{0.0};
  std::mt19937 rng_{std::random_device{}()};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ExtendedKalmanFilterNode>());
  rclcpp::shutdown();
  return 0;
}