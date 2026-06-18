// kf_predict_only_node.cpp
//
// Laeuf nur den KF-Prediction-Step durch — keine Odom-Correction,
// keine Landmark-Correction. Published auf /kf_predict_only_pose.
//
// Zeigt die reine Motion-Model-Drift: Sigma waechst jeden Schritt,
// ohne je durch eine Messung korrigiert zu werden.
//
// Unterschied zum ekf_predict_only_node:
//   Kovarianz-Propagation nutzt die fixe lineare A-Matrix (kein Jacobian).
//   A = [I3 | dt*I3]
//       [0  |  0   ]
//   State-Prediction verwendet trotzdem cos/sin fuer vx/vy.

#include <memory>
#include <string>
#include <cmath>

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

#include "kalman_filter.hpp"

using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::LaserScan>;

static double correctAngle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

class KfPredictOnlyNode : public rclcpp::Node
{
public:
  KfPredictOnlyNode()
  : Node("kf_predict_only_node")
  {
    this->declare_parameter("r_x",     0.05);
    this->declare_parameter("r_y",     0.05);
    this->declare_parameter("r_theta", 0.02);
    this->declare_parameter("r_vx",    0.10);
    this->declare_parameter("r_vy",    0.05);
    this->declare_parameter("r_omega", 0.05);

    R_ = Matrix6d::Zero();
    R_(0, 0) = this->get_parameter("r_x").as_double();
    R_(1, 1) = this->get_parameter("r_y").as_double();
    R_(2, 2) = this->get_parameter("r_theta").as_double();
    R_(3, 3) = this->get_parameter("r_vx").as_double();
    R_(4, 4) = this->get_parameter("r_vy").as_double();
    R_(5, 5) = this->get_parameter("r_omega").as_double();

    mu_ = Vector6d::Zero();
    sigma_ = Matrix6d::Zero();
    sigma_(0, 0) = 0.05;
    sigma_(1, 1) = 0.05;
    sigma_(2, 2) = 0.02;
    sigma_(3, 3) = 0.05;
    sigma_(4, 4) = 0.05;
    sigma_(5, 5) = 0.02;

    initialpose_subscription_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 10,
        [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) {
          mu_(0) = msg->pose.pose.position.x;
          mu_(1) = msg->pose.pose.position.y;
          mu_(2) = getYaw(msg->pose.pose.orientation);
          mu_(3) = 0.0;
          mu_(4) = 0.0;
          mu_(5) = 0.0;

          sigma_ = Matrix6d::Zero();
          sigma_(0, 0) = 0.05;
          sigma_(1, 1) = 0.05;
          sigma_(2, 2) = 0.02;
          sigma_(3, 3) = 0.05;
          sigma_(4, 4) = 0.05;
          sigma_(5, 5) = 0.02;

          odom_initialized_ = false;
          initialized_      = true;

          RCLCPP_INFO(this->get_logger(),
            "[KF PredictOnly] Initialisiert: x=%.3f, y=%.3f, theta=%.3f",
            mu_(0), mu_(1), mu_(2));
        });

    cmd_vel_subscription_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10,
        [this](geometry_msgs::msg::Twist::UniquePtr msg) {
          last_v_        = msg->linear.x;
          last_omega_    = msg->angular.z;
          last_cmd_time_ = this->now();
          has_cmd_vel_   = true;
        });

    odom_sub_.subscribe(this, "/odom");
    scan_sub_.subscribe(this, "/scan");

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(20), odom_sub_, scan_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));
    sync_->registerCallback(
        std::bind(&KfPredictOnlyNode::syncCallback, this,
                  std::placeholders::_1, std::placeholders::_2));

    pose_publisher_ =
      this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/kf_predict_only_pose", 10);

    RCLCPP_INFO(this->get_logger(),
      "[KF PredictOnly] Node gestartet -> /kf_predict_only_pose. Warte auf /initialpose ...");
  }

private:
  void syncCallback(
    const nav_msgs::msg::Odometry::ConstSharedPtr & odom_msg,
    const sensor_msgs::msg::LaserScan::ConstSharedPtr & /*scan_msg*/)
  {
    if (!initialized_)  { return; }
    if (!has_cmd_vel_)  { return; }

    double cmd_age = std::abs((this->now() - last_cmd_time_).seconds());
    if (cmd_age > 0.2) { return; }

    rclcpp::Time current_time = odom_msg->header.stamp;

    if (!odom_initialized_) {
      last_time_        = current_time;
      odom_initialized_ = true;
      RCLCPP_INFO(this->get_logger(), "[KF PredictOnly] Odom initialisiert.");
      return;
    }

    const double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;

    if (dt <= 0.0 || dt > 1.0) { return; }

    // ── KF Prediction-Step ────────────────────────────────────────────────────
    // State-Prediction: cos/sin fuer vx/vy (gleich wie kalman_filter.hpp)
    const double theta = mu_(2);
    const double v     = last_v_;
    const double omega = last_omega_;

    Vector6d mu_bar;
    mu_bar(0) = mu_(0) + v * std::cos(theta) * dt;
    mu_bar(1) = mu_(1) + v * std::sin(theta) * dt;
    mu_bar(2) = correctAngle(mu_(2) + omega * dt);
    mu_bar(3) = v * std::cos(theta);
    mu_bar(4) = v * std::sin(theta);
    mu_bar(5) = omega;

    // Kovarianz-Propagation: fixe lineare A-Matrix (kein Jacobian)
    //   A = [I3 | dt*I3]
    //       [0  |  0   ]
    Matrix6d A = Matrix6d::Zero();
    A(0, 0) = 1.0;  A(0, 3) = dt;
    A(1, 1) = 1.0;  A(1, 4) = dt;
    A(2, 2) = 1.0;  A(2, 5) = dt;

    Matrix6d sigma_bar = A * sigma_ * A.transpose() + R_;

    // Kein correctOdom, kein correctLandmark
    mu_    = mu_bar;
    sigma_ = sigma_bar;

    publishPose(mu_, sigma_, current_time, "map");
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
    const Matrix6d & cov,
    const rclcpp::Time & stamp,
    const std::string & frame_id)
  {
    geometry_msgs::msg::PoseWithCovarianceStamped msg;
    msg.header.stamp    = stamp;
    msg.header.frame_id = frame_id;

    msg.pose.pose.position.x = state(0);
    msg.pose.pose.position.y = state(1);
    msg.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, state(2));
    msg.pose.pose.orientation = tf2::toMsg(q);

    for (int i = 0; i < 36; ++i) { msg.pose.covariance[i] = 0.0; }

    msg.pose.covariance[0]  = cov(0, 0);  // sigma_xx
    msg.pose.covariance[7]  = cov(1, 1);  // sigma_yy
    msg.pose.covariance[35] = cov(2, 2);  // sigma_theta
    msg.pose.covariance[1]  = cov(0, 1);  // sigma_xy
    msg.pose.covariance[6]  = cov(1, 0);  // sigma_yx

    pose_publisher_->publish(msg);

    RCLCPP_DEBUG(this->get_logger(),
      "sigma_xx=%.4f  sigma_yy=%.4f  sigma_tt=%.4f",
      cov(0, 0), cov(1, 1), cov(2, 2));
  }

  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr                     cmd_vel_subscription_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr    pose_publisher_;

  message_filters::Subscriber<nav_msgs::msg::Odometry>     odom_sub_;
  message_filters::Subscriber<sensor_msgs::msg::LaserScan> scan_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  Vector6d mu_    = Vector6d::Zero();
  Matrix6d sigma_ = Matrix6d::Identity() * 0.05;
  Matrix6d R_     = Matrix6d::Zero();

  double last_v_{0.0};
  double last_omega_{0.0};
  rclcpp::Time last_cmd_time_;
  bool has_cmd_vel_{false};

  rclcpp::Time last_time_;
  bool initialized_{false};
  bool odom_initialized_{false};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KfPredictOnlyNode>());
  rclcpp::shutdown();
  return 0;
}