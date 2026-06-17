#include <memory>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"

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

static constexpr double LANDMARK_X = 1.8;
static constexpr double LANDMARK_Y = 0.0;

using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::LaserScan>;

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
          initialized_      = true;

          RCLCPP_INFO(this->get_logger(),
            "PF initialisiert (Map-Frame): x=%.3f, y=%.3f, theta=%.3f",
            x, y, theta);
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
        std::bind(&ParticleFilterNode::syncCallback, this,
                  std::placeholders::_1, std::placeholders::_2));

    pose_publisher_ =
      this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/pf_pose", 10);

    particle_publisher_ =
      this->create_publisher<geometry_msgs::msg::PoseArray>(
        "/my_particle_cloud", 10);

    RCLCPP_INFO(this->get_logger(),
      "PF-Node gestartet. Warte auf /initialpose. Landmark bei (%.2f, %.2f)",
      LANDMARK_X, LANDMARK_Y);
  }

private:
  void syncCallback(
    const nav_msgs::msg::Odometry::ConstSharedPtr & odom_msg,
    const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan_msg)
  {
    if (!initialized_) { return; }

    if (!has_cmd_vel_) { return; }
    double cmd_age = std::abs((this->now() - last_cmd_time_).seconds());
    if (cmd_age > 0.2) { return; }

    rclcpp::Time current_time = odom_msg->header.stamp;

    const double x_odom     = odom_msg->pose.pose.position.x;
    const double y_odom     = odom_msg->pose.pose.position.y;
    const double theta_odom = getYaw(odom_msg->pose.pose.orientation);

    if (!odom_initialized_) {
      const Vector6d & s     = filter_.state();
      const double x_map     = s(0);
      const double y_map     = s(1);
      const double theta_map = s(2);

      offset_theta_ = correctAngle(theta_map - theta_odom);
      const double cos_o = std::cos(offset_theta_);
      const double sin_o = std::sin(offset_theta_);
      offset_x_ = x_map - (cos_o * x_odom - sin_o * y_odom);
      offset_y_ = y_map - (sin_o * x_odom + cos_o * y_odom);

      last_time_        = current_time;
      odom_initialized_ = true;

      RCLCPP_INFO(this->get_logger(),
        "Odom-Offset: dx=%.3f, dy=%.3f, dtheta=%.3f",
        offset_x_, offset_y_, offset_theta_);
      return;
    }

    const double cos_o = std::cos(offset_theta_);
    const double sin_o = std::sin(offset_theta_);

    Eigen::Vector3d z_odom_map;
    z_odom_map(0) = cos_o * x_odom - sin_o * y_odom + offset_x_;
    z_odom_map(1) = sin_o * x_odom + cos_o * y_odom + offset_y_;
    z_odom_map(2) = correctAngle(theta_odom + offset_theta_);

    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;

    if (dt <= 0.0 || dt > 1.0) { return; }

    Eigen::Vector2d u;
    u << last_v_, last_omega_;

    // Particle Filter: predict + odom weighting + resample
    Vector6d estimate = filter_.update(u, z_odom_map, dt);

    // Landmark update falls erkannt
    const Vector6d & state = filter_.state();
    double r_meas, phi_meas;
    const bool detected = detectLandmark(
      *scan_msg,
      state(0), state(1), state(2),
      LANDMARK_X, LANDMARK_Y,
      r_meas, phi_meas);

    if (detected) {
      Eigen::Vector2d z_lm;
      z_lm(0) = r_meas;
      z_lm(1) = phi_meas;

      Eigen::Vector2d landmark;
      landmark(0) = LANDMARK_X;
      landmark(1) = LANDMARK_Y;

      estimate = filter_.updateLandmark(z_lm, landmark);

      RCLCPP_INFO(this->get_logger(),
        "Landmark-Update: r=%.3f, phi=%.3f → x=%.3f, y=%.3f",
        r_meas, phi_meas, estimate(0), estimate(1));
    }

    publishPose(estimate, odom_msg->header.stamp, "map");
  }

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

    // Particle cloud für RViz
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
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr    pose_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr                    particle_publisher_;

  message_filters::Subscriber<nav_msgs::msg::Odometry>     odom_sub_;
  message_filters::Subscriber<sensor_msgs::msg::LaserScan> scan_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  ParticleFilter filter_;

  double offset_x_{0.0};
  double offset_y_{0.0};
  double offset_theta_{0.0};

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
  rclcpp::spin(std::make_shared<ParticleFilterNode>());
  rclcpp::shutdown();
  return 0;
}