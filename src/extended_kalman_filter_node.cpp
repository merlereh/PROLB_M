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
#include "extended_kalman_filter.hpp"

// Landmark position in map frame
static constexpr double LANDMARK_X = 1.8;
static constexpr double LANDMARK_Y = 0.0;

class ExtendedKalmanFilterNode : public rclcpp::Node
{
public:
  ExtendedKalmanFilterNode()
  : Node("ekf_node")
  {
    // -------------------------------------------------------------------------
    // /initialpose — setzt den Filter auf die echte Startposition in Weltkoord.
    // -------------------------------------------------------------------------
    initialpose_subscription_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 10,
        [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) {
          const double x     = msg->pose.pose.position.x;
          const double y     = msg->pose.pose.position.y;
          const double theta = getYaw(msg->pose.pose.orientation);

          Eigen::Vector3d pose;
          pose << x, y, theta;
          filter_.setState(pose);

          odom_initialized_ = false;
          initialized_      = true;

          RCLCPP_INFO(this->get_logger(),
            "EKF initialisiert (Map-Frame): x=%.3f, y=%.3f, theta=%.3f",
            x, y, theta);
        });

    // -------------------------------------------------------------------------
    // /cmd_vel — Steuerbefehl für den Prediction-Step
    // -------------------------------------------------------------------------
    cmd_vel_subscription_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10,
        [this](geometry_msgs::msg::Twist::UniquePtr msg) {
          last_v_        = msg->linear.x;
          last_omega_    = msg->angular.z;
          last_cmd_time_ = this->now();
          has_cmd_vel_   = true;
        });

    // -------------------------------------------------------------------------
    // /scan — speichert die letzte Landmark-Messung zwischen.
    // Die eigentliche Korrektur passiert im Odom-Callback!
    // -------------------------------------------------------------------------
    scan_subscription_ =
      this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 10,
        [this](sensor_msgs::msg::LaserScan::UniquePtr msg) {
          if (!initialized_ || !odom_initialized_) { return; }

          // Aktuellen EKF-State für Landmark-Suche verwenden
          const Vector6d & state = filter_.state();

          double r_meas, phi_meas;
          const bool detected = detectLandmark(
            *msg,
            state(0), state(1), state(2),
            LANDMARK_X, LANDMARK_Y,
            r_meas, phi_meas);

          if (detected) {
            // Messung zwischenspeichern — Korrektur passiert im Odom-Callback
            last_lm_r_    = r_meas;
            last_lm_phi_  = phi_meas;
            has_landmark_ = true;
            RCLCPP_INFO(this->get_logger(),
              "Landmark gespeichert: r=%.3f, phi=%.3f", r_meas, phi_meas);
          }
        });

    // -------------------------------------------------------------------------
    // /odom — Haupt-Loop: predict → correctOdom → (optional) correctLandmark
    //
    // Ablauf pro Zyklus:
    //   1. predict(cmd_vel, dt)          — Bewegungsmodell anwenden
    //   2. correctOdom(z_odom_map)       — mit Odometrie korrigieren
    //   3. correctLandmark(z_lm)         — falls Landmark erkannt wurde
    //   4. publishPose()                 — nur EINMAL am Ende publizieren
    // -------------------------------------------------------------------------
    odom_subscription_ =
      this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10,
        [this](nav_msgs::msg::Odometry::UniquePtr msg) {
          if (!initialized_) { return; }

          rclcpp::Time current_time = msg->header.stamp;

          const double x_odom     = msg->pose.pose.position.x;
          const double y_odom     = msg->pose.pose.position.y;
          const double theta_odom = getYaw(msg->pose.pose.orientation);

          // --- Erster Odom nach Init: Offset berechnen ---
          if (!odom_initialized_) {
            const Vector6d & s = filter_.state();
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

          // --- Odom-Pose in Map-Frame transformieren ---
          const double cos_o = std::cos(offset_theta_);
          const double sin_o = std::sin(offset_theta_);

          Eigen::Vector3d z_odom_map;
          z_odom_map(0) = cos_o * x_odom - sin_o * y_odom + offset_x_;
          z_odom_map(1) = sin_o * x_odom + cos_o * y_odom + offset_y_;
          z_odom_map(2) = correctAngle(theta_odom + offset_theta_);

          // --- Zeitdelta prüfen ---
          double dt = (current_time - last_time_).seconds();
          last_time_ = current_time;

          if (dt <= 0.0 || dt > 1.0) { return; }
          if (!has_cmd_vel_)          { return; }

          double cmd_dt = std::abs((current_time - last_cmd_time_).seconds());
          if (cmd_dt > 0.10) { return; }

          Eigen::Vector2d u;
          u << last_v_, last_omega_;

          // --- Schritt 1: Prediction ---
          filter_.predict(u, dt);

          // --- Schritt 2: Odom-Korrektur ---
          filter_.correctOdom(z_odom_map);

          // --- Schritt 3: Landmark-Korrektur (falls vorhanden) ---
          if (has_landmark_) {
            Eigen::Vector2d z_lm;
            z_lm(0) = last_lm_r_;
            z_lm(1) = last_lm_phi_;

            Eigen::Vector2d landmark;
            landmark(0) = LANDMARK_X;
            landmark(1) = LANDMARK_Y;

            filter_.correctLandmark(z_lm, landmark);

            RCLCPP_INFO(this->get_logger(),
              "Landmark-Korrektur angewendet: x=%.3f, y=%.3f, theta=%.3f",
              filter_.state()(0), filter_.state()(1), filter_.state()(2));

            has_landmark_ = false;  // verbraucht — warten auf nächste Detection
          }

          // --- Schritt 4: Pose publizieren (nur einmal!) ---
          publishPose(filter_.state(), msg->header.stamp, "map");
        });

    pose_publisher_ =
      this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/ekf_pose", 10);

    RCLCPP_INFO(this->get_logger(),
      "EKF-Node gestartet. Warte auf /initialpose. Landmark bei (%.2f, %.2f)",
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

    const auto & cov = filter_.covariance();
    pose_msg.pose.covariance[0]  = cov(0, 0);  // x
    pose_msg.pose.covariance[7]  = cov(1, 1);  // y
    pose_msg.pose.covariance[35] = cov(2, 2);  // theta

    pose_publisher_->publish(pose_msg);
  }

  // Subscriptions & Publisher
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr                     cmd_vel_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr                       odom_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr                   scan_subscription_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr    pose_publisher_;

  ExtendedKalmanFilter filter_;

  // Map-to-Odom Offset
  double offset_x_{0.0};
  double offset_y_{0.0};
  double offset_theta_{0.0};

  // Letzter cmd_vel
  double last_v_{0.0};
  double last_omega_{0.0};
  rclcpp::Time last_cmd_time_;
  rclcpp::Time last_time_;

  // Zwischengespeicherte Landmark-Messung vom Scan-Callback
  double last_lm_r_{0.0};
  double last_lm_phi_{0.0};
  bool has_landmark_{false};

  // Flags
  bool has_cmd_vel_{false};
  bool initialized_{false};
  bool odom_initialized_{false};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ExtendedKalmanFilterNode>());
  rclcpp::shutdown();
  return 0;
}