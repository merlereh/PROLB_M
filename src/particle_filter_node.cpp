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

// ---------------------------------------------------------------------------
// ParticleFilterNode
//
// Trigger:  /odom + /scan     → Synchronizer (ApproximateTime, max 0.1s)
// Input:    /cmd_vel          → Control u = [v, omega] (gecacht, max. 0.2s alt)
//           /odom             → z = [x_map, y_map, theta_map]
//           /scan             → Laser für Landmark
//           /initialpose      → Partikel um Startpose initialisieren
// Output:   /pf_pose          → geschätzte Pose als PoseWithCovarianceStamped
//           /my_particle_cloud → alle Partikel als PoseArray
// ---------------------------------------------------------------------------

using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    nav_msgs::msg::Odometry,
    sensor_msgs::msg::LaserScan>;

class ParticleFilterNode : public rclcpp::Node
{
public:
  ParticleFilterNode() : Node("pf_node")
  {
    // --- ROS-Parameter für Rauschmatrizen R und Q ---
    this->declare_parameter("r_x",                 0.05);
    this->declare_parameter("r_y",                 0.05);
    this->declare_parameter("r_theta",             0.02);
    this->declare_parameter("r_vx",                0.10);
    this->declare_parameter("r_vy",                0.05);
    this->declare_parameter("r_omega",             0.05);
    this->declare_parameter("q_x",                 0.10);
    this->declare_parameter("q_y",                 0.10);
    this->declare_parameter("q_theta",             0.05);
    this->declare_parameter("q_lm_r",              0.05);
    this->declare_parameter("q_lm_phi",            0.01);
    this->declare_parameter("pf_threshold_factor", 0.5);

    // --- ROS-Parameter für künstliches Odom-Rauschen (Standardabweichung) ---
    this->declare_parameter("odom_noise_x",     0.05);
    this->declare_parameter("odom_noise_y",     0.05);
    this->declare_parameter("odom_noise_theta", 0.02);
    odom_noise_x_     = this->get_parameter("odom_noise_x").as_double();
    odom_noise_y_     = this->get_parameter("odom_noise_y").as_double();
    odom_noise_theta_ = this->get_parameter("odom_noise_theta").as_double();

    // --- Filter mit Parametern initialisieren ---
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

    filter_.threshold_factor_ = this->get_parameter("pf_threshold_factor").as_double();

    // --- Subscribe: /initialpose → Partikel um Startpose initialisieren ---
    initialpose_sub_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 10,
        [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg) {
          Eigen::Vector3d pose;
          pose << msg->pose.pose.position.x,
                  msg->pose.pose.position.y,
                  getYaw(msg->pose.pose.orientation);
          filter_.initializeParticlesAroundState(pose);
          odom_initialized_ = false;
          initialized_      = true;
          RCLCPP_INFO(this->get_logger(),
            "PF init: x=%.3f y=%.3f th=%.3f", pose(0), pose(1), pose(2));
        });

    // --- Subscribe: /cmd_vel → Control-Input u = [v, omega] cachen ---
    cmd_vel_sub_ =
      this->create_subscription<geometry_msgs::msg::Twist>("/cmd_vel", 10,
        [this](geometry_msgs::msg::Twist::UniquePtr msg) {
          last_v_        = msg->linear.x;
          last_omega_    = msg->angular.z;
          last_cmd_time_ = this->now();
          has_cmd_vel_   = true;
        });

    // --- Synchronizer: /odom + /scan müssen zeitlich zusammenpassen ---
    odom_sub_.subscribe(this, "/odom");
    scan_sub_.subscribe(this, "/scan");

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(20), odom_sub_, scan_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));
    sync_->registerCallback(
        std::bind(&ParticleFilterNode::syncCallback, this,
                  std::placeholders::_1, std::placeholders::_2));

    // --- Publish: /pf_pose → geschätzte Pose ---
    pose_pub_ =
      this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/pf_pose", 10);

    // --- Publish: /my_particle_cloud → alle Partikel als PoseArray ---
    particle_pub_ =
      this->create_publisher<geometry_msgs::msg::PoseArray>(
        "/my_particle_cloud", 10);

    RCLCPP_INFO(this->get_logger(),
      "PF-Node gestartet. Warte auf /initialpose. Landmark bei (%.2f, %.2f)",
      LANDMARK_X, LANDMARK_Y);
  }

private:
  void syncCallback(
    const nav_msgs::msg::Odometry::ConstSharedPtr      & odom_msg,
    const sensor_msgs::msg::LaserScan::ConstSharedPtr  & scan_msg)
  {
    if (!initialized_) return;
    if (!has_cmd_vel_) return;
    if (std::abs((this->now() - last_cmd_time_).seconds()) > 0.2) return;

    rclcpp::Time current_time = odom_msg->header.stamp;

    const double x_odom     = odom_msg->pose.pose.position.x     + noiseX();
    const double y_odom     = odom_msg->pose.pose.position.y     + noiseY();
    const double theta_odom = correctAngle(getYaw(odom_msg->pose.pose.orientation) + noiseTheta());

    // --- Einmalig: Offset odom → map berechnen ---
    if (!odom_initialized_) {
      const Vector6d & s = filter_.state();
      offset_theta_ = correctAngle(s(2) - theta_odom);
      const double co = std::cos(offset_theta_), so = std::sin(offset_theta_);
      offset_x_ = s(0) - (co * x_odom - so * y_odom);
      offset_y_ = s(1) - (so * x_odom + co * y_odom);
      last_time_        = current_time;
      odom_initialized_ = true;
      RCLCPP_INFO(this->get_logger(),
        "Odom-Offset: dx=%.3f dy=%.3f dth=%.3f",
        offset_x_, offset_y_, offset_theta_);
      return;
    }

    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;
    if (dt <= 0.0 || dt > 1.0) return;

    // --- z = [x_map, y_map, theta_map] ---
    const double co = std::cos(offset_theta_), so = std::sin(offset_theta_);
    const double x_map     = co * x_odom - so * y_odom + offset_x_;
    const double y_map     = so * x_odom + co * y_odom + offset_y_;
    const double theta_map = correctAngle(theta_odom + offset_theta_);

    Eigen::Vector3d z_full;
    z_full(0) = x_map;
    z_full(1) = y_map;
    z_full(2) = theta_map;

    // --- 1) PF Predict + Weight (Full) + Resample ---
    Eigen::Vector2d u;
    u << last_v_, last_omega_;

    const auto t_start = std::chrono::steady_clock::now();

    Vector6d estimate = filter_.update(u, z_full, dt);

    // --- 2) PF Landmark Update ---
    const Vector6d & state = filter_.state();
    double r_meas, phi_meas;
    if (detectLandmark(*scan_msg, state(0), state(1), state(2),
                       LANDMARK_X, LANDMARK_Y, r_meas, phi_meas)) {
      Eigen::Vector2d z_lm; z_lm << r_meas, phi_meas;
      Eigen::Vector2d lm;   lm   << LANDMARK_X, LANDMARK_Y;
      estimate = filter_.updateLandmark(z_lm, lm);
      RCLCPP_INFO(this->get_logger(),
        "Landmark: r=%.3f phi=%.3f -> x=%.3f y=%.3f",
        r_meas, phi_meas, estimate(0), estimate(1));
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double runtime_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // --- 3) Pose + Partikel publizieren ---
    publishPose(estimate, odom_msg->header.stamp, "map", runtime_ms);
  }

  static double correctAngle(double a)
  { return std::atan2(std::sin(a), std::cos(a)); }

  // --- Künstliches Gaussian-Rauschen auf die Odom-Pose ---
  double noiseX()     { return std::normal_distribution<double>(0.0, odom_noise_x_)(rng_); }
  double noiseY()     { return std::normal_distribution<double>(0.0, odom_noise_y_)(rng_); }
  double noiseTheta() { return std::normal_distribution<double>(0.0, odom_noise_theta_)(rng_); }

  double getYaw(const geometry_msgs::msg::Quaternion & q_msg)
  {
    tf2::Quaternion q(q_msg.x, q_msg.y, q_msg.z, q_msg.w);
    double r, p, y; tf2::Matrix3x3(q).getRPY(r, p, y); return y;
  }

  void publishPose(const Vector6d & s, const rclcpp::Time & stamp, const std::string & fid, double runtime_ms)
  {
    geometry_msgs::msg::PoseWithCovarianceStamped msg;
    msg.header.stamp    = stamp;
    msg.header.frame_id = fid;
    msg.pose.pose.position.x = s(0);
    msg.pose.pose.position.y = s(1);
    tf2::Quaternion q; q.setRPY(0, 0, s(2));
    msg.pose.pose.orientation = tf2::toMsg(q);
    for (int i = 0; i < 36; ++i) msg.pose.covariance[i] = 0.0;

    // Empirische Kovarianz aus Partikelwolke
    {
      const auto & particles = filter_.particles();
      const double mx = s(0), my = s(1);
      double cxx = 0, cyy = 0, cxy = 0;
      for (const auto & p : particles) {
        double ex = p(0) - mx, ey = p(1) - my;
        cxx += ex*ex; cyy += ey*ey; cxy += ex*ey;
      }
      double n = static_cast<double>(particles.size());
      msg.pose.covariance[0] = cxx/n;
      msg.pose.covariance[7] = cyy/n;
      msg.pose.covariance[1] = cxy/n;
      msg.pose.covariance[6] = cxy/n;
    }

    // PoseWithCovarianceStamped hat keine Felder für Evaluations-Metriken.
    // Statt eines eigenen .msg-Typs (Rebuild der Interfaces nötig) werden
    // ungenutzte Diagonal-Slots der 6x6-Kovarianz zweckentfremdet — bei
    // einem 2D-Roboter sind z/roll/pitch-Varianz (Indizes 14/21/28) ohnehin
    // immer 0 und kollidieren nicht mit x/y/theta (Indizes 0,1,6,7,35).
    //   covariance[14] (z-z)       -> runtime_ms (Dauer dieses Update-Schritts)
    //   covariance[21] (roll-roll) -> ess (Effective Sample Size)
    //   covariance[28] (pitch-pitch) -> resampling_triggered (1.0/0.0)
    msg.pose.covariance[14] = runtime_ms;
    msg.pose.covariance[21] = filter_.ess();
    msg.pose.covariance[28] = filter_.resamplingTriggered() ? 1.0 : 0.0;

    pose_pub_->publish(msg);

    // Alle Partikel als PoseArray publizieren
    geometry_msgs::msg::PoseArray pa;
    pa.header = msg.header;
    for (const auto & p : filter_.particles()) {
      geometry_msgs::msg::Pose pose;
      pose.position.x = p(0);
      pose.position.y = p(1);
      tf2::Quaternion pq; pq.setRPY(0, 0, p(2));
      pose.orientation = tf2::toMsg(pq);
      pa.poses.push_back(pose);
    }
    particle_pub_->publish(pa);
  }

  message_filters::Subscriber<nav_msgs::msg::Odometry>      odom_sub_;
  message_filters::Subscriber<sensor_msgs::msg::LaserScan>  scan_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr                 particle_pub_;

  ParticleFilter filter_;

  double offset_x_{0.0}, offset_y_{0.0}, offset_theta_{0.0};
  double last_v_{0.0}, last_omega_{0.0};
  rclcpp::Time last_cmd_time_, last_time_;
  bool has_cmd_vel_{false}, initialized_{false}, odom_initialized_{false};

  // --- Künstliches Odom-Rauschen ---
  double odom_noise_x_{0.0}, odom_noise_y_{0.0}, odom_noise_theta_{0.0};
  std::mt19937 rng_{std::random_device{}()};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ParticleFilterNode>());
  rclcpp::shutdown();
  return 0;
}