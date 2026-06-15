#include <memory>
#include <chrono>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"

using namespace std::chrono_literals;

class InitialPoseNode : public rclcpp::Node
{
public:
  InitialPoseNode()
  : Node("initial_pose_node")
  {
    publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/initialpose",
      rclcpp::QoS(10)
    );

    // Retry every 1 s until AMCL has a subscriber and we confirmed the send.
    timer_ = this->create_wall_timer(
      1s,
      std::bind(&InitialPoseNode::try_publish, this)
    );

    RCLCPP_INFO(this->get_logger(), "Waiting for AMCL to subscribe to /initialpose ...");
  }

private:
  void try_publish()
  {
    // Only send once AMCL is actually listening.
    if (publisher_->get_subscription_count() == 0) {
      RCLCPP_INFO(this->get_logger(), "No subscriber yet on /initialpose, retrying ...");
      return;
    }

    const double x   = -2.0;
    const double y   = -0.5;
    const double yaw =  0.0;

    geometry_msgs::msg::PoseWithCovarianceStamped msg;
    msg.header.stamp    = this->now();
    msg.header.frame_id = "map";

    msg.pose.pose.position.x = x;
    msg.pose.pose.position.y = y;
    msg.pose.pose.position.z = 0.0;

    msg.pose.pose.orientation.x = 0.0;
    msg.pose.pose.orientation.y = 0.0;
    msg.pose.pose.orientation.z = std::sin(yaw / 2.0);
    msg.pose.pose.orientation.w = std::cos(yaw / 2.0);

    msg.pose.covariance = {
      0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
      0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
      0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
      0.0, 0.0,  0.0, 0.0, 0.0, 0.0685
    };

    publisher_->publish(msg);

    RCLCPP_INFO(
      this->get_logger(),
      "Initial pose sent: x=%.2f, y=%.2f, yaw=%.2f",
      x, y, yaw
    );

    // Stop retrying — job done.
    timer_->cancel();
  }

  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<InitialPoseNode>());
  rclcpp::shutdown();
  return 0;
}