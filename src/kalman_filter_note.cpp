#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"

class CmdVelSubscriber : public rclcpp::Node
{
public:
  CmdVelSubscriber()
  : Node("cmd_vel_subscriber")
  {
    auto topic_callback =
      [this](geometry_msgs::msg::Twist::UniquePtr msg) -> void {
        RCLCPP_INFO(
          this->get_logger(),
          "I heard cmd_vel: linear.x = %.3f, angular.z = %.3f",
          msg->linear.x,
          msg->angular.z
        );
      };

    subscription_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel",
        10,
        topic_callback
      );
  }

private:
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CmdVelSubscriber>());
  rclcpp::shutdown();
  return 0;
}