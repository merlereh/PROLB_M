#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
//#include "geometry_msgs/msg/twist.hpp"

class MinimalSubscriber : public rclcpp::Node
{
public:
  MinimalSubscriber()
  : Node("minimal_subscriber")
  {
    auto topic_callback =
      [this](std_msgs::msg::String::UniquePtr msg) -> void {
        RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
      };
    subscription_ =
      this->create_subscription<std_msgs::msg::String>("topic", 10, topic_callback);

    /*auto cmd_vel_callback =
    [this](geometry_msgs::msg::Twist::UniquePtr msg) -> void {
      double v_x = msg->linear.x;
      double omega = msg->angular.z;
    };

    cmd_vel_subscription_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel",
        10,
        cmd_vel_callback
      );*/
  }

 

private:
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}


