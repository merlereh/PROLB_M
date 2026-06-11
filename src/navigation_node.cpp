#include <memory>
#include <vector>
#include <chrono>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using namespace std::chrono_literals;

struct Waypoint
{
  double x;
  double y;
  double yaw;
};

class Nav2WaypointNode : public rclcpp::Node
{
public:
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandle = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  Nav2WaypointNode()
  : Node("nav2_waypoint_node")
  {
    initial_pose_pub_ =
      create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose",
        10
      );

    nav_client_ =
      rclcpp_action::create_client<NavigateToPose>(
        this,
        "/navigate_to_pose"
      );

    goals_ = {
      {-2.0, 1.0, 0.0},
      { 1.0, 1.0, 1.57},
      { 0.0, 1.0, 3.14},
      { 0.0, 0.0, -1.57}
    };

    start_timer_ = create_wall_timer(
      1s,
      std::bind(&Nav2WaypointNode::start, this)
    );

    RCLCPP_INFO(get_logger(), "Nav2 waypoint node started.");
  }

private:
  void start()
  {
    start_timer_->cancel();

    if (!nav_client_->wait_for_action_server(10s)) {
      RCLCPP_ERROR(get_logger(), "Nav2 action server not available.");
      return;
    }

    publishInitialPose(-2.0, -0.5, 0.0);

    // Do not sleep here.
    // Give AMCL/Nav2 time using a timer instead.
    goal_timer_ = create_wall_timer(
      2s,
      std::bind(&Nav2WaypointNode::sendNextGoalOnce, this)
    );
  }

  void sendNextGoalOnce()
  {
    goal_timer_->cancel();
    sendNextGoal();
  }

  void publishInitialPose(double x, double y, double yaw)
  {
    geometry_msgs::msg::PoseWithCovarianceStamped msg;

    msg.header.frame_id = "map";
    msg.header.stamp = now();

    msg.pose.pose.position.x = x;
    msg.pose.pose.position.y = y;
    msg.pose.pose.position.z = 0.0;
    msg.pose.pose.orientation = yawToQuaternion(yaw);

    msg.pose.covariance[0] = 0.25;     // x
    msg.pose.covariance[7] = 0.25;     // y
    msg.pose.covariance[35] = 0.0685;  // yaw

    initial_pose_pub_->publish(msg);

    RCLCPP_INFO(
      get_logger(),
      "Initial pose published: x=%.2f y=%.2f yaw=%.2f",
      x, y, yaw
    );
  }

  void sendNextGoal()
  {
    if (current_goal_ >= goals_.size()) {
      RCLCPP_INFO(get_logger(), "All goals completed.");
      return;
    }

    const auto & wp = goals_[current_goal_];

    NavigateToPose::Goal goal;
    goal.pose = makePose(wp.x, wp.y, wp.yaw);

    RCLCPP_INFO(
      get_logger(),
      "Sending goal %zu: x=%.2f y=%.2f yaw=%.2f",
      current_goal_ + 1,
      wp.x,
      wp.y,
      wp.yaw
    );

    auto options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();

    options.goal_response_callback =
      [this](const GoalHandle::SharedPtr & handle)
      {
        if (!handle) {
          RCLCPP_ERROR(get_logger(), "Goal rejected.");
        } else {
          RCLCPP_INFO(get_logger(), "Goal accepted.");
        }
      };

    options.result_callback =
      [this](const GoalHandle::WrappedResult & result)
      {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
          RCLCPP_INFO(get_logger(), "Goal %zu succeeded.", current_goal_ + 1);
          current_goal_++;
          sendNextGoal();
        } else {
          RCLCPP_ERROR(
            get_logger(),
            "Goal %zu failed or was canceled.",
            current_goal_ + 1
          );
        }
      };

    nav_client_->async_send_goal(goal, options);
  }

  geometry_msgs::msg::PoseStamped makePose(double x, double y, double yaw)
  {
    geometry_msgs::msg::PoseStamped msg;

    msg.header.frame_id = "map";
    msg.header.stamp = now();

    msg.pose.position.x = x;
    msg.pose.position.y = y;
    msg.pose.position.z = 0.0;
    msg.pose.orientation = yawToQuaternion(yaw);

    return msg;
  }

  geometry_msgs::msg::Quaternion yawToQuaternion(double yaw)
  {
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);
    q.normalize();
    return tf2::toMsg(q);
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_pub_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;

  rclcpp::TimerBase::SharedPtr start_timer_;
  rclcpp::TimerBase::SharedPtr goal_timer_;

  std::vector<Waypoint> goals_;
  std::size_t current_goal_{0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Nav2WaypointNode>());
  rclcpp::shutdown();
  return 0;
}