#include <memory>
#include <chrono>
#include <cmath>
#include <vector>
#include <array>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"

using namespace std::chrono_literals;
using NavigateToPose = nav2_msgs::action::NavigateToPose;
using GoalHandleNav  = rclcpp_action::ClientGoalHandle<NavigateToPose>;

class InitialPoseAndWaypointNode : public rclcpp::Node
{
public:
  InitialPoseAndWaypointNode()
  : Node("initial_pose_node")
  {
    pose_publisher_ =
      this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", rclcpp::QoS(10)
      );

    nav_client_ = rclcpp_action::create_client<NavigateToPose>(
      this, "navigate_to_pose"
    );

    timer_ = this->create_wall_timer(
      1s, std::bind(&InitialPoseAndWaypointNode::tick, this)
    );

    RCLCPP_INFO(this->get_logger(), "Initial pose and waypoint node started.");
  }

  ~InitialPoseAndWaypointNode()
  {
    if (current_goal_handle_) {
      RCLCPP_INFO(this->get_logger(), "Cancelling active navigation goal ...");
      auto future = nav_client_->async_cancel_goal(current_goal_handle_);
      rclcpp::spin_until_future_complete(
        this->get_node_base_interface(), future, std::chrono::seconds(2));
    }
  }

private:
  // Waypoints: {x, y, yaw}
  // Yaw at point 10 and 11 corrected to match direction of travel:
  //   WP10 (0.56, -0.45): approaching from (0.85, 0.5) -> direction ~atan2(-0.95, -0.29) ~ -1.86 rad
  //   WP11 (0.25, -0.61): approaching from (0.56,-0.45) -> direction ~atan2(-0.16, -0.31) ~ -2.67 rad
  const std::vector<std::array<double, 3>> waypoints_ = {
    { -2.0,  -0.5,   0.0  },
    { -0.5,  -0.5,   1.4  },
    { -0.5,   0.5,   3.14 },
    { -1.75,  1.3,   1.4  },
    { -1.35,  1.75,  0.2  },
    { -0.1,   1.9,   0.0  },
    {  1.3,   1.6,   0.1  },
    {  1.7,   0.55, -3.14 },
    {  0.85,  0.5,  -1.5  },
    {  0.56, -0.45, -1.86 },  // corrected: facing direction of travel toward WP11
    {  0.25, -0.61, -2.67 },  // corrected: facing direction of travel toward WP12
    { -0.5,  -0.65, -1.7  },
    { -0.65, -1.6,  -3.14 },
    { -1.6,  -1.5,   2.2  },
    { -2.0,  -0.5,   0.0  },
  };

  void tick()
  {
    if (!init_pose_done_) {
      sendInitialPose();
      init_pose_done_ = true;
      return;
    }

    if (waiting_for_result_) {
      return;
    }

    if (current_waypoint_index_ >= static_cast<int>(waypoints_.size())) {
      RCLCPP_INFO(this->get_logger(), "All waypoints reached!");
      timer_->cancel();
      return;
    }

    sendNextWaypoint();
  }

  void sendInitialPose()
  {
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

    pose_publisher_->publish(msg);
    RCLCPP_INFO(this->get_logger(),
      "Initial pose sent: x=%.2f, y=%.2f, yaw=%.2f", x, y, yaw);
  }

  void sendNextWaypoint()
  {
    if (!nav_client_->action_server_is_ready()) {
      RCLCPP_INFO(this->get_logger(), "Nav2 action server not ready yet ...");
      return;
    }

    const auto & wp = waypoints_[current_waypoint_index_];

    NavigateToPose::Goal goal;
    goal.pose.header.stamp    = this->now();
    goal.pose.header.frame_id = "map";

    goal.pose.pose.position.x = wp[0];
    goal.pose.pose.position.y = wp[1];
    goal.pose.pose.position.z = 0.0;

    const double yaw = wp[2];
    goal.pose.pose.orientation.x = 0.0;
    goal.pose.pose.orientation.y = 0.0;
    goal.pose.pose.orientation.z = std::sin(yaw / 2.0);
    goal.pose.pose.orientation.w = std::cos(yaw / 2.0);

    RCLCPP_INFO(this->get_logger(),
      "Sending waypoint %d/%zu: x=%.2f, y=%.2f, yaw=%.2f",
      current_waypoint_index_ + 1, waypoints_.size(), wp[0], wp[1], yaw);

    auto options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();

    options.goal_response_callback =
      [this](const GoalHandleNav::SharedPtr & handle) {
        if (!handle) {
          RCLCPP_ERROR(this->get_logger(), "Waypoint goal rejected by Nav2.");
          waiting_for_result_ = false;
        } else {
          current_goal_handle_ = handle;
          RCLCPP_INFO(this->get_logger(), "Waypoint goal accepted.");
        }
      };

    options.feedback_callback =
      [this](GoalHandleNav::SharedPtr,
             const std::shared_ptr<const NavigateToPose::Feedback> fb) {
        RCLCPP_DEBUG(this->get_logger(),
          "Distance remaining: %.2f m", fb->distance_remaining);
      };

    options.result_callback =
      [this](const GoalHandleNav::WrappedResult & result) {
        current_goal_handle_ = nullptr;
        waiting_for_result_  = false;

        switch (result.code) {
          case rclcpp_action::ResultCode::SUCCEEDED:
            RCLCPP_INFO(this->get_logger(),
              "Waypoint %d reached!", current_waypoint_index_ + 1);
            current_waypoint_index_++;
            break;
          case rclcpp_action::ResultCode::ABORTED:
            RCLCPP_WARN(this->get_logger(),
              "Waypoint %d aborted, retrying ...", current_waypoint_index_ + 1);
            break;
          case rclcpp_action::ResultCode::CANCELED:
            RCLCPP_WARN(this->get_logger(), "Waypoint goal canceled.");
            break;
          default:
            RCLCPP_ERROR(this->get_logger(), "Unknown result code.");
            break;
        }
      };

    waiting_for_result_ = true;
    nav_client_->async_send_goal(goal, options);
  }

  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;
  rclcpp::TimerBase::SharedPtr timer_;
  GoalHandleNav::SharedPtr current_goal_handle_{nullptr};

  bool init_pose_done_{false};
  bool waiting_for_result_{false};
  int  current_waypoint_index_{0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<InitialPoseAndWaypointNode>());
  rclcpp::shutdown();
  return 0;
}