#include <memory>
#include <vector>
#include <chrono>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "nav2_msgs/action/navigate_through_poses.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

using namespace std::chrono_literals;
using NavigateThroughPoses = nav2_msgs::action::NavigateThroughPoses;
using GoalHandleNav = rclcpp_action::ClientGoalHandle<NavigateThroughPoses>;

class Nav2WaypointNode : public rclcpp::Node
{
public:
  Nav2WaypointNode()
  : Node("nav2_waypoint_node")
  {
    client_ = rclcpp_action::create_client<NavigateThroughPoses>(
      this,
      "navigate_through_poses"
    );

    // Wait for Nav2 action server, then send waypoints.
    timer_ = this->create_wall_timer(
      2s,
      std::bind(&Nav2WaypointNode::try_send_goal, this)
    );

    RCLCPP_INFO(this->get_logger(), "Waiting for Nav2 action server ...");
  }

private:
  std::vector<geometry_msgs::msg::PoseStamped> build_waypoints()
  {
    // Waypoints: {x, y, yaw}
    const std::vector<std::array<double, 3>> coords = {
      { -2.0,  -0.5,  0.0  },
      { -0.5,  -0.5,  1.4  },
      { -0.5,   0.5,  3.14 },
      { -1.75,  1.3,  1.4  },
      { -1.35,  1.75, 0.2  },
      { -0.1,   1.9,  0.0  },
      {  1.3,   1.6,  0.1  },
      {  1.7,   0.55,-3.14 },
      {  0.85,  0.5, -1.5  },
      {  0.56, -0.45, 0.0  },
      {  0.25, -0.61, 0.0  },
      { -0.5,  -0.65,-1.7  },
      { -0.65, -1.6, -3.14 },
      { -1.6,  -1.5,  2.2  },
      { -2.0,  -0.5,  0.0  },
    };

    std::vector<geometry_msgs::msg::PoseStamped> waypoints;
    waypoints.reserve(coords.size());

    for (const auto & c : coords) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header.frame_id = "map";
      ps.header.stamp    = this->now();

      ps.pose.position.x = c[0];
      ps.pose.position.y = c[1];
      ps.pose.position.z = 0.0;

      const double yaw = c[2];
      ps.pose.orientation.x = 0.0;
      ps.pose.orientation.y = 0.0;
      ps.pose.orientation.z = std::sin(yaw / 2.0);
      ps.pose.orientation.w = std::cos(yaw / 2.0);

      waypoints.push_back(ps);
    }

    return waypoints;
  }

  void try_send_goal()
  {
    if (!client_->action_server_is_ready()) {
      RCLCPP_INFO(this->get_logger(), "Action server not ready yet, retrying ...");
      return;
    }

    timer_->cancel();

    auto goal = NavigateThroughPoses::Goal();
    goal.poses = build_waypoints();

    RCLCPP_INFO(
      this->get_logger(),
      "Sending %zu waypoints to Nav2 ...",
      goal.poses.size()
    );

    auto send_goal_options = rclcpp_action::Client<NavigateThroughPoses>::SendGoalOptions();

    send_goal_options.goal_response_callback =
      [this](const GoalHandleNav::SharedPtr & handle) {
        if (!handle) {
          RCLCPP_ERROR(this->get_logger(), "Goal was rejected by Nav2.");
        } else {
          RCLCPP_INFO(this->get_logger(), "Goal accepted — robot is navigating.");
        }
      };

    send_goal_options.feedback_callback =
      [this](GoalHandleNav::SharedPtr, const std::shared_ptr<const NavigateThroughPoses::Feedback> feedback) {
        RCLCPP_INFO(
          this->get_logger(),
          "Waypoints remaining: %d",
          feedback->number_of_poses_remaining
        );
      };

    send_goal_options.result_callback =
      [this](const GoalHandleNav::WrappedResult & result) {
        switch (result.code) {
          case rclcpp_action::ResultCode::SUCCEEDED:
            RCLCPP_INFO(this->get_logger(), "Navigation complete — all waypoints reached!");
            break;
          case rclcpp_action::ResultCode::ABORTED:
            RCLCPP_ERROR(this->get_logger(), "Navigation aborted by Nav2.");
            break;
          case rclcpp_action::ResultCode::CANCELED:
            RCLCPP_WARN(this->get_logger(), "Navigation was canceled.");
            break;
          default:
            RCLCPP_ERROR(this->get_logger(), "Unknown result code.");
            break;
        }
      };

    client_->async_send_goal(goal, send_goal_options);
  }

  rclcpp_action::Client<NavigateThroughPoses>::SharedPtr client_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Nav2WaypointNode>());
  rclcpp::shutdown();
  return 0;
}