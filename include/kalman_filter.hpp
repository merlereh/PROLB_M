#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>

// State: [x, y, theta, vx, vy, theta_dot]
// Control input: u = [v, omega]  (from /cmd_vel)
// Measurement:   z = [x, y, theta]  (from /odom, absolute pose)

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3x6d = Eigen::Matrix<double, 3, 6>;
using Matrix6x3d = Eigen::Matrix<double, 6, 3>;

class KalmanFilter
{
public:
    // Constructor
    KalmanFilter()
    {
        // Initial state = [x, y, theta, vx, vy, theta_dot]
        mu_ = Vector6d::Zero();

        // Initial covariance
        Sigma_ = Matrix6d::Zero();
        Sigma_(0, 0) = 0.5;   // x
        Sigma_(1, 1) = 0.5;   // y
        Sigma_(2, 2) = 0.1;   // theta
        Sigma_(3, 3) = 0.5;   // vx
        Sigma_(4, 4) = 0.5;   // vy
        Sigma_(5, 5) = 0.2;   // theta_dot

        // Measurement model matrix C (3x6)
        // z = C * x  →  we measure [x, y, theta] directly from odometry
        C_ = Matrix3x6d::Zero();
        C_(0, 0) = 1.0;   // x
        C_(1, 1) = 1.0;   // y
        C_(2, 2) = 1.0;   // theta

        // Process noise R (6x6)
        // Uncertainty in the motion model
        R_ = Matrix6d::Zero();
        R_(0, 0) = 0.1;    // x
        R_(1, 1) = 0.1;   // y
        R_(2, 2) = 0.02;   // theta
        R_(3, 3) = 0.1;    // vx
        R_(4, 4) = 0.1;   // vy
        R_(5, 5) = 0.05;   // theta_dot

        // Measurement noise Q (3x3)
        // Uncertainty in the odometry measurement
        Q_ = Eigen::Matrix3d::Zero();
        Q_(0, 0) = 0.1;   // x
        Q_(1, 1) = 0.1;   // y
        Q_(2, 2) = 0.02;   // theta

        // Kalman Gain (6x3)
        K_ = Matrix6x3d::Zero();
    }

    // Destructor
    ~KalmanFilter() = default;

    // Set initial state from first odometry message
    // Only pose part [x, y, theta] is initialized; velocities start at zero
    void setState(const Eigen::Vector3d & pose)
    {
        mu_(0) = pose(0);
        mu_(1) = pose(1);
        mu_(2) = correctAngle(pose(2));
        mu_(3) = 0.0;
        mu_(4) = 0.0;
        mu_(5) = 0.0;
    }

    // Prediction step
    // u = [v, omega]  from /cmd_vel
    // dt = time step
    //
    // A-Matrix (state transition):
    //   x_new        = x  + vx * dt
    //   y_new        = y  + vy * dt
    //   theta_new    = theta + theta_dot * dt
    //   vx_new       = vx   (constant velocity model, corrected by B*u)
    //   vy_new       = vy
    //   theta_dot_new = theta_dot
    //
    // B-Matrix (control input):
    //   vx        += v * cos(theta)
    //   vy        += v * sin(theta)
    //   theta_dot += omega
    Vector6d predict(const Eigen::Vector2d & u, double dt)
    {
        const double theta = mu_(2);
        const double v     = u(0);
        const double omega = u(1);

        // Build A matrix
        Matrix6d A = Matrix6d::Identity();
        A(0, 3) = dt;   // x  += vx * dt
        A(1, 4) = dt;   // y  += vy * dt
        A(2, 5) = dt;   // theta += theta_dot * dt

        // Build B matrix
        // cmd_vel sets the velocity states directly
        Matrix6d B = Matrix6d::Zero();
        B(3, 3) = 1.0;   // vx       = v * cos(theta)  (applied below)
        B(4, 4) = 1.0;   // vy       = v * sin(theta)
        B(5, 5) = 1.0;   // theta_dot = omega

        // Control vector in state space
        Vector6d u_full = Vector6d::Zero();
        u_full(3) = v * std::cos(theta);
        u_full(4) = v * std::sin(theta);
        u_full(5) = omega;

        // Predicted state: mu_bar = A * mu + B * u_full
        // Since B is identity on velocity rows, this sets velocities from cmd_vel
        // and integrates position/orientation from the previous velocity state
        mu_bar_ = A * mu_;
        // Overwrite velocity states with current cmd_vel (linear treatment as per Prof)
        mu_bar_(3) = v * std::cos(theta);
        mu_bar_(4) = v * std::sin(theta);
        mu_bar_(5) = omega;
        mu_bar_(2) = correctAngle(mu_bar_(2));

        // Predicted covariance: Sigma_bar = A * Sigma * A^T + R
        Sigma_bar_ = A * Sigma_ * A.transpose() + R_;

        return mu_bar_;
    }

    // Compute Kalman Gain  K = Sigma_bar * C^T * (C * Sigma_bar * C^T + Q)^-1
    Matrix6x3d computeKalmanGain()
    {
        Eigen::Matrix3d S = C_ * Sigma_bar_ * C_.transpose() + Q_;
        K_ = Sigma_bar_ * C_.transpose() * S.inverse();
        return K_;
    }

    // Correction step
    // z = [x, y, theta]  from /odom (absolute pose)
    Vector6d correct(const Eigen::Vector3d & z)
    {
        // Innovation: difference between measurement and predicted measurement
        Eigen::Vector3d innovation = z - C_ * mu_bar_;
        innovation(2) = correctAngle(innovation(2));

        // Corrected state
        mu_ = mu_bar_ + K_ * innovation;
        mu_(2) = correctAngle(mu_(2));

        // Corrected covariance
        Sigma_ = (Matrix6d::Identity() - K_ * C_) * Sigma_bar_;

        return mu_;
    }

    // Full update: predict + Kalman gain + correct
    Vector6d update(
        const Eigen::Vector2d & u,
        const Eigen::Vector3d & z,
        double dt)
    {
        predict(u, dt);
        computeKalmanGain();
        correct(z);
        return mu_;
    }

    // Getters
    const Vector6d & state() const { return mu_; }
    const Vector6d & predictedState() const { return mu_bar_; }
    const Matrix6d & covariance() const { return Sigma_; }
    const Matrix6d & predictedCovariance() const { return Sigma_bar_; }
    const Matrix6x3d & kalmanGain() const { return K_; }

private:
    static double correctAngle(double angle)
    {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    // State [x, y, theta, vx, vy, theta_dot]
    Vector6d mu_;
    Vector6d mu_bar_ = Vector6d::Zero();

    // Covariance (6x6)
    Matrix6d Sigma_;
    Matrix6d Sigma_bar_ = Matrix6d::Zero();

    // Measurement model matrix (3x6)
    Matrix3x6d C_;

    // Process noise (6x6)
    Matrix6d R_;

    // Measurement noise (3x3)
    Eigen::Matrix3d Q_;

    // Kalman Gain (6x3)
    Matrix6x3d K_;
};