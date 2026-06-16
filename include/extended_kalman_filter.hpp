#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>

// State: [x, y, theta, vx, vy, theta_dot]
// Control input: u = [v, omega]  (from /cmd_vel)
// Measurement:   z = [x, y, theta]  (from /odom, absolute pose)
//
// The EKF uses the nonlinear motion function g(u, mu):
//   x_new        = x + vx * dt            (integration from velocity state)
//   y_new        = y + vy * dt
//   theta_new    = theta + theta_dot * dt
//   vx_new       = v * cos(theta)          (set from cmd_vel with cos/sin)
//   vy_new       = v * sin(theta)
//   theta_dot_new = omega
//
// The Jacobian G = dg/dx linearizes around the current state mu.

using Vector6d   = Eigen::Matrix<double, 6, 1>;
using Matrix6d   = Eigen::Matrix<double, 6, 6>;
using Matrix3x6d = Eigen::Matrix<double, 3, 6>;
using Matrix6x3d = Eigen::Matrix<double, 6, 3>;

class ExtendedKalmanFilter
{
public:
    ExtendedKalmanFilter()
    {
        // Initial state [x, y, theta, vx, vy, theta_dot]
        mu_ = Vector6d::Zero();

        // Initial covariance
        Sigma_ = Matrix6d::Zero();
        Sigma_(0, 0) = 0.5;    // x
        Sigma_(1, 1) = 0.5;    // y
        Sigma_(2, 2) = 0.1;    // theta
        Sigma_(3, 3) = 0.5;    // vx
        Sigma_(4, 4) = 0.5;    // vy
        Sigma_(5, 5) = 0.2;    // theta_dot

        // Measurement model H (3x6): z = H * x
        // We measure [x, y, theta] directly from odometry
        H_ = Matrix3x6d::Zero();
        H_(0, 0) = 1.0;   // x
        H_(1, 1) = 1.0;   // y
        H_(2, 2) = 1.0;   // theta

        // Process noise R (6x6)
        R_ = Matrix6d::Zero();
        R_(0, 0) = 0.1;   // x
        R_(1, 1) = 0.1;   // y
        R_(2, 2) = 0.02;   // theta
        R_(3, 3) = 0.1;    // vx
        R_(4, 4) = 0.1;   // vy
        R_(5, 5) = 0.02;   // theta_dot

        // Measurement noise Q (3x3)
        Q_ = Eigen::Matrix3d::Zero();
        Q_(0, 0) = 0.1;   // x
        Q_(1, 1) = 0.1;   // y
        Q_(2, 2) = 0.02;   // theta

        // Jacobian G (6x6), initialized to identity
        G_ = Matrix6d::Identity();

        // Kalman Gain (6x3)
        K_ = Matrix6x3d::Zero();
    }

    ~ExtendedKalmanFilter() = default;

    // Set initial state from first odometry measurement
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
    // Applies nonlinear g(u, mu) and linearizes with Jacobian G
    Vector6d predict(const Eigen::Vector2d & u, double dt)
    {
        const double theta = mu_(2);
        const double v     = u(0);
        const double omega = u(1);

        // Nonlinear motion function g(u, mu):
        // Positions integrate from current velocity states (smooth propagation),
        // velocity states are set from cmd_vel with cos/sin (nonlinear part)
        mu_bar_(0) = mu_(0) + mu_(3) * dt;           // x  += vx * dt
        mu_bar_(1) = mu_(1) + mu_(4) * dt;           // y  += vy * dt
        mu_bar_(2) = correctAngle(mu_(2) + mu_(5) * dt); // theta += theta_dot * dt
        mu_bar_(3) = v * std::cos(theta);             // vx = v * cos(theta)
        mu_bar_(4) = v * std::sin(theta);             // vy = v * sin(theta)
        mu_bar_(5) = omega;                           // theta_dot = omega

        // Jacobian G = dg/dx evaluated at mu (the linearization point)
        //
        // g1 = x + vx*dt         → dg1/dvx = dt,  dg1/dtheta = 0
        // g2 = y + vy*dt         → dg2/dvy = dt,  dg2/dtheta = 0
        // g3 = theta+theta_dot*dt→ dg3/dtheta_dot = dt
        // g4 = v*cos(theta)      → dg4/dtheta = -v*sin(theta)
        // g5 = v*sin(theta)      → dg5/dtheta =  v*cos(theta)
        // g6 = omega             → constant, no state dependency
        G_ = Matrix6d::Zero();

        // Position rows
        G_(0, 0) = 1.0;              // dx/dx
        G_(0, 3) = dt;               // dx/dvx
        G_(1, 1) = 1.0;              // dy/dy
        G_(1, 4) = dt;               // dy/dvy
        G_(2, 2) = 1.0;              // dtheta/dtheta
        G_(2, 5) = dt;               // dtheta/dtheta_dot

        // Velocity rows — nonlinear terms from cos/sin
        G_(3, 2) = -v * std::sin(theta);   // dvx/dtheta
        G_(4, 2) =  v * std::cos(theta);   // dvy/dtheta
        G_(5, 5) = 1.0;                    // dtheta_dot/dtheta_dot (omega is constant)

        // Predicted covariance: Sigma_bar = G * Sigma * G^T + R
        Sigma_bar_ = G_ * Sigma_ * G_.transpose() + R_;

        return mu_bar_;
    }

    // Compute Kalman Gain  K = Sigma_bar * H^T * (H * Sigma_bar * H^T + Q)^-1
    Matrix6x3d computeKalmanGain()
    {
        Eigen::Matrix3d S = H_ * Sigma_bar_ * H_.transpose() + Q_;
        K_ = Sigma_bar_ * H_.transpose() * S.inverse();
        return K_;
    }

    // Correction step
    // z = [x, y, theta] from /odom
    Vector6d correct(const Eigen::Vector3d & z)
    {
        // Innovation
        Eigen::Vector3d innovation = z - H_ * mu_bar_;
        innovation(2) = correctAngle(innovation(2));

        // Corrected state
        mu_ = mu_bar_ + K_ * innovation;
        mu_(2) = correctAngle(mu_(2));

        // Corrected covariance
        Sigma_ = (Matrix6d::Identity() - K_ * H_) * Sigma_bar_;

        return mu_;
    }

    // Full EKF update: predict → Kalman gain → correct
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
    const Vector6d   & state()              const { return mu_; }
    const Vector6d   & predictedState()     const { return mu_bar_; }
    const Matrix6d   & covariance()         const { return Sigma_; }
    const Matrix6d   & predictedCovariance()const { return Sigma_bar_; }
    const Matrix6d   & jacobianG()          const { return G_; }
    const Matrix6x3d & kalmanGain()         const { return K_; }

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

    // Measurement model matrix H (3x6)
    Matrix3x6d H_;

    // Jacobian of motion model (6x6)
    Matrix6d G_;

    // Process noise (6x6)
    Matrix6d R_;

    // Measurement noise (3x3)
    Eigen::Matrix3d Q_;

    // Kalman Gain (6x3)
    Matrix6x3d K_;
};