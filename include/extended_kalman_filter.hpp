#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>

class ExtendedKalmanFilter
{
public:
    // Constructor
    ExtendedKalmanFilter()
    {
        // beginning state = [x, y, theta]
        mu_ << 0.0, 0.0, 0.0;

        // covariance in the beginning
        Sigma_ = Eigen::Matrix3d::Zero();
        Sigma_(0, 0) = 0.2;
        Sigma_(1, 1) = 0.2;
        Sigma_(2, 2) = 0.1;

        // measurement model matrix
        // z = C * x
        // here we measure [x, y, theta] directly from odometry
        C_ = Eigen::Matrix3d::Identity();

        // process noise
        // describes uncertainty in the motion model
        R_ = Eigen::Matrix3d::Zero();
        R_(0, 0) = 0.05;
        R_(1, 1) = 0.05;
        R_(2, 2) = 0.02;

        // measurement noise
        // describes uncertainty in the odometry measurement
        Q_ = Eigen::Matrix3d::Zero();
        Q_(0, 0) = 0.01;
        Q_(1, 1) = 0.01;
        Q_(2, 2) = 0.02;

        // Jacobian of the motion model
        G_ = Eigen::Matrix3d::Identity();

        // Kalman Gain
        K_ = Eigen::Matrix3d::Zero();
    }

    // Destructor
    ~ExtendedKalmanFilter() = default;

    // set beginning state from outside, e.g. first odometry message
    void setState(const Eigen::Vector3d & mu)
    {
        mu_ = mu;
        mu_(2) = correctAngle(mu_(2));
    }

    // Prediction step
    // u = [v, omega]
    // dt = time difference between two odometry messages
    Eigen::Vector3d predict(const Eigen::Vector2d & u, double dt)
    {
        // current state
        const double theta = mu_(2);

        // control input
        const double v = u(0);
        const double omega = u(1);

        // predicted state using nonlinear TurtleBot/differential-drive motion model
        // this corresponds to g(mu, u, dt) in the Python code
        mu_bar_(0) = mu_(0) + v * std::cos(theta) * dt;
        mu_bar_(1) = mu_(1) + v * std::sin(theta) * dt;
        mu_bar_(2) = correctAngle(mu_(2) + omega * dt);

        // compute Jacobian of the nonlinear motion model
        // this corresponds to jacobian_g(mu, u, dt) in the Python code
        computeJacobianG(u, dt);

        // predicted covariance
        // EKF difference compared to simple KF:
        // use local linearization with Jacobian G
        Sigma_bar_ = G_ * Sigma_ * G_.transpose() + R_;

        return mu_bar_;
    }

    // Compute Jacobian of the motion model
    Eigen::Matrix3d computeJacobianG(const Eigen::Vector2d & u, double dt)
    {
        // current orientation
        const double theta = mu_(2);

        // linear velocity
        const double v = u(0);

        // Jacobian of:
        // x'     = x + v * cos(theta) * dt
        // y'     = y + v * sin(theta) * dt
        // theta' = theta + omega * dt
        G_ << 1.0, 0.0, -v * std::sin(theta) * dt,
              0.0, 1.0,  v * std::cos(theta) * dt,
              0.0, 0.0,  1.0;

        return G_;
    }

    Eigen::Matrix3d computeKalmanGain()
    {
        // compute Kalman Gain
        // measurement model is direct odometry pose:
        // z = [x, y, theta]
        Eigen::Matrix3d S = C_ * Sigma_bar_ * C_.transpose() + Q_;
        K_ = Sigma_bar_ * C_.transpose() * S.inverse();

        return K_;
    }

    // Correction step
    Eigen::Vector3d correct(const Eigen::Vector3d & z)
    {
        // innovation = difference between measurement and prediction
        Eigen::Vector3d innovation = z - C_ * mu_bar_;
        innovation(2) = correctAngle(innovation(2));

        // corrected state
        mu_ = mu_bar_ + K_ * innovation;
        mu_(2) = correctAngle(mu_(2));

        // corrected covariance
        Sigma_ = (Eigen::Matrix3d::Identity() - K_ * C_) * Sigma_bar_;

        return mu_;
    }

    // predict + correct
    Eigen::Vector3d update(
        const Eigen::Vector2d & u,
        const Eigen::Vector3d & z,
        double dt)
    {
        predict(u, dt);
        computeKalmanGain();
        correct(z);

        return mu_;
    }

    // getter
    const Eigen::Vector3d & state() const
    {
        return mu_;
    }

    const Eigen::Vector3d & predictedState() const
    {
        return mu_bar_;
    }

    const Eigen::Matrix3d & covariance() const
    {
        return Sigma_;
    }

    const Eigen::Matrix3d & predictedCovariance() const
    {
        return Sigma_bar_;
    }

    const Eigen::Matrix3d & jacobianG() const
    {
        return G_;
    }

    const Eigen::Matrix3d & kalmanGain() const
    {
        return K_;
    }

private:
    static double correctAngle(double angle)
    {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    // state
    Eigen::Vector3d mu_;

    // predicted state
    Eigen::Vector3d mu_bar_ = Eigen::Vector3d::Zero();

    // covariance
    Eigen::Matrix3d Sigma_;

    // predicted covariance
    Eigen::Matrix3d Sigma_bar_ = Eigen::Matrix3d::Zero();

    // measurement model matrix
    Eigen::Matrix3d C_;

    // Jacobian of nonlinear motion model
    Eigen::Matrix3d G_;

    // noise
    Eigen::Matrix3d R_;
    Eigen::Matrix3d Q_;

    // Kalman Gain
    Eigen::Matrix3d K_;
};