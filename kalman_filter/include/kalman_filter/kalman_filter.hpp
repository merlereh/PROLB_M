#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>

class KalmanFilter
{
public:
    // Constructor
    KalmanFilter()
    {
    // beginning state = [x, y, theta]
    mu_ << 0.0, 0.0, 0.0;

    // covariance in the beginning
    Sigma_ = Eigen::Matrix3d::Zero();
    Sigma_(0, 0) = 0.2;
    Sigma_(1, 1) = 0.2;
    Sigma_(2, 2) = 0.1;

    // linear model matrices
    A_ = Eigen::Matrix3d::Identity();
    B_ = Eigen::Matrix3d::Identity();
    C_ = Eigen::Matrix3d::Identity();

    // process noise
    R_ = Eigen::Matrix3d::Zero();
    R_(0, 0) = 0.05;
    R_(1, 1) = 0.05;
    R_(2, 2) = 0.02;

    // measurement noise
    Q_ = Eigen::Matrix3d::Zero();
    Q_(0, 0) = 0.01;
    Q_(1, 1) = 0.01;
    Q_(2, 2) = 0.02;

    K_ = Eigen::Matrix3d::Zero();
    }

    // Destructor
    ~KalmanFilter() = default;


    void setState(const Eigen::Vector3d & mu)
    {
        mu_ = mu;
        mu_(2) = correctAngle(mu_(2));
    }

    // Prediction step
    Eigen::Vector3d predict(const Eigen::Vector3d & u)
    {
        // predicted state
        mu_bar_ = A_ * mu_ + B_ * u;
        mu_bar_(2) = correctAngle(mu_bar_(2));

        // predicted covariance
        Sigma_bar_ = A_ * Sigma_ * A_.transpose() + R_;

        return mu_bar_;
    }


    Eigen::Matrix3d computeKalmanGain()
    {
        // compute Kalman Gain
        Eigen::Matrix3d S = C_ * Sigma_bar_ * C_.transpose() + Q_;
        K_ = Sigma_bar_ * C_.transpose() * S.inverse();

        return K_;
    }

    // Correction step
    Eigen::Vector3d correct(const Eigen::Vector3d & z)
    {
        Eigen::Vector3d innovation = z - C_ * mu_bar_;
        innovation(2) = correctAngle(innovation(2));

        mu_ = mu_bar_ + K_ * innovation;
        mu_(2) = correctAngle(mu_(2));

        Sigma_ = (Eigen::Matrix3d::Identity() - K_ * C_) * Sigma_bar_;

        return mu_;
    }

    // predict + correct 
    Eigen::Vector3d update(
        const Eigen::Vector3d & u,
        const Eigen::Vector3d & z)
    {
        predict(u);
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

    // matrices
    Eigen::Matrix3d A_;
    Eigen::Matrix3d B_;
    Eigen::Matrix3d C_;

    // noise
    Eigen::Matrix3d R_;
    Eigen::Matrix3d Q_;

    // Kalman Gain
    Eigen::Matrix3d K_;
};