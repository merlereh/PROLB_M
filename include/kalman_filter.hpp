#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>

// State: [x, y, theta, vx, vy, theta_dot]
// Control input: u = [v, omega]  (from /cmd_vel)
// Odom measurement:     z_odom = [x, y, theta]  (absolute pose, 3x1)
// Landmark measurement: z_lm   = [r, phi]        (range + bearing, 2x1)
//
// KF treats everything as linear (as agreed with Prof).
// Landmark correction uses the nonlinear h() and H_lm but is applied
// as a standard KF correction step (no Jacobian — linear approximation).

using Vector6d   = Eigen::Matrix<double, 6, 1>;
using Matrix6d   = Eigen::Matrix<double, 6, 6>;
using Matrix3x6d = Eigen::Matrix<double, 3, 6>;
using Matrix6x3d = Eigen::Matrix<double, 6, 3>;
using Matrix2x6d = Eigen::Matrix<double, 2, 6>;
using Matrix6x2d = Eigen::Matrix<double, 6, 2>;

class KalmanFilter
{
public:
    KalmanFilter()
    {
        mu_ = Vector6d::Zero();

        Sigma_ = Matrix6d::Zero();
        Sigma_(0, 0) = 0.8;
        Sigma_(1, 1) = 0.8;
        Sigma_(2, 2) = 0.1;
        Sigma_(3, 3) = 0.5;
        Sigma_(4, 4) = 0.8;
        Sigma_(5, 5) = 0.2;

        // Odom measurement model C (3x6)
        C_ = Matrix3x6d::Zero();
        C_(0, 0) = 1.0;
        C_(1, 1) = 1.0;
        C_(2, 2) = 1.0;

        // Process noise R (6x6)
        R_ = Matrix6d::Zero();
        R_(0, 0) = 0.3;
        R_(1, 1) = 0.5;
        R_(2, 2) = 0.2;
        R_(3, 3) = 0.1;
        R_(4, 4) = 0.5;
        R_(5, 5) = 0.5;

        // Odom measurement noise Q_odom (3x3)
        Q_odom_ = Eigen::Matrix3d::Zero();
        Q_odom_(0, 0) = 0.5;
        Q_odom_(1, 1) = 0.5;
        Q_odom_(2, 2) = 0.2;

        // Landmark measurement noise Q_lm (2x2)
        Q_lm_ = Eigen::Matrix2d::Zero();
        Q_lm_(0, 0) = 0.1;
        Q_lm_(1, 1) = 0.02;

        K_odom_ = Matrix6x3d::Zero();
        K_lm_   = Matrix6x2d::Zero();
    }

    ~KalmanFilter() = default;

    void setState(const Eigen::Vector3d & pose)
    {
        mu_(0) = pose(0);
        mu_(1) = pose(1);
        mu_(2) = correctAngle(pose(2));
        mu_(3) = 0.0;
        mu_(4) = 0.0;
        mu_(5) = 0.0;
    }

    // Prediction step (linear)
    Vector6d predict(const Eigen::Vector2d & u, double dt)
    {
        const double theta = mu_(2);
        const double v     = u(0);
        const double omega = u(1);

        Matrix6d A = Matrix6d::Identity();
        A(0, 3) = dt;
        A(1, 4) = dt;
        A(2, 5) = dt;

        mu_bar_ = A * mu_;
        mu_bar_(3) = v * std::cos(theta);
        mu_bar_(4) = v * std::sin(theta);
        mu_bar_(5) = omega;
        mu_bar_(2) = correctAngle(mu_bar_(2));

        Sigma_bar_ = A * Sigma_ * A.transpose() + R_;

        return mu_bar_;
    }

    // Odom correction
    Vector6d correctOdom(const Eigen::Vector3d & z_odom)
    {
        Eigen::Matrix3d S = C_ * Sigma_bar_ * C_.transpose() + Q_odom_;
        K_odom_ = Sigma_bar_ * C_.transpose() * S.inverse();

        Eigen::Vector3d innovation = z_odom - C_ * mu_bar_;
        innovation(2) = correctAngle(innovation(2));

        mu_ = mu_bar_ + K_odom_ * innovation;
        mu_(2) = correctAngle(mu_(2));

        Sigma_ = (Matrix6d::Identity() - K_odom_ * C_) * Sigma_bar_;

        return mu_;
    }

    // Landmark correction
    // z_lm = [r, phi] measured by LiDAR
    // landmark = known (lx, ly) in map frame
    Vector6d correctLandmark(
        const Eigen::Vector2d & z_lm,
        const Eigen::Vector2d & landmark)
    {
        const double x     = mu_(0);
        const double y     = mu_(1);
        const double theta = mu_(2);
        const double lx    = landmark(0);
        const double ly    = landmark(1);

        const double dx  = lx - x;
        const double dy  = ly - y;
        const double q   = dx * dx + dy * dy;
        const double r   = std::sqrt(q);
        const double phi = correctAngle(std::atan2(dy, dx) - theta);

        Eigen::Vector2d z_hat;
        z_hat(0) = r;
        z_hat(1) = phi;

        // H_lm (2x6) — linear approximation (KF)
        Matrix2x6d H_lm = Matrix2x6d::Zero();
        H_lm(0, 0) = -dx / r;
        H_lm(0, 1) = -dy / r;
        H_lm(1, 0) =  dy / q;
        H_lm(1, 1) = -dx / q;
        H_lm(1, 2) = -1.0;

        Eigen::Matrix2d S = H_lm * Sigma_ * H_lm.transpose() + Q_lm_;
        K_lm_ = Sigma_ * H_lm.transpose() * S.inverse();

        Eigen::Vector2d innovation = z_lm - z_hat;
        innovation(1) = correctAngle(innovation(1));

        mu_ = mu_ + K_lm_ * innovation;
        mu_(2) = correctAngle(mu_(2));

        Sigma_ = (Matrix6d::Identity() - K_lm_ * H_lm) * Sigma_;

        return mu_;
    }

    Vector6d update(
        const Eigen::Vector2d & u,
        const Eigen::Vector3d & z_odom,
        double dt)
    {
        predict(u, dt);
        correctOdom(z_odom);
        return mu_;
    }

    const Vector6d   & state()               const { return mu_; }
    const Vector6d   & predictedState()      const { return mu_bar_; }
    const Matrix6d   & covariance()          const { return Sigma_; }
    const Matrix6d   & predictedCovariance() const { return Sigma_bar_; }
    const Matrix6x3d & kalmanGainOdom()      const { return K_odom_; }
    const Matrix6x2d & kalmanGainLandmark()  const { return K_lm_; }

private:
    static double correctAngle(double angle)
    {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    Vector6d mu_;
    Vector6d mu_bar_ = Vector6d::Zero();

    Matrix6d Sigma_;
    Matrix6d Sigma_bar_ = Matrix6d::Zero();

    Matrix3x6d      C_;
    Matrix6d        R_;
    Eigen::Matrix3d Q_odom_;
    Eigen::Matrix2d Q_lm_;

    Matrix6x3d K_odom_;
    Matrix6x2d K_lm_;
};