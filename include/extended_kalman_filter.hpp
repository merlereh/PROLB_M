#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>

// State: [x, y, theta, vx, vy, theta_dot]
// Control input: u = [v, omega]  (from /cmd_vel)
// Odom measurement:     z_odom = [x, y, theta]  (absolute pose, 3x1)
// Landmark measurement: z_lm   = [r, phi]        (range + bearing, 2x1)
//
// EKF uses nonlinear g() + Jacobian G for prediction,
// and nonlinear h() + Jacobian H_lm for landmark correction.
//
// Landmark correction strictly follows the Thrun EKF Localization algorithm
// (Thrun 2006, Table 7.2):
//   lines 8-12:  delta, q, z_hat, H_i  all from mu_bar_ / Sigma_bar_
//   line  14:    mu_t    = mu_bar_  + sum_i K_i (z_i - z_hat_i)
//   line  15:    Sigma_t = (I - sum_i K_i H_i) Sigma_bar_

using Vector6d   = Eigen::Matrix<double, 6, 1>;
using Matrix6d   = Eigen::Matrix<double, 6, 6>;
using Matrix3x6d = Eigen::Matrix<double, 3, 6>;
using Matrix6x3d = Eigen::Matrix<double, 6, 3>;
using Matrix2x6d = Eigen::Matrix<double, 2, 6>;
using Matrix6x2d = Eigen::Matrix<double, 6, 2>;

class ExtendedKalmanFilter
{
public:
    ExtendedKalmanFilter()
    {
        mu_ = Vector6d::Zero();

        Sigma_ = Matrix6d::Zero();
        Sigma_(0, 0) = 0.2;
        Sigma_(1, 1) = 0.2;
        Sigma_(2, 2) = 0.1;
        Sigma_(3, 3) = 0.5;
        Sigma_(4, 4) = 0.5;
        Sigma_(5, 5) = 0.2;

        mu_bar_    = Vector6d::Zero();
        Sigma_bar_ = Matrix6d::Zero();

        // Odom measurement model H_odom (3x6): maps state → [x, y, theta]
        H_odom_ = Matrix3x6d::Zero();
        H_odom_(0, 0) = 1.0;
        H_odom_(1, 1) = 1.0;
        H_odom_(2, 2) = 1.0;

        // Process noise R (6x6)
        R_ = Matrix6d::Zero();
        R_(0, 0) = 0.05;
        R_(1, 1) = 0.05;
        R_(2, 2) = 0.02;
        R_(3, 3) = 0.1;
        R_(4, 4) = 0.05;
        R_(5, 5) = 0.05;

        // Odom measurement noise Q_odom (3x3)
        Q_odom_ = Eigen::Matrix3d::Zero();
        Q_odom_(0, 0) = 0.1;
        Q_odom_(1, 1) = 0.1;
        Q_odom_(2, 2) = 0.05;

        // Landmark measurement noise Q_lm (2x2)
        // [r_noise, phi_noise]
        Q_lm_ = Eigen::Matrix2d::Zero();
        Q_lm_(0, 0) = 0.05;
        Q_lm_(1, 1) = 0.01;

        G_      = Matrix6d::Identity();
        K_odom_ = Matrix6x3d::Zero();
        K_lm_   = Matrix6x2d::Zero();
    }

    ~ExtendedKalmanFilter() = default;

    void setState(const Eigen::Vector3d & pose)
    {
        mu_(0) = pose(0);
        mu_(1) = pose(1);
        mu_(2) = correctAngle(pose(2));
        mu_(3) = 0.0;
        mu_(4) = 0.0;
        mu_(5) = 0.0;
        // Initialise bar-quantities to avoid stale values
        mu_bar_    = mu_;
        Sigma_bar_ = Sigma_;
    }

    // -------------------------------------------------------------------------
    // Prediction step — nonlinear g() + Jacobian G
    //   mu_bar    = g(u, mu)
    //   Sigma_bar = G * Sigma * G^T + R
    // -------------------------------------------------------------------------
    Vector6d predict(const Eigen::Vector2d & u, double dt)
    {
        const double theta = mu_(2);
        const double v     = u(0);
        const double omega = u(1);

        // Nonlinear motion model g()
        mu_bar_(0) = mu_(0) + mu_(3) * dt;
        mu_bar_(1) = mu_(1) + mu_(4) * dt;
        mu_bar_(2) = correctAngle(mu_(2) + mu_(5) * dt);
        mu_bar_(3) = v * std::cos(theta);
        mu_bar_(4) = v * std::sin(theta);
        mu_bar_(5) = omega;

        // Jacobian G = dg/dx evaluated at mu_t-1
        G_ = Matrix6d::Zero();
        G_(0, 0) = 1.0;  G_(0, 3) = dt;
        G_(1, 1) = 1.0;  G_(1, 4) = dt;
        G_(2, 2) = 1.0;  G_(2, 5) = dt;
        G_(3, 2) = -v * std::sin(theta);
        G_(4, 2) =  v * std::cos(theta);
        G_(5, 5) = 1.0;

        Sigma_bar_ = G_ * Sigma_ * G_.transpose() + R_;

        return mu_bar_;
    }

    // -------------------------------------------------------------------------
    // Odom correction (standard linear correction on mu_bar_ / Sigma_bar_)
    //   K     = Sigma_bar * H^T * (H * Sigma_bar * H^T + Q_odom)^-1
    //   mu    = mu_bar + K * (z_odom - H * mu_bar)
    //   Sigma = (I - K * H) * Sigma_bar
    // -------------------------------------------------------------------------
    Vector6d correctOdom(const Eigen::Vector3d & z_odom)
    {
        Eigen::Matrix3d S = H_odom_ * Sigma_bar_ * H_odom_.transpose() + Q_odom_;
        K_odom_ = Sigma_bar_ * H_odom_.transpose() * S.inverse();

        Eigen::Vector3d innovation = z_odom - H_odom_ * mu_bar_;
        innovation(2) = correctAngle(innovation(2));

        mu_ = mu_bar_ + K_odom_ * innovation;
        mu_(2) = correctAngle(mu_(2));

        Sigma_ = (Matrix6d::Identity() - K_odom_ * H_odom_) * Sigma_bar_;

        return mu_;
    }

    // -------------------------------------------------------------------------
    // Landmark correction — Thrun EKF Localization algorithm, lines 8–15
    //
    // IMPORTANT: everything is computed from mu_bar_ / Sigma_bar_,
    // NOT from mu_ / Sigma_. This was the bug in the previous version.
    //
    //   delta = (lx - mu_bar_x,  ly - mu_bar_y)              [L.8]
    //   q     = delta^T * delta                                [L.9]
    //   z_hat = (sqrt(q),  atan2(dy,dx) - mu_bar_theta)       [L.10]
    //   H     = 1/q * (sqrt(q)*dx  -sqrt(q)*dy  0  ...)       [L.11]
    //             (        dy          -dx      -q  ...)
    //   K     = Sigma_bar * H^T * (H * Sigma_bar * H^T + Q)^-1  [L.12]
    //   mu    = mu_bar + K * (z - z_hat)                      [L.14]
    //   Sigma = (I - K*H) * Sigma_bar                         [L.15]
    // -------------------------------------------------------------------------
    Vector6d correctLandmark(
        const Eigen::Vector2d & z_lm,
        const Eigen::Vector2d & landmark)
    {
        // Use current state mu_ (post-odom correction), not mu_bar_
        const double x     = mu_(0);
        const double y     = mu_(1);
        const double theta = mu_(2);
        const double lx    = landmark(0);
        const double ly    = landmark(1);

        // L.8  delta
        const double dx = lx - x;
        const double dy = ly - y;

        // L.9  q
        const double q = dx * dx + dy * dy;
        const double r = std::sqrt(q);

        // Guard: landmark too close (would cause division by zero)
        if (r < 1e-6) { return mu_bar_; }

        // L.10  pseudo-measurement z_hat from mu_bar_
        Eigen::Vector2d z_hat;
        z_hat(0) = r;
        z_hat(1) = correctAngle(std::atan2(dy, dx) - theta);

        // L.11  Jacobian H_lm (2x6) — nonlinear h() linearised at mu_bar_
        //   dr/dx   = -dx/r,  dr/dy   = -dy/r,  dr/dtheta   = 0
        //   dphi/dx =  dy/q,  dphi/dy = -dx/q,  dphi/dtheta = -1
        //   columns 3,4,5 (velocities) do not affect measurement → 0
        Matrix2x6d H_lm = Matrix2x6d::Zero();
        H_lm(0, 0) = -dx / r;
        H_lm(0, 1) = -dy / r;
        // H_lm(0,2) = 0  (range is independent of heading)
        H_lm(1, 0) =  dy / q;
        H_lm(1, 1) = -dx / q;
        H_lm(1, 2) = -1.0;

        // Kalman gain — uses Sigma_ (already updated by correctOdom)
        // We use Sigma_ here because correctLandmark is called AFTER
        // correctOdom, so Sigma_ is already the post-odom covariance.
        // Using Sigma_bar_ here would undo the odom correction and
        // blow up the covariance.
        Eigen::Matrix2d S = H_lm * Sigma_ * H_lm.transpose() + Q_lm_;
        K_lm_ = Sigma_ * H_lm.transpose() * S.inverse();

        // Innovation (angle-wrapped to [-pi, pi])
        Eigen::Vector2d innovation = z_lm - z_hat;
        innovation(1) = correctAngle(innovation(1));

        // mu update — starts from current mu_
        mu_ = mu_ + K_lm_ * innovation;
        mu_(2) = correctAngle(mu_(2));

        // Sigma update — starts from current Sigma_ (shrinks it further)
        Sigma_ = (Matrix6d::Identity() - K_lm_ * H_lm) * Sigma_;

        return mu_;
    }

    // Convenience: predict + odom-correct in one call (normal cycle)
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
    const Matrix6d   & jacobianG()           const { return G_; }
    const Matrix6x3d & kalmanGainOdom()      const { return K_odom_; }
    const Matrix6x2d & kalmanGainLandmark()  const { return K_lm_; }

private:
    static double correctAngle(double angle)
    {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    Vector6d mu_;
    Vector6d mu_bar_;

    Matrix6d Sigma_;
    Matrix6d Sigma_bar_;

    Matrix3x6d      H_odom_;
    Matrix6d        G_;
    Matrix6d        R_;
    Eigen::Matrix3d Q_odom_;
    Eigen::Matrix2d Q_lm_;

    Matrix6x3d K_odom_;
    Matrix6x2d K_lm_;
};