#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>

// State: [x, y, theta, vx, vy, theta_dot]
// Control input: u = [v, omega]  (from /cmd_vel)
// Odom measurement:     z_odom = [x, y, theta]  (absolute pose, 3x1)
// Landmark measurement: z_lm   = [r, phi]        (range + bearing, 2x1)
//
// KF treats everything as linear (as agreed with Prof).
// Landmark correction uses the nonlinear h() and linearised H_lm,
// but crucially operates on mu_bar_ / Sigma_bar_ (predicted state),
// exactly as the Thrun EKF Localization algorithm requires:
//   lines 8-12: delta, q, z_hat, H  all computed from mu_bar_
//   line  14:   mu_t  = mu_bar_  + sum K_i (z_i - z_hat_i)
//   line  15:   Sigma_t = (I - sum K_i H_i) Sigma_bar_

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

        mu_bar_    = Vector6d::Zero();
        Sigma_bar_ = Matrix6d::Zero();

        // Odom measurement model C (3x6): maps state → [x, y, theta]
        C_ = Matrix3x6d::Zero();
        C_(0, 0) = 1.0;
        C_(1, 1) = 1.0;
        C_(2, 2) = 1.0;

        // Process noise R (6x6)
        R_ = Matrix6d::Zero();
        

        // Odom measurement noise Q_odom (3x3)
        Q_odom_ = Eigen::Matrix3d::Zero();
        

        // Landmark measurement noise Q_lm (2x2)
        // [r_noise, phi_noise]
        Q_lm_ = Eigen::Matrix2d::Zero();
        

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
        // Initialise bar-quantities to avoid stale values
        mu_bar_    = mu_;
        Sigma_bar_ = Sigma_;
    }

      void setNoiseParams(
        double r_x,  double r_y,  double r_theta,
        double r_vx, double r_vy, double r_omega,
        double q_odom_x, double q_odom_y, double q_odom_theta,
        double q_lm_r,   double q_lm_phi)
    {
        R_(0, 0) = r_x;
        R_(1, 1) = r_y;
        R_(2, 2) = r_theta;
        R_(3, 3) = r_vx;
        R_(4, 4) = r_vy;
        R_(5, 5) = r_omega;
 
        Q_odom_(0, 0) = q_odom_x;
        Q_odom_(1, 1) = q_odom_y;
        Q_odom_(2, 2) = q_odom_theta;
 
        Q_lm_(0, 0) = q_lm_r;
        Q_lm_(1, 1) = q_lm_phi;
    }

    // -------------------------------------------------------------------------
    // Prediction step (linear KF)
    //   mu_bar    = A * mu + B * u
    //   Sigma_bar = A * Sigma * A^T + R
    // -------------------------------------------------------------------------
    Vector6d predict(const Eigen::Vector2d & u, double dt)
    {
        const double v     = u(0);
        const double omega = u(1);
        const double theta = mu_(2);

        // Motion model: integrate velocity into pose
        mu_bar_(0) = mu_(0) + mu_(3) * dt;
        mu_bar_(1) = mu_(1) + mu_(4) * dt;
        mu_bar_(2) = correctAngle(mu_(2) + mu_(5) * dt);
        // Velocity update from cmd_vel
        mu_bar_(3) = v * std::cos(theta);
        mu_bar_(4) = v * std::sin(theta);
        mu_bar_(5) = omega;

        // State-transition matrix A (linear KF — identity + kinematics)
        Matrix6d A = Matrix6d::Identity();
        A(0, 3) = dt;
        A(1, 4) = dt;
        A(2, 5) = dt;

        Sigma_bar_ = A * Sigma_ * A.transpose() + R_;

        return mu_bar_;
    }

    // -------------------------------------------------------------------------
    // Odom correction (standard KF correction on mu_bar_ / Sigma_bar_)
    //   K     = Sigma_bar * C^T * (C * Sigma_bar * C^T + Q_odom)^-1
    //   mu    = mu_bar + K * (z_odom - C * mu_bar)
    //   Sigma = (I - K * C) * Sigma_bar
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Landmark correction — Thrun EKF Localization algorithm, lines 8–15
    //
    // IMPORTANT: everything is computed from mu_bar_ / Sigma_bar_,
    // NOT from mu_ / Sigma_. This is the fix compared to the old code.
    //
    //   delta = (lx - mu_bar_x,  ly - mu_bar_y)        [L.8]
    //   q     = delta^T * delta                          [L.9]
    //   z_hat = (sqrt(q),  atan2(dy,dx) - mu_bar_theta) [L.10]
    //   H     = 1/q * (...)                              [L.11]
    //   K     = Sigma_bar * H^T * (H*Sigma_bar*H^T + Q)^-1  [L.12]
    //   mu    = mu_bar + K * (z - z_hat)                [L.14]
    //   Sigma = (I - K*H) * Sigma_bar                   [L.15]
    // -------------------------------------------------------------------------
    Vector6d correctLandmark(
        const Eigen::Vector2d & z_lm,
        const Eigen::Vector2d & landmark)
    {
        // Use PREDICTED state (mu_bar_), not the corrected state (mu_)
        const double x     = mu_bar_(0);
        const double y     = mu_bar_(1);
        const double theta = mu_bar_(2);
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

        // L.11  Jacobian H_lm (2x6)
        //   dr/dx   = -dx/r,  dr/dy   = -dy/r,  dr/dtheta = 0
        //   dphi/dx =  dy/q,  dphi/dy = -dx/q,  dphi/dtheta = -1
        //   columns 3,4,5 (velocities) = 0
        Matrix2x6d H_lm = Matrix2x6d::Zero();
        H_lm(0, 0) = -dx / r;
        H_lm(0, 1) = -dy / r;
        H_lm(1, 0) =  dy / q;
        H_lm(1, 1) = -dx / q;
        H_lm(1, 2) = -1.0;

        // L.12  Kalman gain — uses Sigma_bar_
        Eigen::Matrix2d S = H_lm * Sigma_bar_ * H_lm.transpose() + Q_lm_;
        K_lm_ = Sigma_bar_ * H_lm.transpose() * S.inverse();

        // Innovation (angle-wrapped)
        Eigen::Vector2d innovation = z_lm - z_hat;
        innovation(1) = correctAngle(innovation(1));

        // L.14  mu update — starts from mu_bar_
        mu_ = mu_bar_ + K_lm_ * innovation;
        mu_(2) = correctAngle(mu_(2));

        // L.15  Sigma update — starts from Sigma_bar_
        Sigma_ = (Matrix6d::Identity() - K_lm_ * H_lm) * Sigma_bar_;

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

    Matrix3x6d      C_;
    Matrix6d        R_;
    Eigen::Matrix3d Q_odom_;
    Eigen::Matrix2d Q_lm_;

    Matrix6x3d K_odom_;
    Matrix6x2d K_lm_;
};