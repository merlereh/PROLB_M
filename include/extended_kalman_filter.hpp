#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>

// State: [x, y, theta, vx, vy, theta_dot]
// Control input: u = [v, omega]  (from /cmd_vel)
// Odom measurement:     z_odom = [x, y, theta]  (3x1, from /odom)
// Landmark measurement: z_lm   = [r, phi]       (2x1, from /scan)

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
        Sigma_(0, 0) = 0.1;   
        Sigma_(1, 1) = 0.1;   
        Sigma_(2, 2) = 0.02;   
        Sigma_(3, 3) = 0.1;
        Sigma_(4, 4) = 0.1;
        Sigma_(5, 5) = 0.02;

        mu_bar_    = Vector6d::Zero();
        Sigma_bar_ = Matrix6d::Zero();

        R_      = Matrix6d::Zero();
        Q_odom_ = Eigen::Matrix3d::Zero();
        Q_lm_   = Eigen::Matrix2d::Zero();

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
    // Prediction step — nonlinear g() + Jacobian G
    // -------------------------------------------------------------------------
    Vector6d predict(const Eigen::Vector2d & u, double dt)
    {
        const double theta = mu_(2);
        const double v     = u(0);
        const double omega = u(1);

        mu_bar_(0) = mu_(0) + v * std::cos(theta) * dt;
        mu_bar_(1) = mu_(1) + v * std::sin(theta) * dt;
        mu_bar_(2) = correctAngle(mu_(2) + omega * dt);
        mu_bar_(3) = v * std::cos(theta);
        mu_bar_(4) = v * std::sin(theta);
        mu_bar_(5) = omega;

        G_ = Matrix6d::Zero();

        G_(0, 0) = 1.0;
        G_(1, 1) = 1.0;
        G_(2, 2) = 1.0;

        G_(0, 2) = -v * std::sin(theta) * dt;
        G_(1, 2) =  v * std::cos(theta) * dt;

        // x hängt von vx ab: ∂x'/∂vx = dt
        // y hängt von vy ab: ∂y'/∂vy = dt
        // theta hängt von omega ab: ∂theta'/∂omega = dt
        G_(0, 3) = dt;
        G_(1, 4) = dt;
        G_(2, 5) = dt;

        G_(3, 2) = -v * std::sin(theta);
        G_(4, 2) =  v * std::cos(theta);

        Sigma_bar_ = G_ * Sigma_ * G_.transpose() + R_;

        return mu_bar_;
    }

    // -------------------------------------------------------------------------
    // Odom correction — 3D: corrects using [x, y, theta] from /odom
    //   H     = [I3 | 0]  (3x6, selects position/heading states)
    //   K     = Sigma_bar * H^T * (H * Sigma_bar * H^T + Q_odom)^-1
    //   mu    = mu_bar + K * (z_odom - H * mu_bar)
    //   Sigma = (I - K * H) * Sigma_bar
    // -------------------------------------------------------------------------
    Vector6d correctOdom(const Eigen::Vector3d & z_odom)
    {
        // H selects the first 3 states [x, y, theta]
        Matrix3x6d H = Matrix3x6d::Zero();
        H(0, 0) = 1.0;
        H(1, 1) = 1.0;
        H(2, 2) = 1.0;

        Eigen::Matrix3d S = H * Sigma_bar_ * H.transpose() + Q_odom_;
        K_odom_ = Sigma_bar_ * H.transpose() * S.inverse();

        Eigen::Vector3d innovation = z_odom - H * mu_bar_;
        innovation(2) = correctAngle(innovation(2));

        mu_ = mu_bar_ + K_odom_ * innovation;
        mu_(2) = correctAngle(mu_(2));

        Sigma_ = (Matrix6d::Identity() - K_odom_ * H) * Sigma_bar_;

        return mu_;
    }

    // -------------------------------------------------------------------------
    // Landmark correction — uses mu_ / Sigma_ (post-odom)
    // -------------------------------------------------------------------------
    Vector6d correctLandmark(
        const Eigen::Vector2d & z_lm,
        const Eigen::Vector2d & landmark)
    {
        const double x     = mu_(0);
        const double y     = mu_(1);
        const double theta = mu_(2);
        const double lx    = landmark(0);
        const double ly    = landmark(1);

        const double dx = lx - x;
        const double dy = ly - y;
        const double q  = dx * dx + dy * dy;
        const double r  = std::sqrt(q);

        if (r < 1e-6) { return mu_; }

        Eigen::Vector2d z_hat;
        z_hat(0) = r;
        z_hat(1) = correctAngle(std::atan2(dy, dx) - theta);

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

    // Convenience: predict + odom-correct in one call
    Vector6d update(
        const Eigen::Vector2d & u,
        const Eigen::Vector3d & z_odom,
        double dt)
    {
        predict(u, dt);
        correctOdom(z_odom);
        return mu_;
    }

    const Vector6d & state()               const { return mu_; }
    const Vector6d & predictedState()      const { return mu_bar_; }
    const Matrix6d & covariance()          const { return Sigma_; }
    const Matrix6d & predictedCovariance() const { return Sigma_bar_; }
    const Matrix6d & jacobianG()           const { return G_; }
    const Matrix6x3d & kalmanGainOdom()    const { return K_odom_; }
    const Matrix6x2d & kalmanGainLandmark() const { return K_lm_; }

private:
    static double correctAngle(double angle)
    {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    Vector6d mu_;
    Vector6d mu_bar_;

    Matrix6d Sigma_;
    Matrix6d Sigma_bar_;

    Matrix6d        G_;
    Matrix6d        R_;
    Eigen::Matrix3d Q_odom_;
    Eigen::Matrix2d Q_lm_;

    Matrix6x3d K_odom_;
    Matrix6x2d K_lm_;
};