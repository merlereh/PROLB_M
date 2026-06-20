#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>

// ============================================================================
// Kalman Filter  —  Notation nach Vorlesungsfolien (Thrun 2006)
//
// State:   x = [x, y, theta, vx, vy, omega]   (6x1)
// Control: u = [v, omega]                      (2x1)
//
// Correction Full (Odom):
//   z = [x_map, y_map, theta_map]
//   x/y/theta aus odom.pose  (nach Offset-Transform in Map-Frame)
//
//   h(x) = [x, y, theta]  →  linear  →  C = [I3 | 0]  (3x6)
//
// Correction Landmark:
//   z_lm = [r, phi]  —  C_lm bleibt LINEAR
// ============================================================================

using Vector6d   = Eigen::Matrix<double, 6, 1>;
using Matrix6d   = Eigen::Matrix<double, 6, 6>;
using Matrix6x2d = Eigen::Matrix<double, 6, 2>;
using Matrix2x6d = Eigen::Matrix<double, 2, 6>;
using Matrix3x6d = Eigen::Matrix<double, 3, 6>;
using Matrix6x3d = Eigen::Matrix<double, 6, 3>;
using Matrix3d   = Eigen::Matrix<double, 3, 3>;

class KalmanFilter
{
public:
    KalmanFilter()
    {
        mu_ = Vector6d::Zero();

        // Initiale Kovarianz Sigma (6x6) — Diagonalmatrix
        //
        //         x     y   theta   vx     vy   omega
        //   x  [ 0.1   0     0      0      0     0   ]
        //   y  [  0   0.1    0      0      0     0   ]
        // theta [  0    0   0.02    0      0     0   ]
        //  vx  [  0    0     0     0.5     0     0   ]
        //  vy  [  0    0     0      0     0.5    0   ]
        // omega [  0    0     0      0      0    0.02 ]
        Sigma_ = Matrix6d::Zero();
        Sigma_(0,0) = 0.1;
        Sigma_(1,1) = 0.1;
        Sigma_(2,2) = 0.05;
        Sigma_(3,3) = 0.1;
        Sigma_(4,4) = 0.1;
        Sigma_(5,5) = 0.05;

        mu_bar_    = Vector6d::Zero();
        Sigma_bar_ = Matrix6d::Zero();

        // R — Prozessrauschen (6x6) — wird per setNoiseParams befüllt
        //
        //         x     y   theta   vx    vy   omega
        //   x  [ r_x    0     0      0     0     0   ]
        //   y  [  0    r_y    0      0     0     0   ]
        // theta [  0     0  r_theta  0     0     0   ]
        //  vx  [  0     0     0    r_vx    0     0   ]
        //  vy  [  0     0     0      0   r_vy    0   ]
        // omega [  0     0     0      0     0  r_omega]
        R_ = Matrix6d::Zero();

        // Q_full — Messrauschen Full Correction (3x3) — wird per setNoiseParams befüllt
        //
        //         x       y     theta
        //   x  [ q_x      0      0   ]
        //   y  [  0      q_y     0   ]
        // theta [  0       0   q_theta]
        Q_full_ = Matrix3d::Zero();

        // Q_lm — Messrauschen Landmark (2x2) — wird per setNoiseParams befüllt
        //
        //       r          phi
        //   r  [ q_lm_r     0     ]
        //  phi [    0     q_lm_phi ]
        Q_lm_ = Eigen::Matrix2d::Zero();

        K_full_ = Matrix6x3d::Zero();
        K_lm_   = Matrix6x2d::Zero();
    }

    ~KalmanFilter() = default;

    void setNoiseParams(
        double r_x,     double r_y,     double r_theta,
        double r_vx,    double r_vy,    double r_omega,
        double q_x,     double q_y,     double q_theta,
        double q_lm_r,  double q_lm_phi)
    {
        // R — Prozessrauschen
        R_(0,0) = r_x;    R_(1,1) = r_y;    R_(2,2) = r_theta;
        R_(3,3) = r_vx;   R_(4,4) = r_vy;   R_(5,5) = r_omega;

        // Q — Messrauschen Full (nur x, y, theta — aus odom.pose)
        Q_full_(0,0) = q_x;     Q_full_(1,1) = q_y;     Q_full_(2,2) = q_theta;

        // Q — Messrauschen Landmark
        Q_lm_(0,0) = q_lm_r;   Q_lm_(1,1) = q_lm_phi;
    }

    void setState(const Eigen::Vector3d & pose)
    {
        mu_(0) = pose(0); mu_(1) = pose(1); mu_(2) = correctAngle(pose(2));
        mu_(3) = 0.0;     mu_(4) = 0.0;     mu_(5) = 0.0;
        mu_bar_ = mu_;    Sigma_bar_ = Sigma_;
    }

    // -----------------------------------------------------------------------
    // PREDICTION
    // -----------------------------------------------------------------------
    Vector6d predict(const Eigen::Vector2d & u, double dt)
    {
        const double v = u(0), omega = u(1), theta = mu_(2);

        // =====================
        // PREDICT
        // =====================

        // Step 1: Predict mean  [Line 2: mu_bar = A * mu]
        mu_bar_(0) = mu_(0) + v * std::cos(theta) * dt;
        mu_bar_(1) = mu_(1) + v * std::sin(theta) * dt;
        mu_bar_(2) = correctAngle(mu_(2) + omega * dt);
        mu_bar_(3) = v * std::cos(theta);
        mu_bar_(4) = v * std::sin(theta);
        mu_bar_(5) = omega;

        // A — Zustandsübergangsmatrix (6x6)
        //
        //         x    y   theta   vx   vy  omega
        //   x  [  1    0     0     dt    0    0  ]
        //   y  [  0    1     0      0   dt    0  ]
        // theta [  0    0     1      0    0   dt  ]
        //  vx  [  0    0     0      0    0    0  ]
        //  vy  [  0    0     0      0    0    0  ]
        // omega [  0    0     0      0    0    0  ]
        A_ = Matrix6d::Zero();
        A_(0,0) = 1.0;   A_(0,3) = dt;
        A_(1,1) = 1.0;   A_(1,4) = dt;
        A_(2,2) = 1.0;   A_(2,5) = dt;

        // Step 2: Predict covariance  [Line 3: Sigma_bar = A * Sigma * A^T + R]
        Sigma_bar_ = A_ * Sigma_ * A_.transpose() + R_;

        return mu_bar_;
    }

    // -----------------------------------------------------------------------
    // CORRECTION — Full  z = [x_map, y_map, theta_map]
    //
    // Aus odom:
    //   x_map/y_map/theta_map  = odom.pose  (nach Offset-Transform)
    //
    // h(x) = [x, y, theta]  →  C = [I3 | 0]  (3x6)  →  linear, kein Jacobian nötig
    // -----------------------------------------------------------------------
    Vector6d correctFull(const Eigen::Vector3d & z)
    {
        // C — Messmatrix (3x6) = [I3 | 0]
        //
        //         x    y   theta   vx   vy  omega
        //   x  [  1    0     0      0    0    0  ]
        //   y  [  0    1     0      0    0    0  ]
        // theta [  0    0     1      0    0    0  ]
        Matrix3x6d C = Matrix3x6d::Zero();
        C(0,0) = 1.0;
        C(1,1) = 1.0;
        C(2,2) = 1.0;

        // =====================
        // KALMAN-GAIN
        // =====================

        // Step 3: Compute Kalman Gain  [Line 4: K = Sigma_bar * C^T * (C * Sigma_bar * C^T + Q)^-1]
        Matrix3d S = C * Sigma_bar_ * C.transpose() + Q_full_;
        K_full_ = Sigma_bar_ * C.transpose() * S.inverse();

        // =====================
        // CORRECT
        // =====================

        // Step 4: Correct mean  [Line 5: mu = mu_bar + K * (z - C * mu_bar)]
        Eigen::Vector3d innovation = z - C * mu_bar_;
        innovation(2) = correctAngle(innovation(2));
        mu_ = mu_bar_ + K_full_ * innovation;
        mu_(2) = correctAngle(mu_(2));

        // Step 5: Correct covariance  [Line 6: Sigma = (I - K * C) * Sigma_bar]
        Sigma_ = (Matrix6d::Identity() - K_full_ * C) * Sigma_bar_;

        return mu_;
    }

    // -----------------------------------------------------------------------
    // CORRECTION — Landmark  z_lm = [r, phi]
    // -----------------------------------------------------------------------
    Vector6d correctLandmark(
        const Eigen::Vector2d & z_lm,
        const Eigen::Vector2d & landmark)
    {
        const double x = mu_(0), y = mu_(1), theta = mu_(2);
        const double lx = landmark(0), ly = landmark(1);
        const double dx = lx - x, dy = ly - y;
        const double r = std::sqrt(dx*dx + dy*dy);
        if (r < 1e-6) return mu_;

        // h(mu) — erwartete Messung
        Eigen::Vector2d z_hat;
        z_hat(0) = r;
        z_hat(1) = correctAngle(std::atan2(dy, dx) - theta);

        // C_lm — Messmatrix LINEAR (2x6) — KEIN Jacobian
        //
        //        x       y    theta   vx   vy  omega
        //   r  [-dx/r  -dy/r    0      0    0    0  ]
        //  phi [   0      0    -1      0    0    0  ]
        Matrix2x6d C_lm = Matrix2x6d::Zero();
        C_lm(0,0) = -dx/r;
        C_lm(0,1) = -dy/r;
        C_lm(1,2) = -1.0;

        // =====================
        // KALMAN-GAIN
        // =====================

        // Step 3: K = Sigma * C^T * (C * Sigma * C^T + Q)^-1
        Eigen::Matrix2d S = C_lm * Sigma_ * C_lm.transpose() + Q_lm_;
        K_lm_ = Sigma_ * C_lm.transpose() * S.inverse();

        // =====================
        // CORRECT
        // =====================

        // Step 4: mu = mu + K * (z - h(mu))
        Eigen::Vector2d innovation = z_lm - z_hat;
        innovation(1) = correctAngle(innovation(1));
        mu_ = mu_ + K_lm_ * innovation;
        mu_(2) = correctAngle(mu_(2));

        // Step 5: Sigma = (I - K * C) * Sigma
        Sigma_ = (Matrix6d::Identity() - K_lm_ * C_lm) * Sigma_;

        return mu_;
    }

    const Vector6d & state()               const { return mu_; }
    const Vector6d & predictedState()      const { return mu_bar_; }
    const Matrix6d & covariance()          const { return Sigma_; }
    const Matrix6d & predictedCovariance() const { return Sigma_bar_; }
    const Matrix6x3d & kalmanGainFull()    const { return K_full_; }
    const Matrix6x2d & kalmanGainLandmark() const { return K_lm_; }

private:
    static double correctAngle(double a)
    { return std::atan2(std::sin(a), std::cos(a)); }

    Vector6d mu_;
    Vector6d mu_bar_;
    Matrix6d Sigma_;
    Matrix6d Sigma_bar_;

    Matrix6d        A_;
    Matrix6d        R_;
    Matrix3d        Q_full_;
    Eigen::Matrix2d Q_lm_;

    Matrix6x3d K_full_;
    Matrix6x2d K_lm_;
};