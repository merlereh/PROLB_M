#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>

// ============================================================================
// Extended Kalman Filter  —  Notation nach Vorlesungsfolien (Thrun 2006)
//
// State:   x = [x, y, theta, vx, vy, omega]   (6x1)
// Control: u = [v, omega]                      (2x1)
//
// Correction Velocity (Odom-Twist):
//   z = [vx_world, vy_world, omega]
//   vx_world = v * cos(theta)   — nichtlinear in theta!
//   vy_world = v * sin(theta)   — nichtlinear in theta!
//   omega    = odom.twist.angular.z
//
//   h(x) ist nichtlinear → EKF braucht Jacobian H (3x6):
//
//         x    y      theta        vx   vy  omega
//  vx  [  0    0   -v*sin(θ)       1    0    0  ]
//  vy  [  0    0    v*cos(θ)       0    1    0  ]
// omega [  0    0       0           0    0    1  ]
//
// Correction Landmark:
//   z_lm = [r, phi]  — nichtlinear, voller Jacobian H (2x6)
// ============================================================================

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

        // Initiale Kovarianz Sigma (6x6) — Diagonalmatrix
        //
        //         x     y   theta   vx    vy   omega
        //   x  [ 0.5   0     0      0      0     0   ]
        //   y  [  0   0.5    0      0      0     0   ]
        // theta [  0    0    0.1    0      0     0   ]
        //  vx  [  0    0     0     0.5     0     0   ]
        //  vy  [  0    0     0      0     0.5    0   ]
        // omega [  0    0     0      0      0    0.1  ]
        Sigma_ = Matrix6d::Zero();
        Sigma_(0,0) = 0.5;
        Sigma_(1,1) = 0.5;
        Sigma_(2,2) = 0.1;
        Sigma_(3,3) = 0.5;
        Sigma_(4,4) = 0.5;
        Sigma_(5,5) = 0.1;

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

        // Q_vel — Messrauschen Velocity (3x3) — wird per setNoiseParams befüllt
        //
        //         vx        vy      omega
        //  vx  [ q_vx       0        0   ]
        //  vy  [   0      q_vy       0   ]
        // omega [   0        0    q_omega ]
        Q_vel_ = Eigen::Matrix3d::Zero();

        // Q_lm — Messrauschen Landmark (2x2) — wird per setNoiseParams befüllt
        //
        //       r          phi
        //   r  [ q_lm_r     0     ]
        //  phi [    0     q_lm_phi ]
        Q_lm_ = Eigen::Matrix2d::Zero();

        G_     = Matrix6d::Identity();
        K_vel_ = Matrix6x3d::Zero();
        K_lm_  = Matrix6x2d::Zero();
    }

    ~ExtendedKalmanFilter() = default;

    void setState(const Eigen::Vector3d & pose)
    {
        mu_(0) = pose(0); mu_(1) = pose(1); mu_(2) = correctAngle(pose(2));
        mu_(3) = 0.0;     mu_(4) = 0.0;     mu_(5) = 0.0;
        mu_bar_ = mu_;    Sigma_bar_ = Sigma_;
    }

    void setNoiseParams(
        double r_x,   double r_y,   double r_theta,
        double r_vx,  double r_vy,  double r_omega,
        double q_vx,  double q_vy,  double q_omega,
        double q_lm_r, double q_lm_phi)
    {
        // R — Prozessrauschen
        R_(0,0) = r_x;    R_(1,1) = r_y;    R_(2,2) = r_theta;
        R_(3,3) = r_vx;   R_(4,4) = r_vy;   R_(5,5) = r_omega;

        // Q — Messrauschen
        Q_vel_(0,0) = q_vx;   Q_vel_(1,1) = q_vy;   Q_vel_(2,2) = q_omega;
        Q_lm_(0,0)  = q_lm_r; Q_lm_(1,1)  = q_lm_phi;
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

        // Step 1: Predict mean  [Line 2: mu_bar = g(u, mu)]
        // nichtlineare Zustandsübergangsfunktion g(.)
        mu_bar_(0) = mu_(0) + v * std::cos(theta) * dt;
        mu_bar_(1) = mu_(1) + v * std::sin(theta) * dt;
        mu_bar_(2) = correctAngle(mu_(2) + omega * dt);
        mu_bar_(3) = v * std::cos(theta);
        mu_bar_(4) = v * std::sin(theta);
        mu_bar_(5) = omega;

        // G — Jacobi-Matrix von g nach x (6x6), ausgewertet an µ
        // Partielle Ableitungen der nichtlinearen Funktion g(.) nach den Zustandsvariablen
        //
        //         x    y   theta          vx   vy  omega
        //   x  [  1    0  -v*sin(θ)*dt   dt    0    0  ]
        //   y  [  0    1   v*cos(θ)*dt    0   dt    0  ]
        // theta [  0    0       1          0    0   dt  ]
        //  vx  [  0    0  -v*sin(θ)       1    0    0  ]
        //  vy  [  0    0   v*cos(θ)       0    1    0  ]
        // omega [  0    0       0          0    0    1  ]
        G_ = Matrix6d::Zero();
        G_(0,0) = 1.0;   G_(0,2) = -v * std::sin(theta) * dt;   G_(0,3) = dt;
        G_(1,1) = 1.0;   G_(1,2) =  v * std::cos(theta) * dt;   G_(1,4) = dt;
        G_(2,2) = 1.0;                                            G_(2,5) = dt;
        G_(3,2) = -v * std::sin(theta);                           G_(3,3) = 1.0;
        G_(4,2) =  v * std::cos(theta);                           G_(4,4) = 1.0;
        G_(5,5) = 1.0;

        // Step 2: Predict covariance  [Line 3: Sigma_bar = G * Sigma * G^T + R]
        Sigma_bar_ = G_ * Sigma_ * G_.transpose() + R_;

        return mu_bar_;
    }

    // -----------------------------------------------------------------------
    // CORRECTION — Velocity  z = [vx_world, vy_world, omega]
    //
    // Messung aus odom.twist:
    //   v     = odom.twist.linear.x         (Geschwindigkeit im Roboterframe)
    //   omega = odom.twist.angular.z
    //   vx_world = v * cos(theta_current)   (umgerechnet in Weltframe)
    //   vy_world = v * sin(theta_current)
    //
    // h(x) = [v*cos(theta), v*sin(theta), omega]  — nichtlinear in theta!
    // → EKF braucht Jacobian H
    // -----------------------------------------------------------------------
    Vector6d correctVelocity(const Eigen::Vector3d & z)
    {
        const double theta = mu_bar_(2);
        const double vx    = mu_bar_(3);
        const double vy    = mu_bar_(4);
        // v rekonstruiert aus aktuellem State (für Jacobian)
        const double v     = std::sqrt(vx*vx + vy*vy);

        // h(mu_bar) — erwartete Messung
        Eigen::Vector3d z_hat;
        z_hat(0) = v * std::cos(theta);   // erwartetes vx_world
        z_hat(1) = v * std::sin(theta);   // erwartetes vy_world
        z_hat(2) = mu_bar_(5);            // erwartetes omega

        // H — Jacobi-Matrix von h nach x (3x6), ausgewertet an µ_bar
        // Partielle Ableitungen der nichtlinearen Messfunktion h(.) nach den Zustandsvariablen
        //
        //         x    y      theta        vx   vy  omega
        //  vx  [  0    0   -v*sin(θ)       1    0    0  ]
        //  vy  [  0    0    v*cos(θ)       0    1    0  ]
        // omega [  0    0       0           0    0    1  ]
        Matrix3x6d H = Matrix3x6d::Zero();
        H(0,2) = -v * std::sin(theta);   // dvx/dtheta
        H(0,3) = 1.0;                    // dvx/dvx
        H(1,2) =  v * std::cos(theta);   // dvy/dtheta
        H(1,4) = 1.0;                    // dvy/dvy
        H(2,5) = 1.0;                    // domega/domega

        // =====================
        // KALMAN-GAIN
        // =====================

        // Step 3: Compute Kalman Gain  [Line 4: K = Sigma_bar * H^T * (H * Sigma_bar * H^T + Q)^-1]
        Eigen::Matrix3d S = H * Sigma_bar_ * H.transpose() + Q_vel_;
        K_vel_ = Sigma_bar_ * H.transpose() * S.inverse();

        // =====================
        // CORRECT
        // =====================

        // Step 4: Correct mean with measurement  [Line 5: mu = mu_bar + K * (z - h(mu_bar))]
        Eigen::Vector3d innovation = z - z_hat;
        mu_ = mu_bar_ + K_vel_ * innovation;
        mu_(2) = correctAngle(mu_(2));

        // Step 5: Correct covariance  [Line 6: Sigma = (I - K * H) * Sigma_bar]
        Sigma_ = (Matrix6d::Identity() - K_vel_ * H) * Sigma_bar_;

        return mu_;
    }

    // -----------------------------------------------------------------------
    // CORRECTION — Landmark  z_lm = [r, phi]
    //
    // h_lm(x) = [sqrt(dx²+dy²), atan2(dy,dx)-theta]  — nichtlinear!
    // -----------------------------------------------------------------------
    Vector6d correctLandmark(
        const Eigen::Vector2d & z_lm,
        const Eigen::Vector2d & landmark)
    {
        const double x = mu_(0), y = mu_(1), theta = mu_(2);
        const double lx = landmark(0), ly = landmark(1);
        const double dx = lx - x, dy = ly - y;
        const double q = dx*dx + dy*dy;
        const double r = std::sqrt(q);
        if (r < 1e-6) return mu_;

        // h(mu) — erwartete Messung
        Eigen::Vector2d z_hat;
        z_hat(0) = r;
        z_hat(1) = correctAngle(std::atan2(dy, dx) - theta);

        // H — Jacobi-Matrix von h nach x (2x6), EKF-Linearisierung
        //
        //        x        y      theta   vx   vy  omega
        //   r  [-dx/r   -dy/r     0       0    0    0  ]
        //  phi [ dy/q   -dx/q    -1       0    0    0  ]
        Matrix2x6d H = Matrix2x6d::Zero();
        H(0,0) = -dx/r;   // dr/dx
        H(0,1) = -dy/r;   // dr/dy
        H(1,0) =  dy/q;   // dphi/dx
        H(1,1) = -dx/q;   // dphi/dy
        H(1,2) = -1.0;    // dphi/dtheta

        // =====================
        // KALMAN-GAIN
        // =====================

        // Step 3: Compute Kalman Gain  [Line 4: K = Sigma * H^T * (H * Sigma * H^T + Q)^-1]
        Eigen::Matrix2d S = H * Sigma_ * H.transpose() + Q_lm_;
        K_lm_ = Sigma_ * H.transpose() * S.inverse();

        // =====================
        // CORRECT
        // =====================

        // Step 4: Correct mean with measurement  [Line 5: mu = mu + K * (z - h(mu))]
        Eigen::Vector2d innovation = z_lm - z_hat;
        innovation(1) = correctAngle(innovation(1));
        mu_ = mu_ + K_lm_ * innovation;
        mu_(2) = correctAngle(mu_(2));

        // Step 5: Correct covariance  [Line 6: Sigma = (I - K * H) * Sigma]
        Sigma_ = (Matrix6d::Identity() - K_lm_ * H) * Sigma_;

        return mu_;
    }

    const Vector6d   & state()               const { return mu_; }
    const Vector6d   & predictedState()      const { return mu_bar_; }
    const Matrix6d   & covariance()          const { return Sigma_; }
    const Matrix6d   & predictedCovariance() const { return Sigma_bar_; }
    const Matrix6d   & jacobianG()           const { return G_; }
    const Matrix6x3d & kalmanGainVelocity()  const { return K_vel_; }
    const Matrix6x2d & kalmanGainLandmark()  const { return K_lm_; }

private:
    static double correctAngle(double a)
    { return std::atan2(std::sin(a), std::cos(a)); }

    Vector6d mu_;         // Zustand µ
    Vector6d mu_bar_;     // Prädizierter Zustand µ_bar

    Matrix6d Sigma_;      // Kovarianz Σ
    Matrix6d Sigma_bar_;  // Prädizierte Kovarianz Σ_bar

    Matrix6d        G_;      // Jacobi-Matrix von g (EKF Prediction)
    Matrix6d        R_;      // Prozessrauschen R
    Eigen::Matrix3d Q_vel_;  // Messrauschen Velocity (odom twist)
    Eigen::Matrix2d Q_lm_;   // Messrauschen Landmark

    Matrix6x3d K_vel_;   // Kalman Gain Velocity
    Matrix6x2d K_lm_;    // Kalman Gain Landmark
};