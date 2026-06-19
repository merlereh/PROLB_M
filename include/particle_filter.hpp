#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

// ============================================================================
// Particle Filter  —  Notation nach Vorlesungsfolien (Thrun 2006)
//
// State:   x = [x, y, theta, vx, vy, omega]   (6x1)
// Control: u = [v, omega]                      (2x1)
//
// Algorithmus:
//   Initialization:
//     Partikel gleichverteilt im erlaubten Raum erzeugen
//     Alle Gewichte gleich: w_i = 1 / N
//
//   Predict:
//     Jedes Partikel wird mit dem Bewegungsmodell + Rauschen bewegt
//
//   Weighting:
//     Jedes Partikel wird mit der Messung verglichen
//     w_i ∝ exp(-0.5 * ||z - h(x_i)||^2_Q)
//
//   Resampling:
//     Gute Partikel werden häufiger gezogen, schlechte verschwinden
//     Nach Resampling: alle Gewichte wieder gleich w_i = 1 / N
//
// Weighting Velocity (Odom-Twist):
//   z = [vx_world, vy_world, omega]
//   Fehler pro Partikel: err = [p_vx - vx_world, p_vy - vy_world, p_omega - omega]
//   w_i ∝ exp(-0.5 * (err_vx²/Q_vx + err_vy²/Q_vy + err_omega²/Q_omega))
//
// Rauschen:
//   R   — Bewegungsrauschen  (6x6, Diagonale)
//   Q   — Messrauschen       (3x3 Velocity / 2x2 Landmark)
// ============================================================================

using Vector6d = Eigen::Matrix<double, 6, 1>;

class ParticleFilter
{
public:
    ParticleFilter(int num_particles = 1000,
        double x_min = 0.0, double x_max = 6.0,
        double y_min = 0.0, double y_max = 10.0)
    : num_particles_(num_particles),
      x_min_(x_min), x_max_(x_max),
      y_min_(y_min), y_max_(y_max),
      random_generator_(std::random_device{}())
    {
        // R — Bewegungsrauschen (6x6) — Diagonalmatrix
        //
        //         x     y   theta   vx    vy   omega
        //   x  [ 0.05   0     0      0      0     0   ]
        //   y  [  0   0.05    0      0      0     0   ]
        // theta [  0    0    0.02    0      0     0   ]
        //  vx  [  0    0     0     0.1     0     0   ]
        //  vy  [  0    0     0      0    0.05    0   ]
        // omega [  0    0     0      0      0   0.05  ]
        R_ = Eigen::Matrix<double, 6, 6>::Zero();
        R_(0,0) = 0.05;
        R_(1,1) = 0.05;
        R_(2,2) = 0.02;
        R_(3,3) = 0.1;
        R_(4,4) = 0.05;
        R_(5,5) = 0.05;

        // Q_vel — Messrauschen Velocity (3x3) — Diagonalmatrix
        //
        //         vx        vy      omega
        //  vx  [ q_vx       0        0   ]
        //  vy  [   0      q_vy       0   ]
        // omega [   0        0    q_omega ]
        Q_vel_ = Eigen::Matrix3d::Zero();
        Q_vel_(0,0) = 0.05;
        Q_vel_(1,1) = 0.05;
        Q_vel_(2,2) = 0.03;

        // Q_lm — Messrauschen Landmark (2x2) — Diagonalmatrix
        //
        //       r      phi
        //   r  [ 0.1    0   ]
        //  phi [  0   0.02  ]
        Q_lm_ = Eigen::Matrix2d::Zero();
        Q_lm_(0,0) = 0.1;
        Q_lm_(1,1) = 0.02;

        initializeParticles();
    }

    ~ParticleFilter() = default;

    double threshold_factor_ = 0.5;

    void setNoiseParams(
        double r_x,  double r_y,  double r_theta,
        double r_vx, double r_vy, double r_omega,
        double q_odom_x, double q_odom_y, double q_odom_theta,
        double q_lm_r,   double q_lm_phi)
    {
        // R — Bewegungsrauschen
        R_(0,0) = r_x;    R_(1,1) = r_y;    R_(2,2) = r_theta;
        R_(3,3) = r_vx;   R_(4,4) = r_vy;   R_(5,5) = r_omega;

        // Q — Messrauschen
        Q_vel_(0,0)  = q_odom_x;   Q_vel_(1,1)  = q_odom_y;   Q_vel_(2,2)  = q_odom_theta;
        Q_lm_(0,0)   = q_lm_r;     Q_lm_(1,1)   = q_lm_phi;
    }

    // -----------------------------------------------------------------------
    // INITIALIZATION
    // Partikel gleichverteilt im ganzen erlaubten Raum erzeugen
    // Alle Gewichte gleich: w_i = 1 / N
    // -----------------------------------------------------------------------
    void initializeParticles()
    {
        particles_.clear();
        weights_.clear();

        std::uniform_real_distribution<double> x_dist(x_min_, x_max_);
        std::uniform_real_distribution<double> y_dist(y_min_, y_max_);
        std::uniform_real_distribution<double> t_dist(-M_PI, M_PI);

        for (int i = 0; i < num_particles_; ++i) {
            Vector6d p = Vector6d::Zero();
            p(0) = x_dist(random_generator_);
            p(1) = y_dist(random_generator_);
            p(2) = t_dist(random_generator_);
            particles_.push_back(p);
            weights_.push_back(1.0 / num_particles_);
        }

        mu_     = computeMean();
        mu_bar_ = mu_;
    }

    void initializeParticlesAroundState(const Eigen::Vector3d & pose)
    {
        particles_.clear();
        weights_.clear();

        std::normal_distribution<double> n_x(0.0, 1.0);
        std::normal_distribution<double> n_y(0.0, 1.0);
        std::normal_distribution<double> n_t(0.0, 0.5);

        for (int i = 0; i < num_particles_; ++i) {
            Vector6d p = Vector6d::Zero();
            p(0) = pose(0) + n_x(random_generator_);
            p(1) = pose(1) + n_y(random_generator_);
            p(2) = correctAngle(pose(2) + n_t(random_generator_));
            particles_.push_back(p);
            weights_.push_back(1.0 / num_particles_);
        }

        mu_     = computeMean();
        mu_bar_ = mu_;
    }

    // -----------------------------------------------------------------------
    // PREDICTION + WEIGHTING + RESAMPLING — Odometrie (alles in einem Schritt)
    // -----------------------------------------------------------------------
    Vector6d update(const Eigen::Vector2d & u, const Eigen::Vector3d & z_vel, double dt)
    {
        predict(u, dt);
        computeWeightsVelocity(z_vel);
        resample();
        return mu_;
    }

    // -----------------------------------------------------------------------
    // WEIGHTING + RESAMPLING — Landmark (ohne neue Prediction)
    // -----------------------------------------------------------------------
    Vector6d updateLandmark(const Eigen::Vector2d & z_lm, const Eigen::Vector2d & lm)
    {
        computeWeightsLandmark(z_lm, lm);
        resample();
        return mu_;
    }

    // -----------------------------------------------------------------------
    // PREDICT
    // Jedes Partikel wird mit dem Bewegungsmodell bewegt.
    // Zusätzlich bekommt jedes Partikel eigenes Bewegungsrauschen (aus R).
    // -----------------------------------------------------------------------
    Vector6d predict(const Eigen::Vector2d & u, double dt)
    {
        const double v = u(0), omega = u(1);

        // Rauschen pro Achse — Standardabweichung = sqrt(R_diag)
        std::normal_distribution<double> n_x  (0.0, std::sqrt(R_(0,0)));
        std::normal_distribution<double> n_y  (0.0, std::sqrt(R_(1,1)));
        std::normal_distribution<double> n_t  (0.0, std::sqrt(R_(2,2)));
        std::normal_distribution<double> n_vx (0.0, std::sqrt(R_(3,3)));
        std::normal_distribution<double> n_vy (0.0, std::sqrt(R_(4,4)));
        std::normal_distribution<double> n_td (0.0, std::sqrt(R_(5,5)));

        // =====================
        // PREDICT
        // =====================
        // Jedes Partikel wird einzeln bewegt + bekommt eigenes Rauschen

        for (auto & p : particles_) {
            const double theta = p(2);

            p(0) = p(0) + p(3) * dt + n_x(random_generator_);
            p(1) = p(1) + p(4) * dt + n_y(random_generator_);
            p(2) = correctAngle(p(2) + p(5) * dt + n_t(random_generator_));
            p(3) = v * std::cos(theta) + n_vx(random_generator_);
            p(4) = v * std::sin(theta) + n_vy(random_generator_);
            p(5) = omega              + n_td(random_generator_);
        }

        mu_bar_ = computeMean();
        return mu_bar_;
    }

    const Vector6d             & state()          const { return mu_; }
    const Vector6d             & predictedState() const { return mu_bar_; }
    const std::vector<Vector6d>& particles()      const { return particles_; }
    const std::vector<double>  & weights()        const { return weights_; }

private:
    static double correctAngle(double a)
    { return std::atan2(std::sin(a), std::cos(a)); }

    Vector6d computeMean() const
    {
        Vector6d m = Vector6d::Zero();
        double ss = 0.0, cs = 0.0;

        for (const auto & p : particles_) {
            m(0) += p(0);  m(1) += p(1);
            ss   += std::sin(p(2));
            cs   += std::cos(p(2));
            m(3) += p(3);  m(4) += p(4);  m(5) += p(5);
        }

        const double n = num_particles_;
        m(0) /= n;  m(1) /= n;
        m(2) = std::atan2(ss, cs);
        m(3) /= n;  m(4) /= n;  m(5) /= n;

        return m;
    }

    void normalizeWeights(double ws)
    {
        if (ws > 0.0) { for (auto & w : weights_) w /= ws; }
        else          { for (auto & w : weights_) w  = 1.0 / num_particles_; }
    }

    // -----------------------------------------------------------------------
    // WEIGHTING — Velocity (Odom-Twist)
    // Vergleich Partikel-Velocities [vx, vy, omega] mit Messung z_vel
    // z_vel = [vx_world, vy_world, omega]  aus odom.twist umgerechnet
    // w_i ∝ exp(-0.5 * (err_vx²/Q_vx + err_vy²/Q_vy + err_omega²/Q_omega))
    // -----------------------------------------------------------------------
    void computeWeightsVelocity(const Eigen::Vector3d & z_vel)
    {
        // =====================
        // WEIGHTING
        // =====================

        double ws = 0.0;

        for (int i = 0; i < num_particles_; ++i) {
            const double err_vx    = particles_[i](3) - z_vel(0);
            const double err_vy    = particles_[i](4) - z_vel(1);
            const double err_omega = particles_[i](5) - z_vel(2);

            double exp_val = -0.5 * (
                err_vx    * err_vx    / Q_vel_(0,0) +
                err_vy    * err_vy    / Q_vel_(1,1) +
                err_omega * err_omega / Q_vel_(2,2));

            weights_[i] = std::exp(exp_val) + 1e-300;
            ws += weights_[i];
        }

        normalizeWeights(ws);
    }

    // -----------------------------------------------------------------------
    // WEIGHTING — Landmark
    // Vergleich erwartete Messung h(x_i) = [r, phi] mit z_lm
    // w_i ∝ exp(-0.5 * ||z_lm - h(x_i)||^2_Q)
    // -----------------------------------------------------------------------
    void computeWeightsLandmark(const Eigen::Vector2d & z_lm, const Eigen::Vector2d & lm)
    {
        // =====================
        // WEIGHTING
        // =====================

        double ws = 0.0;
        const double lx = lm(0), ly = lm(1);

        for (int i = 0; i < num_particles_; ++i) {
            const double x     = particles_[i](0);
            const double y     = particles_[i](1);
            const double theta = particles_[i](2);
            const double dx    = lx - x;
            const double dy    = ly - y;
            const double r     = std::sqrt(dx*dx + dy*dy);
            const double phi   = correctAngle(std::atan2(dy, dx) - theta);

            Eigen::Vector2d err = z_lm - Eigen::Vector2d(r, phi);
            err(1) = correctAngle(err(1));

            double exp_val = -0.5 * (
                err(0)*err(0) / Q_lm_(0,0) +
                err(1)*err(1) / Q_lm_(1,1));

            weights_[i] *= std::exp(exp_val) + 1e-300;
            ws += weights_[i];
        }

        normalizeWeights(ws);
    }

    // -----------------------------------------------------------------------
    // RESAMPLING
    // Gute Partikel (hohes Gewicht) werden häufiger gezogen.
    // Schlechte Partikel (niedriges Gewicht) verschwinden.
    // Nach Resampling: alle Gewichte wieder gleich w_i = 1 / N
    // -----------------------------------------------------------------------
    void resample()
    {
        // =====================
        // RESAMPLING
        // =====================

        const double avg = 1.0 / num_particles_;
        const double thr = threshold_factor_ * avg;

        // Nur Partikel über Schwellwert behalten
        std::vector<int> good;
        good.reserve(num_particles_);
        for (int i = 0; i < num_particles_; ++i)
            if (weights_[i] >= thr) good.push_back(i);

        if (good.empty()) {
            good.resize(num_particles_);
            std::iota(good.begin(), good.end(), 0);
        }

        std::vector<double> gw;
        gw.reserve(good.size());
        for (int idx : good) gw.push_back(weights_[idx]);

        std::discrete_distribution<int> dist(gw.begin(), gw.end());

        std::vector<Vector6d> resampled;
        resampled.reserve(num_particles_);
        for (int i = 0; i < num_particles_; ++i)
            resampled.push_back(particles_[good[dist(random_generator_)]]);

        particles_ = resampled;

        // Nach Resampling: alle Gewichte wieder gleich
        for (auto & w : weights_) w = avg;

        mu_ = computeMean();
    }

    int    num_particles_;
    double x_min_, x_max_, y_min_, y_max_;

    std::vector<Vector6d> particles_;
    std::vector<double>   weights_;

    Vector6d mu_     = Vector6d::Zero();
    Vector6d mu_bar_ = Vector6d::Zero();

    Eigen::Matrix<double, 6, 6> R_;
    Eigen::Matrix3d             Q_vel_;
    Eigen::Matrix2d             Q_lm_;

    std::mt19937 random_generator_;
};