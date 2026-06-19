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
// Weighting Full (Odom):
//   z = [x_map, y_map, theta_map, vx_world, vy_world, omega]
//   Fehler pro Partikel über alle 6 Komponenten:
//   w_i ∝ exp(-0.5 * sum(err_k² / Q_k))
//
// Rauschen:
//   R      — Bewegungsrauschen  (6x6, Diagonale)
//   Q_full — Messrauschen Full  (6x6, Diagonale)
//   Q_lm   — Messrauschen Landmark (2x2)
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

        // Q_full — Messrauschen Full (6x6) — Diagonalmatrix
        //
        //         x       y     theta    vx      vy    omega
        //   x  [ 0.05     0      0       0       0      0   ]
        //   y  [  0     0.05     0       0       0      0   ]
        // theta [  0       0    0.02     0       0      0   ]
        //  vx  [  0       0      0      0.05     0      0   ]
        //  vy  [  0       0      0       0      0.05    0   ]
        // omega [  0       0      0       0       0     0.03 ]
        Q_full_ = Eigen::Matrix<double, 6, 6>::Zero();
        Q_full_(0,0) = 0.05;
        Q_full_(1,1) = 0.05;
        Q_full_(2,2) = 0.02;
        Q_full_(3,3) = 0.05;
        Q_full_(4,4) = 0.05;
        Q_full_(5,5) = 0.03;

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
        double r_x,    double r_y,    double r_theta,
        double r_vx,   double r_vy,   double r_omega,
        double q_x,    double q_y,    double q_theta,
        double q_vx,   double q_vy,   double q_omega,
        double q_lm_r, double q_lm_phi)
    {
        // R — Bewegungsrauschen
        R_(0,0) = r_x;    R_(1,1) = r_y;    R_(2,2) = r_theta;
        R_(3,3) = r_vx;   R_(4,4) = r_vy;   R_(5,5) = r_omega;

        // Q — Messrauschen Full
        Q_full_(0,0) = q_x;    Q_full_(1,1) = q_y;    Q_full_(2,2) = q_theta;
        Q_full_(3,3) = q_vx;   Q_full_(4,4) = q_vy;   Q_full_(5,5) = q_omega;

        // Q — Messrauschen Landmark
        Q_lm_(0,0) = q_lm_r;   Q_lm_(1,1) = q_lm_phi;
    }

    // -----------------------------------------------------------------------
    // INITIALIZATION
    // Partikel gleichverteilt im ganzen erlaubten Raum erzeugen
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
    // PREDICTION + WEIGHTING + RESAMPLING
    // -----------------------------------------------------------------------
    Vector6d update(const Eigen::Vector2d & u, const Vector6d & z_full, double dt)
    {
        predict(u, dt);
        computeWeightsFull(z_full);
        resample();
        return mu_;
    }

    Vector6d updateLandmark(const Eigen::Vector2d & z_lm, const Eigen::Vector2d & lm)
    {
        computeWeightsLandmark(z_lm, lm);
        resample();
        return mu_;
    }

    // -----------------------------------------------------------------------
    // PREDICT
    // -----------------------------------------------------------------------
    Vector6d predict(const Eigen::Vector2d & u, double dt)
    {
        const double v = u(0), omega = u(1);

        std::normal_distribution<double> n_x  (0.0, std::sqrt(R_(0,0)));
        std::normal_distribution<double> n_y  (0.0, std::sqrt(R_(1,1)));
        std::normal_distribution<double> n_t  (0.0, std::sqrt(R_(2,2)));
        std::normal_distribution<double> n_vx (0.0, std::sqrt(R_(3,3)));
        std::normal_distribution<double> n_vy (0.0, std::sqrt(R_(4,4)));
        std::normal_distribution<double> n_td (0.0, std::sqrt(R_(5,5)));

        // =====================
        // PREDICT
        // =====================

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

    const Vector6d             & state()      const { return mu_; }
    const Vector6d             & predictedState() const { return mu_bar_; }
    const std::vector<Vector6d>& particles()  const { return particles_; }
    const std::vector<double>  & weights()    const { return weights_; }

private:
    static double correctAngle(double a)
    { return std::atan2(std::sin(a), std::cos(a)); }

    Vector6d computeMean() const
    {
        Vector6d m = Vector6d::Zero();
        double ss = 0.0, cs = 0.0;
        for (const auto & p : particles_) {
            m(0) += p(0); m(1) += p(1);
            ss   += std::sin(p(2));
            cs   += std::cos(p(2));
            m(3) += p(3); m(4) += p(4); m(5) += p(5);
        }
        const double n = num_particles_;
        m(0) /= n; m(1) /= n;
        m(2) = std::atan2(ss, cs);
        m(3) /= n; m(4) /= n; m(5) /= n;
        return m;
    }

    void normalizeWeights(double ws)
    {
        if (ws > 0.0) { for (auto & w : weights_) w /= ws; }
        else          { for (auto & w : weights_) w  = 1.0 / num_particles_; }
    }

    // -----------------------------------------------------------------------
    // WEIGHTING — Full
    // Vergleich Partikel [x, y, theta, vx, vy, omega] mit z_full
    // w_i ∝ exp(-0.5 * sum(err_k² / Q_k))
    // -----------------------------------------------------------------------
    void computeWeightsFull(const Vector6d & z_full)
    {
        // =====================
        // WEIGHTING
        // =====================

        double ws = 0.0;

        for (int i = 0; i < num_particles_; ++i) {
            Vector6d err = particles_[i] - z_full;
            err(2) = correctAngle(err(2));  // theta wrappen

            double exp_val = -0.5 * (
                err(0)*err(0) / Q_full_(0,0) +
                err(1)*err(1) / Q_full_(1,1) +
                err(2)*err(2) / Q_full_(2,2) +
                err(3)*err(3) / Q_full_(3,3) +
                err(4)*err(4) / Q_full_(4,4) +
                err(5)*err(5) / Q_full_(5,5));

            weights_[i] = std::exp(exp_val) + 1e-300;
            ws += weights_[i];
        }

        normalizeWeights(ws);
    }

    // -----------------------------------------------------------------------
    // WEIGHTING — Landmark
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
    // -----------------------------------------------------------------------
    void resample()
    {
        // =====================
        // RESAMPLING
        // =====================

        const double avg = 1.0 / num_particles_;
        const double thr = threshold_factor_ * avg;

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
    Eigen::Matrix<double, 6, 6> Q_full_;
    Eigen::Matrix2d             Q_lm_;

    std::mt19937 random_generator_;
};