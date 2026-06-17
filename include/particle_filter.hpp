#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
#include <string>

// State: [x, y, theta, vx, vy, theta_dot]
// Control input: u = [v, omega]  (from /cmd_vel)
// Odom measurement:     z_odom = [x, y, theta]
// Landmark measurement: z_lm   = [r, phi]
//
// Resampling strategy: Threshold + Multinomial
//
//   Before drawing, every particle whose weight is below
//       threshold_factor_ * (1.0 / N)
//   is excluded from the candidate pool.
//   The remaining "good" particles are then resampled with
//   standard multinomial drawing (proportional to weight).
//
//   threshold_factor_ = 0.0  →  pure multinomial (no filtering)
//   threshold_factor_ = 0.5  →  particles below 50 % of average are dropped
//   threshold_factor_ = 1.5  →  only well-above-average particles survive
//
//   *** Change only threshold_factor_ to compare the three strategies. ***

using Vector6d = Eigen::Matrix<double, 6, 1>;

class ParticleFilter
{
public:
    ParticleFilter(
        int num_particles = 1000,
        double x_min = 0.0, double x_max = 6.0,
        double y_min = 0.0, double y_max = 10.0)
    : num_particles_(num_particles),
      x_min_(x_min), x_max_(x_max),
      y_min_(y_min), y_max_(y_max),
      random_generator_(std::random_device{}())
    {
        R_ = Eigen::Matrix<double, 6, 6>::Zero();
        R_(0, 0) = 0.05;
        R_(1, 1) = 0.05;
        R_(2, 2) = 0.02;
        R_(3, 3) = 0.1;
        R_(4, 4) = 0.05;
        R_(5, 5) = 0.05;

        Q_odom_ = Eigen::Matrix3d::Zero();
        Q_odom_(0, 0) = 0.05;
        Q_odom_(1, 1) = 0.05;
        Q_odom_(2, 2) = 0.03;

        Q_lm_ = Eigen::Matrix2d::Zero();
        Q_lm_(0, 0) = 0.1;
        Q_lm_(1, 1) = 0.02;

        initializeParticles();
    }

    ~ParticleFilter() = default;

    // =========================================================================
    //  *** CHANGE THIS VALUE to compare resampling strategies ***
    //
    //   0.0  →  pure multinomial  (baseline, no threshold)
    //   0.5  →  drop below 50 % of average weight
    //   1.5  →  drop below 150 % of average weight (aggressive)
    // =========================================================================
    double threshold_factor_ = 0.5;

    // Set noise parameters from ROS2 params / filter_params.yaml
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

    void initializeParticles()
    {
        particles_.clear();
        weights_.clear();

        std::uniform_real_distribution<double> x_dist(x_min_, x_max_);
        std::uniform_real_distribution<double> y_dist(y_min_, y_max_);
        std::uniform_real_distribution<double> theta_dist(-M_PI, M_PI);

        for (int i = 0; i < num_particles_; ++i) {
            Vector6d p = Vector6d::Zero();
            p(0) = x_dist(random_generator_);
            p(1) = y_dist(random_generator_);
            p(2) = theta_dist(random_generator_);
            particles_.push_back(p);
            weights_.push_back(1.0 / static_cast<double>(num_particles_));
        }

        mu_     = computeMean();
        mu_bar_ = mu_;
    }

    void initializeParticlesAroundState(const Eigen::Vector3d & initial_pose)
    {
        particles_.clear();
        weights_.clear();

        std::normal_distribution<double> nx(0.0, 0.2);
        std::normal_distribution<double> ny(0.0, 0.2);
        std::normal_distribution<double> ntheta(0.0, 0.1);

        for (int i = 0; i < num_particles_; ++i) {
            Vector6d p = Vector6d::Zero();
            p(0) = initial_pose(0) + nx(random_generator_);
            p(1) = initial_pose(1) + ny(random_generator_);
            p(2) = correctAngle(initial_pose(2) + ntheta(random_generator_));
            particles_.push_back(p);
            weights_.push_back(1.0 / static_cast<double>(num_particles_));
        }

        mu_     = computeMean();
        mu_bar_ = mu_;
    }

    // Prediction: sample motion model for each particle
    Vector6d predict(const Eigen::Vector2d & u, double dt)
    {
        const double v     = u(0);
        const double omega = u(1);

        std::normal_distribution<double> n_x    (0.0, std::sqrt(R_(0, 0)));
        std::normal_distribution<double> n_y    (0.0, std::sqrt(R_(1, 1)));
        std::normal_distribution<double> n_theta(0.0, std::sqrt(R_(2, 2)));
        std::normal_distribution<double> n_vx   (0.0, std::sqrt(R_(3, 3)));
        std::normal_distribution<double> n_vy   (0.0, std::sqrt(R_(4, 4)));
        std::normal_distribution<double> n_tdot (0.0, std::sqrt(R_(5, 5)));

        for (auto & p : particles_) {
            const double theta = p(2);
            p(0) = p(0) + p(3) * dt + n_x(random_generator_);
            p(1) = p(1) + p(4) * dt + n_y(random_generator_);
            p(2) = correctAngle(p(2) + p(5) * dt + n_theta(random_generator_));
            p(3) = v * std::cos(theta) + n_vx(random_generator_);
            p(4) = v * std::sin(theta) + n_vy(random_generator_);
            p(5) = omega               + n_tdot(random_generator_);
        }

        mu_bar_ = computeMean();
        return mu_bar_;
    }

    // Normal cycle: predict + odom weighting + resample
    Vector6d update(
        const Eigen::Vector2d & u,
        const Eigen::Vector3d & z_odom,
        double dt)
    {
        predict(u, dt);
        computeWeightsOdom(z_odom);
        resample();
        return mu_;
    }

    // Landmark update: call AFTER update() in the same timestep
    Vector6d updateLandmark(
        const Eigen::Vector2d & z_lm,
        const Eigen::Vector2d & landmark)
    {
        computeWeightsLandmark(z_lm, landmark);
        resample();
        return mu_;
    }

    const Vector6d & state()          const { return mu_; }
    const Vector6d & predictedState() const { return mu_bar_; }
    const std::vector<Vector6d> & particles() const { return particles_; }
    const std::vector<double>   & weights()   const { return weights_; }

private:
    static double correctAngle(double angle)
    {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    Vector6d computeMean() const
    {
        Vector6d mean = Vector6d::Zero();
        double sin_sum = 0.0;
        double cos_sum = 0.0;

        for (const auto & p : particles_) {
            mean(0) += p(0);
            mean(1) += p(1);
            sin_sum += std::sin(p(2));
            cos_sum += std::cos(p(2));
            mean(3) += p(3);
            mean(4) += p(4);
            mean(5) += p(5);
        }

        const double n = static_cast<double>(num_particles_);
        mean(0) /= n;
        mean(1) /= n;
        mean(2)  = std::atan2(sin_sum, cos_sum);
        mean(3) /= n;
        mean(4) /= n;
        mean(5) /= n;

        return mean;
    }

    void normalizeWeights(double weight_sum)
    {
        if (weight_sum > 0.0) {
            for (auto & w : weights_) { w /= weight_sum; }
        } else {
            for (auto & w : weights_) {
                w = 1.0 / static_cast<double>(num_particles_);
            }
        }
    }

    // Odom weighting
    void computeWeightsOdom(const Eigen::Vector3d & z_odom)
    {
        double weight_sum = 0.0;

        for (int i = 0; i < num_particles_; ++i) {
            Eigen::Vector3d error;
            error(0) = particles_[i](0) - z_odom(0);
            error(1) = particles_[i](1) - z_odom(1);
            error(2) = correctAngle(particles_[i](2) - z_odom(2));

            double exponent =
                -0.5 * (
                    (error(0) * error(0)) / Q_odom_(0, 0) +
                    (error(1) * error(1)) / Q_odom_(1, 1) +
                    (error(2) * error(2)) / Q_odom_(2, 2)
                );

            weights_[i]  = std::exp(exponent) + 1e-300;
            weight_sum  += weights_[i];
        }

        normalizeWeights(weight_sum);
    }

    // Landmark weighting
    void computeWeightsLandmark(
        const Eigen::Vector2d & z_lm,
        const Eigen::Vector2d & landmark)
    {
        double weight_sum = 0.0;
        const double lx = landmark(0);
        const double ly = landmark(1);

        for (int i = 0; i < num_particles_; ++i) {
            const double x     = particles_[i](0);
            const double y     = particles_[i](1);
            const double theta = particles_[i](2);

            const double dx  = lx - x;
            const double dy  = ly - y;
            const double r   = std::sqrt(dx * dx + dy * dy);
            const double phi = correctAngle(std::atan2(dy, dx) - theta);

            Eigen::Vector2d z_hat;
            z_hat(0) = r;
            z_hat(1) = phi;

            Eigen::Vector2d error = z_lm - z_hat;
            error(1) = correctAngle(error(1));

            double exponent =
                -0.5 * (
                    (error(0) * error(0)) / Q_lm_(0, 0) +
                    (error(1) * error(1)) / Q_lm_(1, 1)
                );

            weights_[i] *= std::exp(exponent) + 1e-300;
            weight_sum  += weights_[i];
        }

        normalizeWeights(weight_sum);
    }

    // =========================================================================
    // Threshold + Multinomial resampling
    //
    // Step 1: Drop all particles whose weight is below the threshold.
    //         threshold = threshold_factor_ * (1.0 / N)
    //         If threshold_factor_ = 0.0, no particle is dropped (= pure multinomial).
    //
    // Step 2: Draw N particles from the surviving candidates,
    //         proportional to their weight (multinomial).
    //
    // Step 3: Reset all weights to 1/N.
    // =========================================================================
    void resample()
    {
        const double avg_weight = 1.0 / static_cast<double>(num_particles_);
        const double threshold  = threshold_factor_ * avg_weight;

        // Step 1: collect indices of particles that pass the threshold
        std::vector<int> good_indices;
        good_indices.reserve(num_particles_);
        for (int i = 0; i < num_particles_; ++i) {
            if (weights_[i] >= threshold) {
                good_indices.push_back(i);
            }
        }

        // Safety: if everything got filtered (shouldn't happen but just in case),
        // fall back to all particles so the filter never crashes.
        if (good_indices.empty()) {
            good_indices.resize(num_particles_);
            std::iota(good_indices.begin(), good_indices.end(), 0);
        }

        // Step 2: build weight list for only the good particles
        std::vector<double> good_weights;
        good_weights.reserve(good_indices.size());
        for (int idx : good_indices) {
            good_weights.push_back(weights_[idx]);
        }

        // Step 3: multinomial draw from the good candidates
        std::discrete_distribution<int> dist(good_weights.begin(), good_weights.end());

        std::vector<Vector6d> resampled;
        resampled.reserve(num_particles_);
        for (int i = 0; i < num_particles_; ++i) {
            int chosen = good_indices[dist(random_generator_)];
            resampled.push_back(particles_[chosen]);
        }

        particles_ = resampled;
        for (auto & w : weights_) { w = avg_weight; }
        mu_ = computeMean();
    }

    // =========================================================================
    // Member variables
    // =========================================================================
    int    num_particles_;
    double x_min_, x_max_, y_min_, y_max_;

    std::vector<Vector6d> particles_;
    std::vector<double>   weights_;

    Vector6d mu_     = Vector6d::Zero();
    Vector6d mu_bar_ = Vector6d::Zero();

    Eigen::Matrix<double, 6, 6> R_;
    Eigen::Matrix3d             Q_odom_;
    Eigen::Matrix2d             Q_lm_;

    std::mt19937 random_generator_;
};