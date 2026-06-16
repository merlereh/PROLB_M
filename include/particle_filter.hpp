#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <random>
#include <vector>
#include <string>

// State: [x, y, theta, vx, vy, theta_dot]
// Each particle represents a full 6D state hypothesis.
//
// Control input: u = [v, omega]  (from /cmd_vel)
// Measurement:   z = [x, y, theta]  (from /odom, absolute pose)
//
// Motion model per particle (same nonlinear model as EKF):
//   x_new        = x + vx * dt
//   y_new        = y + vy * dt
//   theta_new    = theta + theta_dot * dt
//   vx_new       = v * cos(theta) + noise
//   vy_new       = v * sin(theta) + noise
//   theta_dot_new = omega + noise
//
// Weighting uses only the observable part z = [x, y, theta].

using Vector6d = Eigen::Matrix<double, 6, 1>;

class ParticleFilter
{
public:
    ParticleFilter(
        int num_particles = 500,
        double x_min = 0.0,
        double x_max = 6.0,
        double y_min = 0.0,
        double y_max = 10.0)
    : num_particles_(num_particles),
      x_min_(x_min),
      x_max_(x_max),
      y_min_(y_min),
      y_max_(y_max),
      random_generator_(std::random_device{}())
    {
        // Process noise R (6x6)
        R_ = Eigen::Matrix<double, 6, 6>::Zero();
        R_(0, 0) = 0.05;   // x
        R_(1, 1) = 0.05;   // y
        R_(2, 2) = 0.02;   // theta
        R_(3, 3) = 0.1;    // vx
        R_(4, 4) = 0.05;   // vy
        R_(5, 5) = 0.05;   // theta_dot

        // Measurement noise Q (3x3) — used for particle weighting
        Q_ = Eigen::Matrix3d::Zero();
        Q_(0, 0) = 0.05;   // x
        Q_(1, 1) = 0.05;   // y
        Q_(2, 2) = 0.03;   // theta

        initializeParticles();
    }

    ~ParticleFilter() = default;

    void setResamplingMethod(const std::string & method)
    {
        resampling_method_ = method;
    }

    // Initialize particles globally in the given x-y range
    // Velocities start at zero
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
            // vx, vy, theta_dot start at 0
            particles_.push_back(p);
            weights_.push_back(1.0 / static_cast<double>(num_particles_));
        }

        mu_     = computeMean();
        mu_bar_ = mu_;
    }

    // Initialize particles around a known first pose
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
            // vx, vy, theta_dot start at 0
            particles_.push_back(p);
            weights_.push_back(1.0 / static_cast<double>(num_particles_));
        }

        mu_     = computeMean();
        mu_bar_ = mu_;
    }

    // Prediction step
    // Each particle is propagated with the nonlinear motion model + noise
    Vector6d predict(const Eigen::Vector2d & u, double dt)
    {
        const double v     = u(0);
        const double omega = u(1);

        std::normal_distribution<double> n_x     (0.0, std::sqrt(R_(0, 0)));
        std::normal_distribution<double> n_y     (0.0, std::sqrt(R_(1, 1)));
        std::normal_distribution<double> n_theta (0.0, std::sqrt(R_(2, 2)));
        std::normal_distribution<double> n_vx    (0.0, std::sqrt(R_(3, 3)));
        std::normal_distribution<double> n_vy    (0.0, std::sqrt(R_(4, 4)));
        std::normal_distribution<double> n_tdot  (0.0, std::sqrt(R_(5, 5)));

        for (auto & p : particles_) {
            const double theta = p(2);

            // Propagate position from current velocity state
            p(0) = p(0) + p(3) * dt + n_x(random_generator_);
            p(1) = p(1) + p(4) * dt + n_y(random_generator_);
            p(2) = correctAngle(p(2) + p(5) * dt + n_theta(random_generator_));

            // Update velocity states from cmd_vel with cos/sin + noise
            p(3) = v * std::cos(theta) + n_vx(random_generator_);
            p(4) = v * std::sin(theta) + n_vy(random_generator_);
            p(5) = omega               + n_tdot(random_generator_);
        }

        mu_bar_ = computeMean();
        return mu_bar_;
    }

    // Weighting step
    // z = [x, y, theta]  — we can only compare the observable part
    void computeWeights(const Eigen::Vector3d & z)
    {
        double weight_sum = 0.0;

        for (int i = 0; i < num_particles_; ++i) {
            // Compare only [x, y, theta] part of the particle against measurement
            Eigen::Vector3d error;
            error(0) = particles_[i](0) - z(0);
            error(1) = particles_[i](1) - z(1);
            error(2) = correctAngle(particles_[i](2) - z(2));

            double exponent =
                -0.5 * (
                    (error(0) * error(0)) / Q_(0, 0) +
                    (error(1) * error(1)) / Q_(1, 1) +
                    (error(2) * error(2)) / Q_(2, 2)
                );

            weights_[i]  = std::exp(exponent) + 1e-300;
            weight_sum  += weights_[i];
        }

        if (weight_sum > 0.0) {
            for (auto & w : weights_) {
                w /= weight_sum;
            }
        } else {
            resetWeights();
        }
    }

    // Resampling step
    void resample()
    {
        if (resampling_method_ == "multinomial") {
            resampleMultinomial();
        } else if (resampling_method_ == "systematic") {
            resampleSystematic();
        } else if (resampling_method_ == "stratified") {
            resampleStratified();
        } else {
            resampleMultinomial();
        }
    }

    // Full update: predict → weight → resample
    Vector6d update(
        const Eigen::Vector2d & u,
        const Eigen::Vector3d & z,
        double dt)
    {
        predict(u, dt);
        computeWeights(z);
        resample();
        return mu_;
    }

    // Getters
    const Vector6d & state()         const { return mu_; }
    const Vector6d & predictedState()const { return mu_bar_; }
    const std::vector<Vector6d> & particles() const { return particles_; }
    const std::vector<double>   & weights()   const { return weights_; }
    const std::string & resamplingMethod()    const { return resampling_method_; }

private:
    static double correctAngle(double angle)
    {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    // Weighted mean of all particles
    // Theta uses circular mean (sin/cos averaging)
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

    void resetWeights()
    {
        for (auto & w : weights_) {
            w = 1.0 / static_cast<double>(num_particles_);
        }
    }

    void resampleMultinomial()
    {
        std::vector<Vector6d> resampled;
        resampled.reserve(num_particles_);

        std::discrete_distribution<int> dist(weights_.begin(), weights_.end());

        for (int i = 0; i < num_particles_; ++i) {
            resampled.push_back(particles_[dist(random_generator_)]);
        }

        particles_ = resampled;
        resetWeights();
        mu_ = computeMean();
    }

    void resampleSystematic()
    {
        std::vector<Vector6d> resampled;
        resampled.reserve(num_particles_);

        std::uniform_real_distribution<double> dist(
            0.0, 1.0 / static_cast<double>(num_particles_));

        double start = dist(random_generator_);
        double cumul = weights_[0];
        int idx = 0;

        for (int i = 0; i < num_particles_; ++i) {
            double pos = start + static_cast<double>(i) /
                         static_cast<double>(num_particles_);
            while (pos > cumul && idx < num_particles_ - 1) {
                ++idx;
                cumul += weights_[idx];
            }
            resampled.push_back(particles_[idx]);
        }

        particles_ = resampled;
        resetWeights();
        mu_ = computeMean();
    }

    void resampleStratified()
    {
        std::vector<Vector6d> resampled;
        resampled.reserve(num_particles_);

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double cumul = weights_[0];
        int idx = 0;

        for (int i = 0; i < num_particles_; ++i) {
            double pos = (static_cast<double>(i) + dist(random_generator_)) /
                          static_cast<double>(num_particles_);
            while (pos > cumul && idx < num_particles_ - 1) {
                ++idx;
                cumul += weights_[idx];
            }
            resampled.push_back(particles_[idx]);
        }

        particles_ = resampled;
        resetWeights();
        mu_ = computeMean();
    }

    int num_particles_;
    double x_min_, x_max_, y_min_, y_max_;

    std::vector<Vector6d>  particles_;
    std::vector<double>    weights_;
    std::string            resampling_method_{"multinomial"};

    Vector6d mu_     = Vector6d::Zero();
    Vector6d mu_bar_ = Vector6d::Zero();

    Eigen::Matrix<double, 6, 6> R_;
    Eigen::Matrix3d Q_;

    std::mt19937 random_generator_;
};