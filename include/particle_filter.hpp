#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <random>
#include <vector>
#include <string>

// State: [x, y, theta, vx, vy, theta_dot]
// Control input: u = [v, omega]  (from /cmd_vel)
// Odom measurement:     z_odom = [x, y, theta]
// Landmark measurement: z_lm   = [r, phi]
//
// Particle weighting uses both odom and landmark measurements.
// When a landmark measurement arrives, particle weights are updated
// based on how well each particle predicts the observed [r, phi].

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
        // Process noise R (6x6)
        R_ = Eigen::Matrix<double, 6, 6>::Zero();
        R_(0, 0) = 0.05;
        R_(1, 1) = 0.05;
        R_(2, 2) = 0.02;
        R_(3, 3) = 0.1;
        R_(4, 4) = 0.05;
        R_(5, 5) = 0.05;

        // Odom measurement noise Q_odom (3x3)
        Q_odom_ = Eigen::Matrix3d::Zero();
        Q_odom_(0, 0) = 0.05;
        Q_odom_(1, 1) = 0.05;
        Q_odom_(2, 2) = 0.03;

        // Landmark measurement noise Q_lm (2x2)
        Q_lm_ = Eigen::Matrix2d::Zero();
        Q_lm_(0, 0) = 0.1;
        Q_lm_(1, 1) = 0.02;

        initializeParticles();
    }

    ~ParticleFilter() = default;

    void setResamplingMethod(const std::string & method)
    {
        resampling_method_ = method;
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

    // Prediction step
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

    // Odom weighting — compare [x, y, theta] part of each particle
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

    // Landmark weighting — compare predicted [r, phi] per particle
    // against actual LiDAR measurement
    // landmark = known (lx, ly) in map frame
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

            // Multiply into existing weights (combine with odom weight)
            weights_[i] *= std::exp(exponent) + 1e-300;
            weight_sum  += weights_[i];
        }

        normalizeWeights(weight_sum);
    }

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

    // Normal update: predict + odom weighting + resample
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

    // Landmark update: additional weighting when landmark is detected
    // Call AFTER update() in the same timestep
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
    const std::string & resamplingMethod()    const { return resampling_method_; }

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

    void resampleMultinomial()
    {
        std::vector<Vector6d> resampled;
        resampled.reserve(num_particles_);
        std::discrete_distribution<int> dist(weights_.begin(), weights_.end());
        for (int i = 0; i < num_particles_; ++i) {
            resampled.push_back(particles_[dist(random_generator_)]);
        }
        particles_ = resampled;
        for (auto & w : weights_) { w = 1.0 / static_cast<double>(num_particles_); }
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
            double pos = start + static_cast<double>(i) / static_cast<double>(num_particles_);
            while (pos > cumul && idx < num_particles_ - 1) { ++idx; cumul += weights_[idx]; }
            resampled.push_back(particles_[idx]);
        }
        particles_ = resampled;
        for (auto & w : weights_) { w = 1.0 / static_cast<double>(num_particles_); }
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
            while (pos > cumul && idx < num_particles_ - 1) { ++idx; cumul += weights_[idx]; }
            resampled.push_back(particles_[idx]);
        }
        particles_ = resampled;
        for (auto & w : weights_) { w = 1.0 / static_cast<double>(num_particles_); }
        mu_ = computeMean();
    }

    int num_particles_;
    double x_min_, x_max_, y_min_, y_max_;

    std::vector<Vector6d> particles_;
    std::vector<double>   weights_;
    std::string           resampling_method_{"multinomial"};

    Vector6d mu_     = Vector6d::Zero();
    Vector6d mu_bar_ = Vector6d::Zero();

    Eigen::Matrix<double, 6, 6> R_;
    Eigen::Matrix3d             Q_odom_;
    Eigen::Matrix2d             Q_lm_;

    std::mt19937 random_generator_;
};