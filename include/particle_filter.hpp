#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <random>
#include <vector>
#include <string>

class ParticleFilter
{
public:
    // Constructor
    ParticleFilter(
        int num_particles = 1000,
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
        // process noise
        // describes uncertainty in the motion model
        R_ = Eigen::Matrix3d::Zero();
        R_(0, 0) = 0.05;
        R_(1, 1) = 0.05;
        R_(2, 2) = 0.02;

        // measurement noise
        // describes uncertainty in the odometry measurement
        Q_ = Eigen::Matrix3d::Zero();
        Q_(0, 0) = 0.05;
        Q_(1, 1) = 0.05;
        Q_(2, 2) = 0.03;

        initializeParticles();
    }

    // Destructor
    ~ParticleFilter() = default;

    // Set resampling method from outside, e.g. from ROS parameter
    // Available methods:
    // "multinomial"
    // "systematic"
    // "stratified"
    void setResamplingMethod(const std::string & method)
    {
        resampling_method_ = method;
    }

    // Initialize particles globally in the given x-y range
    void initializeParticles()
    {
        particles_.clear();
        weights_.clear();

        std::uniform_real_distribution<double> x_distribution(x_min_, x_max_);
        std::uniform_real_distribution<double> y_distribution(y_min_, y_max_);
        std::uniform_real_distribution<double> theta_distribution(-M_PI, M_PI);

        for (int i = 0; i < num_particles_; ++i) {
            Eigen::Vector3d particle;

            particle << x_distribution(random_generator_),
                        y_distribution(random_generator_),
                        theta_distribution(random_generator_);

            particles_.push_back(particle);
            weights_.push_back(1.0 / static_cast<double>(num_particles_));
        }

        mu_ = computeMean();
        mu_bar_ = mu_;
    }

    // Initialize particles around a given start state
    // This is useful when the first odometry measurement is available
    void initializeParticlesAroundState(const Eigen::Vector3d & initial_state)
    {
        particles_.clear();
        weights_.clear();

        std::normal_distribution<double> noise_x(0.0, 0.2);
        std::normal_distribution<double> noise_y(0.0, 0.2);
        std::normal_distribution<double> noise_theta(0.0, 0.1);

        for (int i = 0; i < num_particles_; ++i) {
            Eigen::Vector3d particle;

            particle << initial_state(0) + noise_x(random_generator_),
                        initial_state(1) + noise_y(random_generator_),
                        correctAngle(initial_state(2) + noise_theta(random_generator_));

            particles_.push_back(particle);
            weights_.push_back(1.0 / static_cast<double>(num_particles_));
        }

        mu_ = computeMean();
        mu_bar_ = mu_;
    }

    // Prediction step
    // u = [v, omega]
    // dt = time difference between two odometry messages
    Eigen::Vector3d predict(const Eigen::Vector2d & u, double dt)
    {
        // control input
        const double v = u(0);
        const double omega = u(1);

        // Motion noise distributions
        std::normal_distribution<double> noise_x(0.0, std::sqrt(R_(0, 0)));
        std::normal_distribution<double> noise_y(0.0, std::sqrt(R_(1, 1)));
        std::normal_distribution<double> noise_theta(0.0, std::sqrt(R_(2, 2)));

        // Move every particle using the motion model
        for (auto & particle : particles_) {
            const double theta = particle(2);

            particle(0) = particle(0)
                + v * std::cos(theta) * dt
                + noise_x(random_generator_);

            particle(1) = particle(1)
                + v * std::sin(theta) * dt
                + noise_y(random_generator_);

            particle(2) = correctAngle(
                particle(2)
                + omega * dt
                + noise_theta(random_generator_)
            );
        }

        // Mean of predicted particles
        mu_bar_ = computeMean();

        return mu_bar_;
    }

    // Weighting step
    // z = [x, y, theta]
    void computeWeights(const Eigen::Vector3d & z)
    {
        double weight_sum = 0.0;

        for (int i = 0; i < num_particles_; ++i) {
            Eigen::Vector3d error = particles_[i] - z;
            error(2) = correctAngle(error(2));

            // Same idea as in the Python code:
            // weight = exp(-0.5 * sum(error^2 / Q_diag))
            double exponent =
                -0.5 * (
                    (error(0) * error(0)) / Q_(0, 0) +
                    (error(1) * error(1)) / Q_(1, 1) +
                    (error(2) * error(2)) / Q_(2, 2)
                );

            weights_[i] = std::exp(exponent) + 1e-300;
            weight_sum += weights_[i];
        }

        // Normalize weights
        if (weight_sum > 0.0) {
            for (auto & weight : weights_) {
                weight = weight / weight_sum;
            }
        } else {
            // fallback: reset to uniform weights
            resetWeights();
        }
    }

    // Resampling step
    // Calls the selected resampling strategy
    void resample()
    {
        if (resampling_method_ == "multinomial") {
            resampleMultinomial();
        } else if (resampling_method_ == "systematic") {
            resampleSystematic();
        } else if (resampling_method_ == "stratified") {
            resampleStratified();
        } else {
            // fallback if unknown method was selected
            resampleMultinomial();
        }
    }

    // predict + correct + resample
    Eigen::Vector3d update(
        const Eigen::Vector2d & u,
        const Eigen::Vector3d & z,
        double dt)
    {
        predict(u, dt);
        computeWeights(z);
        resample();

        return mu_;
    }

    // getter
    const Eigen::Vector3d & state() const
    {
        return mu_;
    }

    const Eigen::Vector3d & predictedState() const
    {
        return mu_bar_;
    }

    const std::vector<Eigen::Vector3d> & particles() const
    {
        return particles_;
    }

    const std::vector<double> & weights() const
    {
        return weights_;
    }

    const std::string & resamplingMethod() const
    {
        return resampling_method_;
    }

private:
    static double correctAngle(double angle)
    {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    // Compute mean pose of all particles
    Eigen::Vector3d computeMean() const
    {
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();

        double sin_sum = 0.0;
        double cos_sum = 0.0;

        for (const auto & particle : particles_) {
            mean(0) += particle(0);
            mean(1) += particle(1);

            sin_sum += std::sin(particle(2));
            cos_sum += std::cos(particle(2));
        }

        mean(0) = mean(0) / static_cast<double>(num_particles_);
        mean(1) = mean(1) / static_cast<double>(num_particles_);
        mean(2) = std::atan2(sin_sum, cos_sum);

        return mean;
    }

    // Reset all particle weights to uniform distribution
    void resetWeights()
    {
        for (auto & weight : weights_) {
            weight = 1.0 / static_cast<double>(num_particles_);
        }
    }

    // Multinomial resampling
    // This is the simplest strategy.
    // It samples each new particle independently according to the weights.
    void resampleMultinomial()
    {
        std::vector<Eigen::Vector3d> resampled_particles;
        resampled_particles.reserve(num_particles_);

        std::discrete_distribution<int> distribution(
            weights_.begin(),
            weights_.end()
        );

        for (int i = 0; i < num_particles_; ++i) {
            int index = distribution(random_generator_);
            resampled_particles.push_back(particles_[index]);
        }

        particles_ = resampled_particles;
        resetWeights();

        // Mean of corrected particles
        mu_ = computeMean();
    }

    // Systematic resampling
    // Uses one random starting point and then samples at fixed intervals.
    // Usually has lower variance than multinomial resampling.
    void resampleSystematic()
    {
        std::vector<Eigen::Vector3d> resampled_particles;
        resampled_particles.reserve(num_particles_);

        std::uniform_real_distribution<double> distribution(
            0.0,
            1.0 / static_cast<double>(num_particles_)
        );

        double start = distribution(random_generator_);
        double cumulative_weight = weights_[0];
        int index = 0;

        for (int i = 0; i < num_particles_; ++i) {
            double position =
                start + static_cast<double>(i) / static_cast<double>(num_particles_);

            while (position > cumulative_weight && index < num_particles_ - 1) {
                index++;
                cumulative_weight += weights_[index];
            }

            resampled_particles.push_back(particles_[index]);
        }

        particles_ = resampled_particles;
        resetWeights();

        // Mean of corrected particles
        mu_ = computeMean();
    }

    // Stratified resampling
    // Divides [0, 1] into N intervals and samples once inside each interval.
    // This also reduces sampling variance compared to multinomial resampling.
    void resampleStratified()
    {
        std::vector<Eigen::Vector3d> resampled_particles;
        resampled_particles.reserve(num_particles_);

        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        double cumulative_weight = weights_[0];
        int index = 0;

        for (int i = 0; i < num_particles_; ++i) {
            double random_value = distribution(random_generator_);

            double position =
                (static_cast<double>(i) + random_value) /
                static_cast<double>(num_particles_);

            while (position > cumulative_weight && index < num_particles_ - 1) {
                index++;
                cumulative_weight += weights_[index];
            }

            resampled_particles.push_back(particles_[index]);
        }

        particles_ = resampled_particles;
        resetWeights();

        // Mean of corrected particles
        mu_ = computeMean();
    }

    // number of particles
    int num_particles_;

    // allowed initialization range
    double x_min_;
    double x_max_;
    double y_min_;
    double y_max_;

    // particles
    std::vector<Eigen::Vector3d> particles_;

    // weights
    std::vector<double> weights_;

    // selected resampling method
    std::string resampling_method_{"multinomial"};

    // state estimate
    Eigen::Vector3d mu_ = Eigen::Vector3d::Zero();

    // predicted state estimate
    Eigen::Vector3d mu_bar_ = Eigen::Vector3d::Zero();

    // noise
    Eigen::Matrix3d R_;
    Eigen::Matrix3d Q_;

    // random generator
    std::mt19937 random_generator_;
};