#include <iostream>
#include <random>

#include "graph/FactorGraph.h"
#include "graph/BaseUnaryEdge.h"
#include "graph/BaseVertex.h"
#include "solver/GaussNewtonSolver.h"
#include "solver/GaussNewtonShurSolver.h"
#include "solver/GaussNewtonSparseSolver.h"
#include "solver/LevenbergMarquartSparseSolver.h"
#include "util/TicToc.h"

struct CurveFittingVertex : public gopt::BaseVertex<3, Eigen::Vector3d> {
public: 
    CurveFittingVertex() : 
    BaseVertex() {
        estimate_.setZero();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void plus(const Eigen::VectorXd &update) override {
        estimate_ += update;
    }

    virtual void setToOrigin() override {
        estimate_.setZero();
    }
};

struct CurveFittingEdge : public gopt::BaseUnaryEdge<1, double, CurveFittingVertex> {
public: 
    CurveFittingEdge(double x) : 
    BaseUnaryEdge(), x_(x) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void computeResidual() override
    {
        std::shared_ptr<const CurveFittingVertex> v = std::dynamic_pointer_cast<const CurveFittingVertex>(vertices_[0]);
        Eigen::Vector3d abc = v->getEstimate();
        residual_.resize(1);
        y_ = std::exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2));
        residual_(0) = measurement_ - y_;
    }

    // virtual void computeJacobians() override {
    //     jacobians_[0] = Eigen::MatrixXd::Zero(1, 3);
    //     jacobians_[0] << -y_ * x_ * x_, 
    //                      -y_ * x_, 
    //                      -y_;
    // }

protected: 
    double x_;
    double y_;
};

int main(int argc, char **argv) {
    const double ar = 1.0, br = 2.0, cr = 1.0;
    const double ae = 2.0, be = -1.0, ce = 5.0;
    const int N = 100;
    const double noise_mean = 0.0, noise_sigma = 1.0; 
    const double inv_sigma = 1.0 / noise_sigma;
    std::default_random_engine engine;
    std::normal_distribution<double> dist(noise_mean, noise_sigma);

    std::vector<double> x_data, y_data;
    for (int i = 0; i < N; ++i) {
        double x = i / 100.0;
        double y = std::exp(ar * x * x + br * x + cr) + dist(engine);
        x_data.push_back(x);
        y_data.push_back(y);
    }

    std::cout << "Set solver. " << std::endl;
    gopt::LevenbergMarquartSparseSolver solver;
    gopt::FactorGraph graph;
    graph.setOptSolver(&solver);

    std::cout << "Set opt config. " << std::endl;
    gopt::optimization_config_t config;
    config.verbose = true;
    graph.setOptConfig(config);

    // Add the vertex
    std::cout << "Add the vertex. " << std::endl;
    std::shared_ptr<CurveFittingVertex> v = std::make_shared<CurveFittingVertex>();
    v->setId(0);
    v->setEstimate(Eigen::Vector3d(ae, be, ce));
    v->setMarginalized(true);
    if(!graph.addVertex(v)) {
        std::cout << "Failed to add vertex. " << std::endl;
        return -1;
    }

    // Add the edges
    std::cout << "Add the edges. " << std::endl;
    for (int i = 0; i < N; ++i) {
        std::shared_ptr<CurveFittingEdge> e = std::make_shared<CurveFittingEdge>(x_data[i]);
        e->setId(i);
        e->setVertex(v);
        e->setMeasurement(y_data[i]);
        e->setInformation(Eigen::MatrixXd::Identity(1, 1) * inv_sigma * inv_sigma);
        if (!graph.addEdge(e)) {
            std::cout << "Failed to add edge" << i << ". " << std::endl;
        }
    }

    // Conduct graph optimization
    std::cout << "Conduct graph optimization. " << std::endl;
    gopt::TicToc tictoc;
    tictoc.tic();
    int ret = graph.optimize();
    double time = tictoc.toc();
    if (ret >= 0) {
        std::cout << "Optimization done. " << std::endl;
        std::cout << "Estimate: " << v->getEstimate().transpose() << std::endl;
        std::cout << "Cost: " << graph.getCost() << std::endl;
        std::cout << "Time: " << time << " sec(s) " << std::endl;
    }
    else {
        std::cout << "Optimization failed. Return value: " << ret << std::endl;
    }

    return 0;
}