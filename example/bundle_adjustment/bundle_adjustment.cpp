#include <iostream>

#include "graph/FactorGraph.h"
#include "graph/BaseBinaryEdge.h"
#include "graph/BaseVertex.h"
#include "solver/GaussNewtonSolver.h"
#include "solver/GaussNewtonShurSolver.h"
#include "solver/GaussNewtonSparseShurSolver.h"
#include "solver/LevenbergMarquartSparseShurSolver.h"
#include "loss/HuberLoss.h"
#include "util/TicToc.h"

#include "sophus/se3.hpp"
#include "common.h"

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v) {
    Eigen::Matrix3d vhat;
    vhat << 0, -v(2), v(1), 
            v(2), 0, -v(0), 
            -v(1), v(0), 0;

    return vhat;
}

struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// set from given data address
    explicit PoseAndIntrinsics(double *data_addr) {
        rotation = Sophus::SO3d::exp(Eigen::Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Eigen::Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    /// 将估计值放入内存
    void set_to(double *data_addr) {
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    Sophus::SO3d rotation;
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    double focal = 10;
    double k1 = 0, k2 = 0;
};

struct VertexPoseAndIntrinsics : gopt::BaseVertex<9, PoseAndIntrinsics> {
public: 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexPoseAndIntrinsics() {}

    virtual void setToOrigin() override {
        estimate_ = PoseAndIntrinsics();
    }

    virtual void plus(const Eigen::VectorXd &update) override {
        estimate_.rotation = Sophus::SO3d::exp(Eigen::Vector3d(update(0), update(1), update(2))) * estimate_.rotation;
        estimate_.translation += Eigen::Vector3d(update(3), update(4), update(5));
        estimate_.focal += update(6);
        estimate_.k1 += update(7);
        estimate_.k2 += update(8); 
    }

    Eigen::Vector2d project(const Eigen::Vector3d &point) {
        pc = estimate_.rotation * point + estimate_.translation;
        pc_norm = -pc / pc[2];
        r2 = pc_norm.head<2>().squaredNorm();
        distortion = 1.0 + r2 * (estimate_.k1 + estimate_.k2 * r2);
        return Eigen::Vector2d(estimate_.focal * distortion * pc_norm[0],
                               estimate_.focal * distortion * pc_norm[1]);
    }

    Eigen::Vector3d pc, pc_norm;
    double r2;
    double distortion;
};

struct VertexPoint : public gopt::BaseVertex<3, Eigen::Vector3d> {
public: 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() {}

    virtual void setToOrigin() override {
        estimate_.setZero();
    }

    virtual void plus(const Eigen::VectorXd &update) {
        estimate_ += Eigen::Vector3d(update(0), update(1), update(2));
    }
};

struct EdgeProjection : 
    public gopt::BaseBinaryEdge<2, Eigen::Vector2d, VertexPoseAndIntrinsics, VertexPoint> {
public:
    EdgeProjection() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void computeResidual() override {
        auto v0 = std::dynamic_pointer_cast<VertexPoseAndIntrinsics>(vertices_[0]);
        auto v1 = std::dynamic_pointer_cast<VertexPoint>(vertices_[1]);
        auto proj = v0->project(v1->getEstimate());
        residual_ = proj - measurement_;
        // std::cout << proj.transpose() << ", " << measurement_.transpose() << ", " << residual_.transpose() << std::endl;
    }

    // virtual void computeJacobians() override {
    //     /* Compute Jacobians */
    //     auto v0 = std::dynamic_pointer_cast<VertexPoseAndIntrinsics>(vertices_[0]);
    //     auto v1 = std::dynamic_pointer_cast<VertexPoint>(vertices_[1]);
    //     auto pose_and_intrinsics = v0->getEstimate();
    //     auto point = v1->getEstimate();
    //     double f = pose_and_intrinsics.focal;
    //     double k1 = pose_and_intrinsics.k1;
    //     double k2 = pose_and_intrinsics.k2;
    //     double distortion = v0->distortion;
    //     Eigen::Vector3d pc_norm = v0->pc_norm;

    //     Eigen::Matrix<double, 2, 3> dpcnormdpc;
    //     dpcnormdpc << -1.0 / v0->pc(2), 0, v0->pc(0) / (v0->pc(2) * v0->pc(2)), 
    //                   0, -1.0 / v0->pc(2), v0->pc(1) / (v0->pc(2) * v0->pc(2));

    //     Eigen::Matrix<double, 2, 3> dpdpc = f * distortion * dpcnormdpc + 
    //         f * (2.0 * k1 + 4.0 * k2 * v0->r2) * pc_norm.head<2>() * pc_norm.head<2>().transpose() * dpcnormdpc;

    //     // std::cout << "compute Jacobian w.r.t. PoseAndIntrisics" << std::endl;
    //     jacobians_.resize(2);
    //     jacobians_[0].resize(2, 9);
    //     Eigen::Matrix<double, 3, 6> dpcdT;
    //     dpcdT.block<3, 3>(0, 0) = -skewSymmetric(pose_and_intrinsics.rotation.matrix() * point);
    //     dpcdT.block<3, 3>(0, 3).setIdentity();
    //     jacobians_[0].block<2, 6>(0, 0) = dpdpc * dpcdT;
    //     jacobians_[0].col(6) = distortion * pc_norm.head<2>();
    //     jacobians_[0].block<2, 2>(0, 7) << f * v0->r2 * pc_norm(0), f * v0->r2 * v0->r2 * pc_norm(0),
    //                                        f * v0->r2 * pc_norm(1), f * v0->r2 * v0->r2 * pc_norm(1);        // compute Jacobian w.r.t. Point
    //     // std::cout << "compute Jacobian w.r.t. Point" << std::endl;
    //     jacobians_[1] = dpdpc * v0->getEstimate().rotation.matrix();
    //     // std::cout << v0->r2 << " " << f << " " << distortion << ", " << pc_norm.transpose() << std::endl;
    //     // std::cout << jacobians_[0] << std::endl;
    //     // std::cout << jacobians_[1] << std::endl;
    // }
};

void solveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    std::cout << "Set solver. " << std::endl;
    gopt::LevenbergMarquartSparseShurSolver solver;
    gopt::FactorGraph graph;
    graph.setOptSolver(&solver);

    std::cout << "Set opt config. " << std::endl;
    gopt::optimization_config_t config;
    config.verbose = true;
    config.max_iteration_num = 100;
    graph.setOptConfig(config);

    // Add the vertices
    std::cout << "Add the vertices. " << std::endl;
    const double *observations = bal_problem.observations();
    std::vector<std::shared_ptr<VertexPoseAndIntrinsics>> v_pi;
    std::vector<std::shared_ptr<VertexPoint>> v_p;

    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        auto v = std::make_shared<VertexPoseAndIntrinsics>();
        double *camera = cameras + camera_block_size * i;
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera));
        graph.addVertex(v);
        v_pi.push_back(v);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        auto v = std::make_shared<VertexPoint>();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Eigen::Vector3d(point[0], point[1], point[2]));
        // v->setFixed(true);
        v->setMarginalized(true);
        graph.addVertex(v);
        v_p.push_back(v);
    }
    
    std::cout << "num_cameras: " << bal_problem.num_cameras() << ", " 
              << "num_points: " << bal_problem.num_points() << std::endl;

    // Add the edges
    std::cout << "Add the edges. " << std::endl;
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        auto edge = std::make_shared<EdgeProjection>();
        edge->setId(i);
        edge->setVertex(0, v_pi[bal_problem.camera_index()[i]]);
        edge->setVertex(1, v_p[bal_problem.point_index()[i]]);
        edge->setMeasurement(Eigen::Vector2d(observations[2 * i + 0], observations[2 * i + 1]));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setLossFunction(std::make_shared<gopt::HuberLoss>());
        if (!graph.addEdge(edge)) std::cout << "Failed to add edge for observation " << i << std::endl;
    }

    // Conduct optimization
    std::cout << "Start optimization. " << std::endl;
    int ret = graph.optimize();
    if (ret >= 0) {
        std::cout << "Finish optimization. " << std::endl;
    } else {
        std::cout << "Optimization failed. Return value: " << ret << std::endl;
    }

    // set to bal problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = v_pi[i];
        auto estimate = vertex->getEstimate();
        estimate.set_to(camera);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = v_p[i];
        for (int k = 0; k < 3; ++k) point[k] = vertex->getEstimate()[k];
    }
}

int main(int argc, char **argv) {
     if (argc != 2) {
        std::cout << "usage: bundle_adjustment_g2o bal_data.txt" << std::endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("../initial.ply");
    solveBA(bal_problem);
    bal_problem.WriteToPLYFile("../final.ply");

    return 0;
}