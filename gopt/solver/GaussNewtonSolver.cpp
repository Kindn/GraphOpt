/*
 * filename: GaussNewtonSolver.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "solver/GaussNewtonSolver.h"
#include <chrono>

namespace gopt {

int GaussNewtonSolver::solve(Eigen::VectorXd &delta, double &cost, Eigen::VectorXd &residual) {
    assert(graph_ != nullptr && "Graph should not be null. ");

    size_t dim_res = graph_->getDimResidual();
    size_t dim_var = graph_->getDimVariables();
    
    // Compute Jacobian
    Eigen::MatrixXd jacobian(dim_res, dim_var);
    jacobian.setZero();
    residual.resize(dim_res);
    cost = 0.0;
    FactorGraph::EdgeSet edges = graph_->getEdges();
    for (auto &id_edge : edges) {
        FactorGraph::EdgePtr edge = id_edge.second;
        size_t block_id = edge->getBlockId();
        Eigen::MatrixXd info = edge->getInformation();

        double loss_grad = 1.0, loss_grad2 = 0.0;
        edge->computeResidual();
        edge->computeJacobians();
        double error = edge->computeError2();
        if (edge->loss_ != nullptr) {
            error = edge->loss_->operator()(error, &loss_grad, &loss_grad2);
        }
        cost += error;
        residual.segment(block_id, edge->dimension()) = edge->getResidual();
        for (size_t i = 0; i < edge->vertices_.size(); ++i) {
            if (!edge->vertices_[i]->isSetFixed()) {
                jacobian.block(block_id, edge->vertices_[i]->getBlockId(), 
                           edge->dimension(), edge->vertices_[i]->localDimension()) += loss_grad * info * edge->getJacobian(i);
            }
        }
    }
    
    // Compute H and b
    std::cout << jacobian.rows() << " " << jacobian.cols() << std::endl;
    Eigen::MatrixXd H = jacobian.transpose() * jacobian;
    Eigen::MatrixXd b = -jacobian.transpose() * residual;

    // Solve H * delta = b
    delta = H.ldlt().solve(b);
    // std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // std::cout << elapsed_seconds.count() * 1000 << std::endl;
    if (std::isnan(delta(0))) {
        return 1;
    }
    else {
        return 0;
    }
}

} // namespace gopt