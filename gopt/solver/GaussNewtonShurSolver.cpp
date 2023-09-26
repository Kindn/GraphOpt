/*
 * filename: GaussNewtonShurSolver.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "solver/GaussNewtonShurSolver.h"

namespace gopt {

void GaussNewtonShurSolver::setGraph(FactorGraph *graph) {
    OptSolverBase::setGraph(graph);

    dim_res_ = graph_->getDimResidual();
    dim_var_ = graph_->getDimVariables();
    dim_marg_ = graph_->getDimMarginalized();
    dim_r_ = dim_var_ - dim_marg_;
    
    Hmm_.resize(dim_marg_, dim_marg_);
    Hrr_.resize(dim_var_ - dim_marg_, dim_var_ - dim_marg_);
    Hrm_.resize(dim_marg_, dim_var_ - dim_marg_);
    bmm_ = Eigen::VectorXd::Zero(dim_marg_);
    brr_ = Eigen::VectorXd::Zero(dim_var_ - dim_marg_);
}


int GaussNewtonShurSolver::solve(Eigen::VectorXd &delta, double &cost, Eigen::VectorXd &residual) {
    assert(graph_ != nullptr && "Graph should not be null. ");

    // Compute Jacobian
    residual.resize(dim_res_);
    cost = 0.0;
    FactorGraph::EdgeSet edges = graph_->getEdges();
    Hmm_.setZero(), Hrr_.setZero(), Hrm_.setZero();
    bmm_.setZero(), brr_.setZero();
    // std::cout << "edge number: " << edges.size() << std::endl;
    int count = 0;
    for (auto &id_edge : edges) {
        FactorGraph::EdgePtr edge = id_edge.second;
        size_t block_id = edge->getBlockId();
        Eigen::MatrixXd info = edge->getInformation();

        // Compute error
        double loss_grad = 1.0, loss_grad2 = 0.0;
        edge->computeResidual();
        edge->computeJacobians();
        double error2 = edge->computeError2();
        if (edge->loss_ != nullptr) {
            error2 = edge->loss_->operator()(std::sqrt(error2), &loss_grad, &loss_grad2);
        }
        cost += error2;
        residual.segment(block_id, edge->dimension()) = edge->getResidual();

        // Build block system
        std::cout << "Building block system " << count << std::endl;
        buildBlockSystem(edge, info, loss_grad);
        ++count;
    }
    
    // Solve the block system
    std::cout << "Solving block system ... " << std::endl;
    bool success = solveBlockSystemShur(delta);

    if (!success) {
        return 1;
    }
    else {
        return 0;
    }
}

void GaussNewtonShurSolver::buildBlockSystem(const FactorGraph::EdgePtr & edge, 
                                          const Eigen::MatrixXd &info, 
                                          double loss_grad) {
    for (int i = 0; i < edge->vertices_.size(); ++i) {
        auto vi = edge->vertices_[i];
        size_t vi_block_id = vi->getBlockId();
        if (vi->isSetFixed()) {
            continue;
        }

        for (int j = i; j < edge->vertices_.size(); ++j) {
            auto vj = edge->vertices_[j];
            size_t vj_block_id = vj->getBlockId();
            if (vj->isSetFixed()) {
                continue;
            }

            Eigen::MatrixXd hessian_block;
            if (std::max(vi_block_id, vj_block_id) >= dim_r_) {
                hessian_block = loss_grad * edge->getJacobian(i).transpose() * info * edge->getJacobian(j);
                addBlockToSparseMatrix(Hmm_, hessian_block, vi_block_id - dim_r_, vj_block_id - dim_r_);
                if (i != j ) {
                    addBlockToSparseMatrix(Hmm_, hessian_block.transpose(), vj_block_id, vi_block_id);
                }
            } else if (std::min(vi_block_id, vj_block_id) < dim_r_) {
                hessian_block = loss_grad * edge->getJacobian(i).transpose() * info * edge->getJacobian(j);
                addBlockToSparseMatrix(Hrr_, hessian_block, vi_block_id, vj_block_id);
                if (i != j ) {
                    addBlockToSparseMatrix(Hrr_, hessian_block.transpose(), vj_block_id - dim_marg_, vi_block_id - dim_marg_);
                }
            } else if (vi_block_id >= dim_r_) {
                hessian_block = loss_grad * edge->getJacobian(j).transpose() * info * edge->getJacobian(i);
                addBlockToSparseMatrix(Hrm_, hessian_block, vj_block_id, vi_block_id - dim_r_);
            } else {
                hessian_block = loss_grad * edge->getJacobian(i).transpose() * info * edge->getJacobian(j);
                addBlockToSparseMatrix(Hrm_, hessian_block, vi_block_id, vj_block_id - dim_r_);
            }
        }

        if (vi_block_id < dim_r_) {
            brr_.segment(vi_block_id, vi->localDimension()) += -loss_grad * edge->getJacobian(i).transpose() * info * edge->getResidual(); 
            // std::cout << "bmm_ " << -edge->getJacobian(i).transpose() * edge->getResidual() << std::endl;
        } else {
            bmm_.segment(vi_block_id - dim_r_, vi->localDimension()) += -loss_grad * edge->getJacobian(i).transpose() * info * edge->getResidual();
            // std::cout << "brr_ " << -edge->getJacobian(i).transpose() * edge->getResidual() << std::endl; 
        }
    }
}

bool GaussNewtonShurSolver::solveBlockSystemShur(Eigen::VectorXd &delta) {
    if (init_) {
        Hmm_Chol_.analyzePattern(Hrr_);
    }
    Hmm_Chol_.factorize(Hrr_);
    
    if (dim_marg_ > 0) {
        // SpMatType identity(dim_var_ - dim_marg_, dim_var_ - dim_marg_);
        SpMatType Hmm_inv_Hrm_T = Hmm_Chol_.solve(SpMatType(Hrm_.transpose()));
        SpMatType Hrr_Shur_ = Hrr_ - Hrm_ * Hmm_inv_Hrm_T;
        if (init_) {
            Hrr_Shur_Chol_.analyzePattern(Hrr_Shur_);
        }
        Hrr_Shur_Chol_.factorize(Hrr_Shur_);
        Eigen::VectorXd brr_Shur = brr_ - Hrm_ * Hmm_Chol_.solve(bmm_);

        delta.resize(dim_var_);
        delta.head(dim_r_) = Hrr_Shur_Chol_.solve(brr_Shur);
        delta.tail(dim_var_ -dim_r_) = Hmm_Chol_.solve(bmm_ - Hrm_.transpose() * delta.head(dim_r_));
    } else {
        delta = Hmm_Chol_.solve(brr_);
    }

    if (init_) {
        init_ = false;
    }

    if (std::isnan(delta(0)) || std::isnan(delta(dim_r_))) {
        return false;
    } else {
        return true;
    }
}

}
