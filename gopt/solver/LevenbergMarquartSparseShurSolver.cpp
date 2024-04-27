/*
 * filename: LevenbergMarquartSparseShurSolver.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "solver/LevenbergMarquartSparseShurSolver.h"

namespace gopt {

void LevenbergMarquartSparseShurSolver::setGraph(FactorGraph *graph) {
    OptSolverBase::setGraph(graph);

    dim_res_ = graph_->getDimResidual();
    dim_var_ = graph_->getDimVariables();
    dim_marg_ = graph_->getDimMarginalized();
    num_marg_ = graph_->getNumMarginalizedVertex();
    dim_r_ = dim_var_ - dim_marg_;
    // num_r_ = graph_->getNumUnfixed() - num_marg_;

     std::cout << "[" << typeid(*this).name() << "]" 
               << "dim_r = " << dim_r_ << " " 
              << "dim_var = " << dim_var_ << " "
              << "dim_marg = " << dim_marg_ << " " 
              << "num_unfixed = " << graph_->getNumUnfixed() << std::endl;

    Hmm_blocks_.reserve(dim_marg_);
    Hrr_blocks_.reserve(dim_r_);
    Hrm_blocks_.reserve(dim_r_);
    Hrr_.resize(dim_r_, dim_r_);
    Hrr_Shur_.resize(dim_r_, dim_r_);
    Hmm_inv_.resize(dim_marg_, dim_marg_);
    Hrm_.resize(dim_r_, dim_marg_);
    Irr_.resize(dim_r_, dim_r_), Irr_.setIdentity();
    Imm_.resize(dim_marg_, dim_marg_), Imm_.setIdentity();
    bmm_ = Eigen::VectorXd::Zero(dim_marg_);
    brr_ = Eigen::VectorXd::Zero(dim_r_);
}


int LevenbergMarquartSparseShurSolver::solve(Eigen::VectorXd &delta, double &cost, Eigen::VectorXd &residual) {
    assert(graph_ != nullptr && "Graph should not be null. ");

    // Compute Jacobian
    residual.resize(dim_res_);
    cost = 0.0;
    FactorGraph::EdgeSet edges = graph_->getEdges();
    
    Hmm_blocks_.clear(), Hrr_blocks_.clear(), Hrm_blocks_.clear();
    Hmm_blocks_.reserve(num_marg_);
    Hrr_blocks_.reserve(graph_->getNumVertices() - num_marg_);
    Hrm_blocks_.reserve(graph_->getNumVertices() - num_marg_);
    bmm_.setZero(), brr_.setZero();
    Hrr_.setZero(), Hmm_inv_.setZero(), Hrm_.setZero();
    // std::cout << "edge number: " << edges.size() << std::endl;
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
        buildBlockSystem(edge, info, loss_grad);
    }

    if (init_) {
        computeInitLambda();
        nu_ = 2.0;
        std::cout << __PRETTY_FUNCTION__ << ": initial lambda: " << lambda_ << std::endl;
    }

    update_flag_ = false;
    bool success;
    do {
        // Solve the block system
        std::cout << "Solving block system ... lambda = " << lambda_ << ", rho = " << rho_ << std::endl;
        success = solveBlockSystemShur(delta);
        // update lamda_ and nu_
        updateLambdaAndNu(cost, delta);
        if (!std::isfinite(lambda_)) {
            break;
        }
    } while (!update_flag_);

    if (!success) {
        return 1;
    }
    else {
        return 0;
    }
}

void LevenbergMarquartSparseShurSolver::buildBlockSystem(const FactorGraph::EdgePtr & edge, 
                                          const Eigen::MatrixXd &info, 
                                          double loss_grad) {
    for (size_t i = 0; i < edge->vertices_.size(); ++i) {
        auto vi = edge->vertices_[i];
        const size_t vi_block_id = vi->getBlockId();
        // if (vi->isSetFixed()) {
        //     continue;
        // }

        for (size_t j = i; j < edge->vertices_.size(); ++j) {
            auto vj = edge->vertices_[j];
            const size_t vj_block_id = vj->getBlockId();
            // if (vj->isSetFixed()) {
            //     continue;
            // }

            // std::cout << edge->getJacobian(i).rows() << " " << edge->getJacobian(i).cols() << " " 
            //           << info.rows() << " " << info.cols() << " " 
            //           << edge->getJacobian(j).rows() << " " << edge->getJacobian(j).cols() << std::endl; 
            Eigen::MatrixXd hessian_block = 
                loss_grad * edge->getJacobian(i).transpose() * info * edge->getJacobian(j);
            if (i == j) {
                if (vi_block_id < dim_r_) {
                    auto it = Hrr_blocks_.find(vi_block_id);
                    if (it == Hrr_blocks_.end()) {
                        Hrr_blocks_[vi_block_id] = hessian_block;
                    }
                    else {
                        // std::cout << it->second.rows() << " " << it->second.cols() << " " 
                        //           << hessian_block.rows() << " " << hessian_block.cols() << std::endl; 
                        it->second += hessian_block;
                    }
                } else {
                    auto it = Hmm_blocks_.find(vi_block_id - dim_r_);
                    if (it == Hmm_blocks_.end()) {
                        Hmm_blocks_[vi_block_id - dim_r_] = hessian_block;
                    }
                    else {
                        it->second += hessian_block;
                    }
                }
            } else {
                // Only consider camera-landmark case
                if (vi_block_id >= dim_r_ && vj_block_id < dim_r_) {
                    auto it_r = Hrm_blocks_.find(vj_block_id);
                    if (it_r == Hrm_blocks_.end()) {
                        Hrm_blocks_[vj_block_id].reserve(num_marg_);
                        Hrm_blocks_[vj_block_id][vi_block_id - dim_r_] = hessian_block.transpose();
                    } else {
                        auto it_m = Hrm_blocks_[vj_block_id].find(vi_block_id - dim_r_);
                        if (it_m == Hrm_blocks_[vj_block_id].end()) {
                            Hrm_blocks_[vj_block_id][vi_block_id - dim_r_] = hessian_block.transpose();
                        } else {
                            it_m->second += hessian_block.transpose();
                        }
                    }
                } else if (vj_block_id >= dim_r_ && vi_block_id < dim_r_) {
                    auto it_r = Hrm_blocks_.find(vi_block_id);
                    if (it_r == Hrm_blocks_.end()) {
                        Hrm_blocks_[vi_block_id].reserve(num_marg_);
                        Hrm_blocks_[vi_block_id][vj_block_id - dim_r_] = hessian_block;
                    } else {
                        auto it_m = Hrm_blocks_[vi_block_id].find(vj_block_id - dim_r_);
                        if (it_m == Hrm_blocks_[vi_block_id].end()) {
                            Hrm_blocks_[vi_block_id][vj_block_id - dim_r_] = hessian_block;
                        } else {
                            it_m->second += hessian_block;
                        }
                    }
                }
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

bool LevenbergMarquartSparseShurSolver::solveBlockSystemShur(Eigen::VectorXd &delta) {
    // Compute the inverse of Hrr
    Hmm_inv_blocks_.reserve(Hmm_blocks_.size());
    for (auto &block : Hmm_blocks_) {
        Hmm_inv_blocks_[block.first] = 
            (block.second + lambda_ * Eigen::MatrixXd::Identity(block.second.rows(), block.second.rows())).inverse();
    }

    // Compute Shur complement
    fillSparseMatrices();
    delta.resize(dim_var_);
    Eigen::VectorXd b;
    // Solve for delta
    if (dim_marg_ > 0) {
        if (dim_r_ > 0) {
            Hrr_Shur_ = Hrr_ + lambda_ * Irr_ - Hrm_ * Hmm_inv_ * SpMatType(Hrm_.transpose());
            brr_Shur_ = brr_ - Hrm_ * Hmm_inv_ * bmm_;
            if (init_) {
                Hrr_Shur_Chol_.analyzePattern(Hrr_Shur_);
            }
            Hrr_Shur_Chol_.factorize(Hrr_Shur_);
            // Solve for delta_r
            delta.head(dim_r_) = Hrr_Shur_Chol_.solve(brr_Shur_);
            b = bmm_ - SpMatType(Hrm_.transpose()) * delta.head(dim_r_);

            if (std::isnan(delta(0))) {
                return false;
            }
        } else {
            b = bmm_;
        }
        // Solve for delta_m
        for (auto &block : Hmm_inv_blocks_) {
            delta.segment(dim_r_ + block.first, block.second.rows()) = 
                block.second * b.segment(block.first, block.second.cols());
        }

        if (std::isnan(delta(dim_r_))) {
            return false;
        }
        
    } else if (dim_r_ > 0) {
        Hrr_Shur_ = Hrr_;
        brr_Shur_ = brr_;
        if (init_) {
            Hrr_Shur_Chol_.analyzePattern(Hrr_Shur_);
        }
        Hrr_Shur_Chol_.factorize(Hrr_Shur_);
        delta = Hrr_Shur_Chol_.solve(brr_Shur_);

        // std::cout << "dim_r = " << dim_r_ << " " 
        //       << "dim_var = " << dim_var_ << " "
        //       << "dim_marg = " << dim_marg_ << " " 
        //       << "num_unfixed = " << graph_->getNumUnfixed() << std::endl;

        if (std::isnan(delta(0))) {
            return false;
        }
    }

    if (init_) {
        init_ = false;
    }

    return true;
}

void LevenbergMarquartSparseShurSolver::fillSparseMatrices() {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(dim_r_);
    for (typename std::unordered_map<size_t, Eigen::MatrixXd>::const_iterator it = Hrr_blocks_.begin(); 
         it != Hrr_blocks_.end(); it++) {
        for (int r = 0; r < it->second.rows(); ++r) {
            for (int c = 0; c < it->second.cols(); ++c) {
                triplets.emplace_back(r + it->first, c + it->first, it->second(r, c));
            }
        }
    }
    Hrr_.setFromTriplets(triplets.begin(), triplets.end());
    
    triplets.clear();
    triplets.reserve(dim_marg_);
    for (typename std::unordered_map<size_t, Eigen::MatrixXd>::const_iterator it = Hmm_inv_blocks_.begin(); 
         it != Hmm_inv_blocks_.end(); it++) {
        for (int r = 0; r < it->second.rows(); ++r) {
            for (int c = 0; c < it->second.cols(); ++c) {
                triplets.emplace_back(r + it->first, c + it->first, it->second(r, c));
            }
        }
    }
    Hmm_inv_.setFromTriplets(triplets.begin(), triplets.end());

    triplets.clear();
    triplets.reserve(dim_r_);
    for (typename std::unordered_map<size_t, std::unordered_map<size_t, Eigen::MatrixXd>>::const_iterator it_r = Hrm_blocks_.begin(); 
         it_r != Hrm_blocks_.end(); it_r++) {
        for (typename std::unordered_map<size_t, Eigen::MatrixXd>::const_iterator it_c = it_r->second.begin(); 
             it_c != it_r->second.end(); it_c++) {
            for (int r = 0; r < it_c->second.rows(); ++r) {
                for (int c = 0; c < it_c->second.cols(); ++c) {
                    triplets.emplace_back(r + it_r->first, c + it_c->first, it_c->second(r, c));
                }
            }
        }
    }
    Hrm_.setFromTriplets(triplets.begin(), triplets.end());
}

void LevenbergMarquartSparseShurSolver::updateLambdaAndNu(double cost, const Eigen::VectorXd &delta) {
    if (true) {
        double new_cost = computeNewCost(delta);
        double delta_cost = new_cost - cost;
        double approx_delta_cost = 
            delta.head(dim_var_ - dim_marg_).dot(-brr_) + delta.tail(dim_marg_).dot(-bmm_) - 
            lambda_ * delta.norm();
        approx_delta_cost += 1e-6;

        rho_ = delta_cost / approx_delta_cost;

        if (rho_ > 0 && std::isfinite(new_cost)) {
            update_flag_ = true;
            double scale = std::min(lambda_scale_upper_, 1.0 - std::pow(2.0 * rho_ - 1.0, 3));
            lambda_ *= std::max(lambda_scale_lower_, scale);
            nu_ = 2.0;
        } else {
            update_flag_ = false;
            lambda_ *= nu_;
            nu_ *= 2.0;
        }
    }
}

double LevenbergMarquartSparseShurSolver::computeNewCost(const Eigen::VectorXd &delta) {
    FactorGraph::EdgeSet edges = graph_->getEdges();
    FactorGraph::VertexSet vertices = graph_->getVertices();

    for (auto &v : vertices) {
        if (!v.second->isSetFixed()) {
            v.second->push();
            v.second->plus(delta.segment(v.second->getBlockId(), v.second->localDimension()));
        }
    }

    double new_cost = 0.0;
    for (auto &e : edges) {
        e.second->computeResidual();
        double error2 = e.second->computeError2();
        if (e.second->getLossFunction() != nullptr) {
            error2 = e.second->getLossFunction()->operator()(std::sqrt(error2), nullptr, nullptr);
        }
        new_cost += error2;
    }

    for (auto &v : vertices) {
        if (!v.second->isSetFixed()) {
            v.second->pop();
        }
    }

    return new_cost;
}

void LevenbergMarquartSparseShurSolver::computeInitLambda() {
    lambda_ = -std::numeric_limits<double>::infinity();
    for (auto &block : Hrr_blocks_) {
        for (int i = 0; i < block.second.rows(); ++i) {
            lambda_ = std::max(block.second.operator()(i, i), lambda_);
        }
    }
    for (auto &block : Hmm_blocks_) {
        for (int i = 0; i < block.second.rows(); ++i) {
            lambda_ = std::max(block.second.operator()(i, i), lambda_);
        }
    }
    lambda_ *= 1e-5;
    assert(lambda_ > -std::numeric_limits<double>::infinity());
}

} // namespace gopt
