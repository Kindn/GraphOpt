/*
 * filename: FactorGraph.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "solver/OptSolverBase.h"
#include "util/TicToc.h"

namespace gopt {

Eigen::MatrixXd FactorGraph::Edge::getJacobian(size_t i) const {
    assert(i < vertices_.size() && "Index out of bound. ");
    // if (vertices_[i]->set_fixed_) {
    //     return Eigen::MatrixXd::Zero(dimension_, vertices_[i]->localDimension());
    // } else {
    //     return jacobians_[i];
    // }
    return jacobians_[i];
}

// Eigen::MatrixXd *FactorGraph::Edge::getJacobianPtr(size_t i) const {
//     assert(i < vertices_.size() && "Index out of bound. ");
//     if (vertices_[i]->set_fixed_) {
//         return &Eigen::MatrixXd::Zero(dimension_, vertices_[i]->localDimension());
//     } else {
//         return &(jacobians_[i]);
//     }
// }

void FactorGraph::Edge::setInformation(const Eigen::MatrixXd &info) {
    assert(static_cast<size_t>(info.rows()) == dimension_ && static_cast<size_t>(info.cols()) == dimension_ && "Dimensions of information mismatch. "); 
    information_ = info;
}

Eigen::MatrixXd FactorGraph::Edge::getInformation() const {
    if (info_set_) {
        return information_;
    } else {
        return Eigen::MatrixXd::Identity(dimension_, dimension_);
    }
}

void FactorGraph::Edge::computeJacobians() {
    jacobians_.resize(vertices_.size());

    const double delta = static_cast<double>(1e-9);
    const double scale = 1.0 / (2.0 * delta);
    Eigen::VectorXd residual;
    for (size_t i = 0; i < jacobians_.size(); ++i) {
        // if (vertices_[i]->isSetFixed()) {
        //     continue;
        // }

        FactorGraph::VertexPtr v = vertices_[i];
        size_t dim_v = v->localDimension();
        jacobians_[i] = Eigen::MatrixXd::Zero(dimension_, dim_v);

        Eigen::VectorXd update = Eigen::VectorXd::Zero(dim_v);
        for (size_t j = 0; j < dim_v; ++j) {
            update(static_cast<Eigen::Index>(j)) = delta;
            v->push();
            v->plus(update);
            computeResidual();
            residual = residual_;
            v->pop();
            v->push();
            v->plus(-update);
            computeResidual();
            residual -= residual_;
            v->pop();
            
            jacobians_[i].col(static_cast<Eigen::Index>(j)) = scale * residual;
            update.setZero();
        }
    }
}

bool FactorGraph::addVertex(const FactorGraph::VertexPtr &vertex) {
    if (vertices_.find(vertex->getId()) == vertices_.end()) {
        auto ret = vertices_.insert(std::make_pair(vertex->getId(), vertex));
        if (ret.second) {
            dim_var_ += vertex->local_dimension_;
            if (vertex->set_marginalized_) {
                dim_marg_ += vertex->local_dimension_;
                num_marg_++;
            }
        }

        return ret.second;
    }

    return true;
}

bool FactorGraph::addEdge(const EdgePtr &edge) {
    bool success = true;
    success = success && edges_.insert(std::make_pair(edge->id_, edge)).second;
    for (auto &v : edge->vertices_) {
        addVertex(v);
    }
    
    if (success) {
        dim_res_ += edge->dimension_;

        return true;
    } else {
        return false;
    }
}

int FactorGraph::optimize() {
    assert(solver_ != nullptr && "Solver should not be null. ");

    sortVerticesAndEdges();
    solver_->setGraph(this);
    solver_->setInit();

    Eigen::VectorXd delta, residual;
    double last_cost; 
    TicToc tic_toc;
    double cum_time = 0;
    for (size_t iter = 0; iter < opt_config_.max_iteration_num; ++iter) {
        if (opt_config_.verbose) {
            tic_toc.tic();
        }
        bool ret = solver_->solve(delta, cost_, residual);

        if (opt_config_.verbose) {
            double time = tic_toc.toc();
            cum_time += time;
            std::cout << "iteration = " << iter << ", "
                      << "cost = " << cost_ << ", "
                      << "delta_norm = " << delta.norm() << ", "
                      << "time = " << time << " s, "
                      << "cum_time = " << cum_time << " s." << std::endl;
        }

        if (ret == 0) {
            if (delta.norm() < opt_config_.epsilon) {
                return optimization_state_t::CONVERGED;
            }
            // if (iter > 0 && cost_ > last_cost) {
            //     return optimization_state_t::WRONG_DELTA;
            // }
            std::cout << "delta_size " << delta.size() << std::endl;
            for (auto &vertex : vertices_) {
                VertexPtr vp = vertex.second;
                if (!vp->set_fixed_) {
                    size_t block_id = vp->block_id_;
                    vp->plus(delta.segment(block_id, vp->local_dimension_));
                    // std::cout << block_id << std::endl;
                }
            }
            last_cost = cost_;
        } else {
            return optimization_state_t::FAILED_TO_SOLVE_DELTA;
        }
    }

    return optimization_state_t::REACH_ITERATION_LIMIT;
}

void FactorGraph::sortVerticesAndEdges() {
    size_t keeped_vertex_block_id = 0;
    dim_var_ = 0;
    dim_marg_ = 0;
    num_marg_ = 0;
    num_unfixed_ = 0;

    for (auto &vertex : vertices_) {
        VertexPtr v = vertex.second;
        if (!v->set_marginalized_) {
            v->block_id_ = keeped_vertex_block_id;
            keeped_vertex_block_id += v->local_dimension_;
            dim_var_ += v->local_dimension_;
            if (!v->set_fixed_) {
                num_unfixed_++;
            }
        }
    }

    size_t marginalized_vertex_block_id = keeped_vertex_block_id;
    for (auto &vertex : vertices_) {
        VertexPtr v = vertex.second;
        if (v->set_marginalized_) {
            v->block_id_ = marginalized_vertex_block_id;
            marginalized_vertex_block_id += v->local_dimension_;
            dim_var_ += v->local_dimension_;
            dim_marg_ += v->local_dimension_;
            num_marg_++;
            if (!v->set_fixed_) {
                num_unfixed_++;
            }
        }
    }

    std::cout << "[Sort]: dim_var = " << dim_var_ << std::endl;

    dim_res_ = 0;
    size_t edge_block_id = 0;
    for (auto &edge : edges_) {
        EdgePtr e = edge.second;
        e->block_id_ = edge_block_id;
        edge_block_id += e->dimension_;
        dim_res_ += e->dimension_;
    }
}
} // namespace gopt
