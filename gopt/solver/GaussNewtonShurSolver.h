/*
 * filename: GaussNewtonShurSolver.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _GAUSS_NEWTON_SHUR_SOLVER_H_
#define _GAUSS_NEWTON_SPHUR_SOLVER_H_

#include "solver/OptSolverBase.h"

namespace gopt {

class GaussNewtonShurSolver : public OptSolverBase, public std::enable_shared_from_this<GaussNewtonShurSolver> {
    friend class FactorGraph;
public: 
    GaussNewtonShurSolver(): 
    OptSolverBase() {}
    virtual ~GaussNewtonShurSolver() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setGraph(FactorGraph *graph) override;

public: 
    virtual int solve(Eigen::VectorXd &delta, double &cost, Eigen::VectorXd &residual) override;

    void buildBlockSystem(const FactorGraph::EdgePtr & edge, 
                     const Eigen::MatrixXd &info, 
                     double loss_grad);

    bool solveBlockSystemShur(Eigen::VectorXd &delta);

protected: 
    SpMatType Hmm_;
    SpMatType Hrr_;
    SpMatType Hrm_;
    Eigen::VectorXd bmm_;
    Eigen::VectorXd brr_;
    size_t dim_res_;
    size_t dim_var_;
    size_t dim_marg_;
    size_t dim_r_;
    Eigen::SimplicialCholesky<SpMatType> Hrr_Shur_Chol_;
    Eigen::SimplicialCholesky<SpMatType> Hmm_Chol_;
};

}

#endif // _GAUSS_NEWTON_SHUR_SOLVER_H_
