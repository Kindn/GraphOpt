/*
 * filename: GaussNewtonSolver.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */
#ifndef _GOPT_GAUSS_NEWTON_SOLVER_
#define _GOPT_GAUSS_NEWTON_SOLVER_

#include "solver/OptSolverBase.h"

namespace gopt {

class GaussNewtonSolver : public OptSolverBase, public std::enable_shared_from_this<GaussNewtonSolver> {
    friend class FactorGraph;
public: 
    GaussNewtonSolver(): 
    OptSolverBase() {}
    virtual ~GaussNewtonSolver() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public: 
    virtual int solve(Eigen::VectorXd &delta, double &cost, Eigen::VectorXd &residual) override;
};

} // namespace gopt

#endif // _GAUSS_NEWTON_SOLVER_