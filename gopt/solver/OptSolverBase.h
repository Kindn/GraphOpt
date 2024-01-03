/*
 * filename: OptSolverBase.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _GOPT_OPT_SOLVER_BASE_
#define _GOPT_OPT_SOLVER_BASE_

#include <memory>
#include <assert.h>
#include <typeinfo>

#include "util/eigen_types.h"
#include "graph/FactorGraph.h"

namespace gopt {

class OptSolverBase : public std::enable_shared_from_this<OptSolverBase> {
public: 
    OptSolverBase() {}
    virtual ~OptSolverBase() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

public: 
    virtual int solve(Eigen::VectorXd &delta, double &cost, Eigen::VectorXd &residual) = 0;

    virtual void setGraph(FactorGraph *graph) {
        assert(graph != nullptr && "Graph should not be null. ");
        graph_ = graph;
    }

    virtual void setInit() { init_ = true; }

protected: 
    FactorGraph *graph_;
    bool init_{true};
};

} // namespace gopt

#endif // _OPT_SOLVER_BASE_
