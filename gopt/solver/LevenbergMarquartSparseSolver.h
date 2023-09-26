/*
 * filename: LevenbergMarquartSparseSolver.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include <unordered_map>

#include "solver/OptSolverBase.h"

#ifndef _LEVENBERG_MARQUART_SPARSE_SOLVER_
#define _LEVENBERG_MARQUART_SPARSE_SOLVER_

namespace gopt {

/**
 * @brief This solver consider the sparse structure of the Hessian in traditionnal 
 * Bundle-Adjustment problem. All the edges in the graph should be binary edge, and
 * each of them should connect a marginalized vertex and a in-marginalized vertex.
*/
class LevenbergMarquartSparseSolver : public OptSolverBase, public std::enable_shared_from_this<LevenbergMarquartSparseSolver> {
    friend class FactorGraph;
public: 
    LevenbergMarquartSparseSolver(): 
    OptSolverBase() {}
    virtual ~LevenbergMarquartSparseSolver() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setGraph(FactorGraph *graph) override;

public: 
    virtual int solve(Eigen::VectorXd &delta, double &cost, Eigen::VectorXd &residual) override;

    void buildBlockSystem(const FactorGraph::EdgePtr & edge, 
                            const Eigen::MatrixXd &info, 
                            double loss_grad);

    bool solveBlockSystemShur(Eigen::VectorXd &delta);

    void fillSparseMatrices();

    void setLambda(double lambda) { lambda_ = lambda; }

    void updateLambdaAndNu(double cost, const Eigen::VectorXd &delta);

    double computeNewCost(const Eigen::VectorXd &delta);

    void computeInitLambda();

protected: 
    std::unordered_map<size_t, Eigen::MatrixXd> Hmm_blocks_, Hmm_inv_blocks_;
    std::unordered_map<size_t, Eigen::MatrixXd> Hrr_blocks_;
    std::unordered_map<size_t, std::unordered_map<size_t, Eigen::MatrixXd>> Hrm_blocks_;
    SpMatType Hrr_, Hrr_Shur_;
    SpMatType Hmm_inv_;
    SpMatType Hrm_;
    SpMatType Irr_, Imm_;
    Eigen::SimplicialCholesky<SpMatType> Hrr_Shur_Chol_;
    Eigen::VectorXd bmm_;
    Eigen::VectorXd brr_, brr_Shur_;
    size_t dim_res_;
    size_t dim_var_;
    size_t dim_marg_;
    size_t num_marg_;
    size_t dim_r_;
    size_t num_r_;
    double lambda_, nu_{2.0};
    Eigen::VectorXd last_delta_;
    Eigen::VectorXd residual_;
    // double last_cost_;
    double rho_;
    bool update_flag_{true};
    double lambda_scale_upper_{2. / 3.};
    double lambda_scale_lower_{1. / 3.};
    // VecMatrixXd informations_;
    // std::vector<double> loss_grads_;
};

}

#endif // _LEVENBERG_MARQUART_SPARSE_SOLVER_H_

