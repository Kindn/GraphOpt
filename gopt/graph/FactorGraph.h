/*
 * filename: FactorGraph.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _GOPT_FACTOR_GRAPH_H_
#define _GOPT_FACTOR_GRAPH_H_

#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <memory>
#include <assert.h>

#include "util/eigen_types.h"
#include "util/ownership.h"
#include "loss/LossFunctionBase.h"

namespace gopt {

// forward declaration
class OptSolverBase;

struct optimization_config_t {
    size_t max_iteration_num{10};
    double epsilon{1e-5};
    bool verbose{false};
};

typedef enum {
    CONVERGED = 0, 
    REACH_ITERATION_LIMIT = 1, 
    FAILED_TO_SOLVE_DELTA = -1, 
    WRONG_DELTA = -2
} optimization_state_t;

class FactorGraph : public std::enable_shared_from_this<FactorGraph> {
public:
    class Vertex;
    class Edge;

    typedef std::shared_ptr<Vertex> VertexPtr;
    typedef std::shared_ptr<Edge> EdgePtr;

    typedef std::unordered_map<size_t, VertexPtr> VertexSet; // (ordered_id, v)
    typedef std::unordered_map<size_t, EdgePtr> EdgeSet; // (ordered_id, e)
    // typedef std::vector<VertexPtr> VertexSet;
    // typedef std::vector<EdgePtr> EdgeSet;

public: 
    FactorGraph() {};
    ~FactorGraph() {
        if (solver_ != nullptr) {
            release(solver_);
        }
    };

    FactorGraph(const FactorGraph&) = delete;
    FactorGraph &operator = (const FactorGraph&) = delete;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * @brief Abstract base class of vertex.
    */
    struct Vertex {
    public: 
        friend class FactorGraph;
        Vertex() :
        id_(0), block_id_(0) {};
        virtual ~Vertex() {};

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    public: 
        void setId(size_t id) { id_ = id; }

        size_t getId() const { return id_; }
        
        void setBlockId(size_t block_id) { block_id_ = block_id; }

        size_t getBlockId() const { return block_id_; }

        size_t dimension() const { return dimension_; }

        size_t localDimension() const { return local_dimension_; }

        void setMarginalized(bool flag) { set_marginalized_ = flag; }

        bool isSetMarginalized() const { return set_marginalized_; }

        void setFixed(bool flag) { set_fixed_ = flag; }

        bool isSetFixed() const { return set_fixed_; }

        /**
         * @brief Local plus;
        */
        virtual void plus(const Eigen::VectorXd &update) = 0;

        /**
         * @brief Set to origin.
        */
        virtual void setToOrigin() = 0;

        /**
         * @brief Add current state to back-up stack.
        */
        virtual void push() = 0;

        /**
         * @brief Recover last state
        */
        virtual void pop() = 0;

    protected:
        size_t id_;
        size_t block_id_;
        size_t dimension_;
        size_t local_dimension_;
        FactorGraph *graph_;
        bool set_marginalized_{false};
        bool set_fixed_{false};
    };

    /**
     * @brief Abstract base class of edge.
    */
    struct Edge {
    public:
        friend class FactorGraph;
        Edge() :
        id_(0), block_id_(0) {}
        virtual ~Edge() {}

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    public:
        void setVertex(size_t i, const VertexPtr &vertex) {
            assert(i < vertices_.size() && "Index out of bound. ");
            vertices_[i] = vertex;
        }

        VertexPtr getVertex(size_t i) const {
            assert(i < vertices_.size() && "Index out of bound. ");

            return vertices_[i];
        }

        void setId(size_t id) { id_ = id; }

        size_t getId() const { return id_; }
        
        void setBlockId(size_t block_id) { block_id_ = block_id; }

        size_t getBlockId() const { return block_id_; }

        size_t dimension() const { return dimension_; }
    
    public:
        /**
         * @brief This function should be overrided. It compute the residual.
         * 
        */
        virtual void computeResidual() = 0;

        virtual void computeJacobians();
        
        virtual Eigen::MatrixXd getJacobian(size_t i) const;

        // virtual Eigen::MatrixXd *getJacobianPtr(size_t i) const;

        virtual void setInformation(const Eigen::MatrixXd &info);

        virtual Eigen::MatrixXd getInformation() const;

        virtual double computeError2() { return residual_.dot(information_ * residual_); }

        virtual Eigen::VectorXd getResidual() const { return residual_; }

        virtual void setLossFunction(const std::shared_ptr<LossFunctionBase> &loss) { loss_ = loss; }

        virtual std::shared_ptr<LossFunctionBase> getLossFunction() const { return loss_; }

    public: 
        std::vector<VertexPtr> vertices_;
        /* Loss function. */
        std::shared_ptr<LossFunctionBase> loss_{nullptr};

    protected: 
        size_t id_;
        size_t block_id_;
        size_t dimension_;
        FactorGraph *graph_;
        Eigen::VectorXd residual_;
        VecMatrixXd jacobians_;
        Eigen::MatrixXd information_;

        bool info_set_{false};
    };

    virtual bool addVertex(const VertexPtr &vertex);
    virtual bool addEdge(const EdgePtr &edge);

    VertexSet getVertices() const {
        return vertices_;
    }

    size_t getNumVertices() const {
        return vertices_.size();
    }

    EdgeSet getEdges() const {
        return edges_;
    }

    size_t getNumEdges() const {
        return edges_.size();
    }

    /**
     * @brief Do the optimization.
    */
    virtual int optimize();

    /**
     * @brief Set the optimization algorithm for the graph optimization problem 
     * 
     * @param solver    the solver pointer pointing to dynamically allocated memory. GOPT will 
     *                  help you to manage the memory you allocate 
    */
    void setOptSolver(OptSolverBase *solver) {
        assert(solver != nullptr && "Solver should not be null. ");
        solver_ = solver;
    }

    size_t getDimResidual() const {
        return dim_res_;
    }

    size_t getDimVariables() const {
        return dim_var_;
    }

    size_t getDimMarginalized() const {
        return dim_marg_;
    }

    size_t getNumMarginalizedVertex() const {
        return num_marg_;
    }

    size_t getNumUnfixed() const {
        return num_unfixed_;
    }

    void setOptConfig(const optimization_config_t &config) { opt_config_ = config; }

    double getCost() const { return cost_; }

protected:
    void sortVerticesAndEdges();

protected:
    /* Vertices of factor graph. */
    VertexSet vertices_;
    /* Edges of factor graph. */
    EdgeSet edges_;
    /* Optimization solver. */
    OptSolverBase *solver_;
    /* Total dimension of residual. */
    size_t dim_res_{0};
    /* Local dimension of variables. */
    size_t dim_var_{0};
    /* Local dimension of variables to be marginalized. */
    size_t dim_marg_{0};
    /* Number of marginalized vertices. */
    size_t num_marg_{0};
    /* Number of unfixed vertices. */
    size_t num_unfixed_{0};
    /* Configuration for optimization. */
    optimization_config_t opt_config_;
    /* Value of cost function. */
    double cost_;
};

} // namespace gopt



#endif // _FACTOR_GRAPH_H_
