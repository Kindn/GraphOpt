/*
 * filename: BaseVertex.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _GOPT_BASE_VERTEX_H_
#define _GOPT_BASE_VERTEX_H_

#include <stack>

#include "graph/FactorGraph.h"

namespace gopt {

template <size_t LocalDim, typename EstType>
struct BaseVertex : public FactorGraph::Vertex, public std::enable_shared_from_this<BaseVertex<LocalDim, EstType>> {
public: 
    typedef std::stack<EstType> BackupStackType;

    BaseVertex() : 
    Vertex() {
        dimension_ = LocalDim;
        local_dimension_ = LocalDim;
    }
    virtual ~BaseVertex() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    
    virtual EstType getEstimate() const { return estimate_; }

    virtual void setEstimate(const EstType &est) { estimate_ = est; }

    virtual void push() override {
        backup_stack_.push(estimate_);
    }

    virtual void pop() override {
        assert(!backup_stack_.empty() && "Back-up stack should not be empty. ");
        estimate_ = backup_stack_.top();
        backup_stack_.pop();
    }

protected:
    EstType estimate_;
    BackupStackType backup_stack_;
};

} // namespace gopt

#endif // _BASE_VERTEX_H_
