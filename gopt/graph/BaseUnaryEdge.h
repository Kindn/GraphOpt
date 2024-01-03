/*
 * filename: BaseUnaryEdge.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _GOPT_BASE_UNARY_EDGE_
#define _GOPT_BASE_UNARY_EDGE_

#include "graph/BaseEdge.h"

namespace gopt {

template <size_t Dim, typename MeasurementType, typename VertexType>
struct BaseUnaryEdge : public BaseEdge<Dim, MeasurementType>, public std::enable_shared_from_this<BaseUnaryEdge<Dim, MeasurementType, VertexType>> {
public: 
    BaseUnaryEdge(): 
    BaseEdge<Dim, MeasurementType>() {
        vertices_.resize(1);
        jacobians_.resize(1);
    }
    virtual ~BaseUnaryEdge() {}
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    void setVertex(const std::shared_ptr<VertexType> &vertex) {
        FactorGraph::VertexPtr v = std::static_pointer_cast<FactorGraph::Vertex>(vertex);
        assert(v != nullptr);
        FactorGraph::Edge::setVertex(0, v);
    }  

public:
    using FactorGraph::Edge::vertices_;
    using FactorGraph::Edge::jacobians_;
};

} // namespace gopt

#endif // _BASE_UNARY_EDGE_
