/*
 * filename: BaseBinaryEdge.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _GOPT_BASE_BINARY_EDGE_
#define _GOPT_BASE_BINARY_EDGE_

#include "graph/BaseEdge.h"

namespace gopt {

template <size_t Dim, typename MeasurementType, typename VertexType1, typename VertexType2>
struct BaseBinaryEdge : public BaseEdge<Dim, MeasurementType>, public std::enable_shared_from_this<BaseBinaryEdge<Dim, MeasurementType, VertexType1, VertexType2>> {
public: 
    BaseBinaryEdge(): 
    BaseEdge<Dim, MeasurementType>() {
        vertices_.resize(2);
        jacobians_.resize(2);
    }
    virtual ~BaseBinaryEdge() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using FactorGraph::Edge::vertices_;
    using FactorGraph::Edge::jacobians_;
};

} // namespace gopt

#endif // _BASE_BINARY_EDGE_
