/*
 * filename: BaseEdge.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _GOPT_BASE_EDGE_
#define _GOPT_BASE_EDGE_

#include "graph/FactorGraph.h"

namespace gopt {

template <size_t Dim, typename MeasurementType>
struct BaseEdge : public FactorGraph::Edge, public std::enable_shared_from_this<BaseEdge<Dim, MeasurementType>> {
public: 
    BaseEdge(): 
    Edge() {
        dimension_ = Dim;
    }
    virtual ~BaseEdge() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setMeasurement(const MeasurementType &measurement) {
        measurement_ = measurement;
    }

    virtual MeasurementType getMeasurement() const {
        return measurement_;
    }

protected: 
    MeasurementType measurement_;
};

} // namespace gopt

#endif // _BASE_EDGE_
