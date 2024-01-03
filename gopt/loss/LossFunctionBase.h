/*
 * filename: LossFunctionBase.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _GOPT_LOSS_FUNCTION_BASE_
#define _GOPT_LOSS_FUNCTION_BASE_

#include <memory>

#include "util/eigen_types.h"

namespace gopt {

struct LossFunctionBase : public std::enable_shared_from_this<LossFunctionBase> {
public: 
    LossFunctionBase() {}
    virtual ~LossFunctionBase() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual double operator () (double input, double *grad, double *grad2) = 0;
};

} // namespace gopt


#endif // _LOSS_FUNCTION_BASE_
