/*
 * filename: HuberLoss.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _GOPT_HUBER_LOSS_H_
#define _GOPT_HUBER_LOSS_H_

#include <numeric>

#include "loss/LossFunctionBase.h"

namespace gopt {

struct HuberLoss : public LossFunctionBase, public std::enable_shared_from_this<HuberLoss> {
public: 
    HuberLoss(double delta = 1.) : 
    LossFunctionBase(), 
    delta_(delta) {
        assert(delta_ > std::numeric_limits<double>::epsilon() && 
               "\\delta should be > 0. ");
    }

    ~HuberLoss() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void setDelta(double delta) { 
        assert(delta > std::numeric_limits<double>::epsilon() && 
               "\\delta should be > 0. ");
        delta_ = delta; 
    }
    
    double getDelta() const { return delta_; }

    virtual double operator () (double input, double *grad, double *grad2) override;

protected:
    double delta_;
};

} // namespace gopt

#endif // _HUBER_LOSS_H_