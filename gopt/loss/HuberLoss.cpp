/*
 * filename: HuberLoss.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "loss/HuberLoss.h"

namespace gopt {

double HuberLoss::operator () (double input, double *grad, double *grad2) {
    double output = std::abs(input) <= delta_ ? 
                        input * input / 2.0 : 
                        delta_ * (std::abs(input) - delta_ / 2.0);
    if (grad) {
        *grad = std::abs(input) <= delta_ ? 
               input : 
               input / std::abs(input) * delta_;
    }
    
    return output;
}

}