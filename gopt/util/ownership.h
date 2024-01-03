#ifndef _GOPT_OWNERSHIP_H_
#define _GOPT_OWNERSHIP_H_

namespace gopt {

template <typename T> 
void release(T *obj) {
    (void)obj; 
    delete obj; 
}

}

#endif // _GOPT_OWNERSHIP_H_