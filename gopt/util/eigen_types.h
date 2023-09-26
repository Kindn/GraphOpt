/*
 * filename: eigen_types.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Typedefs of some Eigen-related types.
 */

#ifndef _EIGEN_TYPES_H_
#define _EIGEN_TYPES_H_

#include <eigen3/Eigen/Eigen>
#include <vector>
#include <map>

namespace gopt {

typedef std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> VecVectorXd;
typedef std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> VecMatrixXd;

typedef Eigen::SparseMatrix<double> SpMatType;

void addBlockToSparseMatrix(SpMatType &sp_mat, const Eigen::MatrixXd &block, 
                            size_t start_rows, size_t start_cols);

}

#endif // _EIGEN_TYPES_H_
