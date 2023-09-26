/*
 * filename: eigen_types.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Typedefs of some Eigen-related types.
 */

#include "util/eigen_types.h"

namespace gopt {

void addBlockToSparseMatrix(SpMatType &sp_mat, const Eigen::MatrixXd &block, 
                            size_t start_rows, size_t start_cols) {
    for (int row = 0; row < block.rows(); ++row) {
        for (int col = 0; col < block.cols(); ++col) {
            sp_mat.coeffRef(start_rows + row, start_cols + col) += block(row, col);
        }
    }
}

} // namespace gopt
