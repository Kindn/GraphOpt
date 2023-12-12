add_library(${LIB_NAME}_solver SHARED 
    OptSolverBase.h 
    GaussNewtonSolver.cpp GaussNewtonSolver.h
    GaussNewtonShurSolver.cpp GaussNewtonShurSolver.h
    GaussNewtonSparseShurSolver.cpp GaussNewtonSparseShurSolver.h
    LevenbergMarquartSparseShurSolver.cpp LevenbergMarquartSparseShurSolver.h
)

target_link_libraries(${LIB_NAME}_solver 
${LIB_NAME}_util
)