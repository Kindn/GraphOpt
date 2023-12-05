add_library(${LIB_NAME}_solver SHARED 
    OptSolverBase.h 
    GaussNewtonSolver.cpp GaussNewtonSolver.h
    GaussNewtonShurSolver.cpp GaussNewtonShurSolver.h
    GaussNewtonSparseSolver.cpp GaussNewtonSparseSolver.h
    LevenbergMarquartSparseSolver.cpp LevenbergMarquartSparseSolver.h
)

target_link_libraries(${LIB_NAME}_solver 
${LIB_NAME}_util
)