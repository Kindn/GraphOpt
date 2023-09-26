add_library(solver SHARED 
    OptSolverBase.h 
    GaussNewtonSolver.cpp GaussNewtonSolver.h
    GaussNewtonShurSolver.cpp GaussNewtonShurSolver.h
    GaussNewtonSparseSolver.cpp GaussNewtonSparseSolver.h
    LevenbergMarquartSparseSolver.cpp LevenbergMarquartSparseSolver.h
)

target_link_libraries(solver 
    util
)