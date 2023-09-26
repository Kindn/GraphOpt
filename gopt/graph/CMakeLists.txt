add_library(graph SHARED
    FactorGraph.cpp FactorGraph.h 
    BaseVertex.h
    BaseEdge.h
    BaseBinaryEdge.h 
    BaseUnaryEdge.h 
)

target_link_libraries(graph 
    solver 
    util
)