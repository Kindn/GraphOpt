add_library(${LIB_NAME}_graph SHARED
    FactorGraph.cpp FactorGraph.h 
    BaseVertex.h
    BaseEdge.h
    BaseBinaryEdge.h 
    BaseUnaryEdge.h 
)

target_link_libraries(${LIB_NAME}_graph 
${LIB_NAME}_solver 
${LIB_NAME}_util
)