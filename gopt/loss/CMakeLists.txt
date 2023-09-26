add_library(loss SHARED
    LossFunctionBase.h
    HuberLoss.cpp HuberLoss.h
)

target_link_libraries(loss  
    util
)