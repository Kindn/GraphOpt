add_library(${LIB_NAME}_loss SHARED
    LossFunctionBase.h
    HuberLoss.cpp HuberLoss.h
)

target_link_libraries(${LIB_NAME}_loss  
${LIB_NAME}_util
)