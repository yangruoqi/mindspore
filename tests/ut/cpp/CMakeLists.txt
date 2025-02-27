message("build ut testcases...")

# virtual project for common include and library file path.
project(ut)

set(PROJECT_DIR "${PROJECT_SOURCE_DIR}/../../..")
if(ENABLE_DUMP_IR)
    add_compile_definitions(ENABLE_DUMP_IR)
endif()
if(ENABLE_D)
    add_compile_definitions(ENABLE_D)
endif()
if(NOT ENABLE_GLIBCXX)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

file(STRINGS "${CMAKE_SOURCE_DIR}/version.txt" MSVERSION)
add_definitions(-DMSVERSION=\"${MSVERSION}\")

#add python lib and include for all ut executables;
message("PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message("PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${MS_CCSRC_PATH})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stub/runtime/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stub/pass/)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/graphengine_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/plugin/device/cpu/kernel)
# graphengine include directories
include(${CMAKE_SOURCE_DIR}/cmake/graphengine_variables.cmake)
MESSAGE("check  ut_test ${CMAKE_BINARY_DIR}")

link_directories(${MS_CCSRC_BUILD_PATH})

if(ENABLE_MINDDATA)
    add_definitions(-D ENABLE_MINDDATA)
    link_directories(${MS_CCSRC_BUILD_PATH}/minddata/dataset)
    link_directories(${MS_CCSRC_BUILD_PATH}/minddata/mindrecord)
endif()

# fetch ut test files
file(GLOB_RECURSE UT_STUB_AND_COMMON_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ./stub/*.cc
        ./common/*.cc
        )

file(GLOB_RECURSE UT_API_AND_OPERATORS_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ./operator/*.cc
        ./vm/*.cc
        ./ps/*.cc
        ./fl/*.cc
        ./cxx_api/*.cc
        ./c_api/*.cc
        ./tbe/*.cc
        ./mindapi/*.cc
        ./runtime/graph_scheduler/*.cc
        ./runtime/device/gsm/*.cc
        )

if(ENABLE_MINDDATA)
    include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset)
    include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/kernels/image)
    file(GLOB_RECURSE UT_MINDDATA_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            ./dataset/*.cc
            ./mindrecord/*.cc
            )

    file(GLOB_RECURSE UT_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            ./core/abstract/*.cc
            ./core/utils/*.cc
            ./base/*.cc
            ./ir/dtype/*.cc
            ./device/*.cc
            ./ir/*.cc
            ./kernel/*.cc
            ./optimizer/*.cc
            ./parallel/*.cc
            ./pipeline/*.cc
            ./pre_activate/*.cc
            ./pynative/*.cc
            ./session/*.cc
            ./side_effect/*.cc
            ./transform/*.cc
            ./utils/*.cc
            ./plugin/device/cpu/hal/*.cc
            ./place/*.cc
            ./ops/test_ops_fake_quant_param.cc
            ./mindir/*.cc
            )

    if(NOT ENABLE_SECURITY)
        file(GLOB_RECURSE UT_SRCS_DEBUG RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                ./debug/*.cc)
        list(APPEND UT_API_AND_OPERATORS_SRCS ${UT_SRCS_DEBUG})
    endif()
    if(NOT ENABLE_CPU OR WIN32 OR APPLE)
        file(GLOB_RECURSE UT_DISTRIBUTED_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                ./distributed/persistent/*.cc
                ./distributed/rpc/tcp/*.cc
                ./distributed/cluster/*.cc
                ./distributed/cluster/topology/*.cc
                ./distributed/recovery/*.cc
                ./distributed/embedding_cache/*.cc)
        if(${ENABLE_RDMA} STREQUAL "ON")
            include_directories(/usr/include/umdk)
            file(GLOB_RECURSE UT_DISTRIBUTED_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./distributed/rpc/rdma/*.cc)
        endif()
        # consider which should be deleted
        list(APPEND UT_API_AND_OPERATORS_SRCS ${UT_DISTRIBUTED_SRCS})
    endif()
    if(NOT ENABLE_PYTHON)
        set(PYTHON_RELATED_SRCS
                dataset/filter_op_test.cc
                dataset/voc_op_test.cc
                dataset/manifest_op_test.cc
                dataset/sentence_piece_vocab_op_test.cc
                )
        list(REMOVE_ITEM UT_MINDDATA_SRCS ${PYTHON_RELATED_SRCS})
    endif()

    if(NOT ENABLE_ACL)
        set(ASCEND310_RELATED_SRCS
                dataset/dvpp_decode_jpeg_test.cc
                )
        list(REMOVE_ITEM UT_MINDDATA_SRCS ${ASCEND310_RELATED_SRCS})
    endif()
else()
    file(GLOB_RECURSE TEMP_UT_SRCS ./*.cc)
    foreach(OBJ ${TEMP_UT_SRCS})
        if(NOT ${OBJ} MATCHES "./dataset/" AND NOT ${OBJ} MATCHES "./mindrecord/" AND
                NOT ${OBJ} MATCHES "/ps/" AND
                NOT ${OBJ} MATCHES "/fl/" AND
                NOT ${OBJ} MATCHES "/c_api/" AND
                NOT ${OBJ} MATCHES "/cxx_api/" AND
                NOT ${OBJ} MATCHES "/operator/" AND
                NOT ${OBJ} MATCHES "/tbe/" AND
                NOT ${OBJ} MATCHES "/vm/" AND
                NOT ${OBJ} MATCHES "/mindapi/")
            list(APPEND UT_SRCS ${OBJ})
        endif()
    endforeach()
endif()

file(GLOB_RECURSE EXTEND_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        # dont remove the 4 lines above
        "../../../mindspore/ccsrc/pybind_api/*.cc"
        "../../../mindspore/ccsrc/frontend/optimizer/*.cc"
        "../../../mindspore/ccsrc/frontend/parallel/*.cc"
        "../../../mindspore/ccsrc/frontend/operator/*.cc"
        )
list(REMOVE_ITEM EXTEND_SRC_LIST
        "../../../mindspore/ccsrc/frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.cc")

file(GLOB_RECURSE MINDSPORE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore/ccsrc/debug/data_dump/dump_json_parser.cc"
        "../../../mindspore/ccsrc/debug/common.cc"
        "../../../mindspore/ccsrc/debug/utils.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hccl_adapter/all_to_all_v_calc_param.cc"
        "../../../mindspore/ccsrc/runtime/device/kernel_runtime.cc"
        "../../../mindspore/ccsrc/runtime/device/memory_manager.cc"
        "../../../mindspore/ccsrc/runtime/device/memory_scheduler.cc"
        "../../../mindspore/ccsrc/runtime/device/memory_offload_strategy.cc"
        "../../../mindspore/ccsrc/runtime/device/kernel_runtime_manager.cc"
        "../../../mindspore/ccsrc/runtime/device/kernel_info.cc"
        "../../../mindspore/ccsrc/runtime/device/launch_kernel.cc"
        "../../../mindspore/ccsrc/runtime/graph_scheduler/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/profiling/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ge_runtime/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_launch_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_launch_mul.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_launch_transdata.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_graph_kernel.cc"
        "../../../mindspore/ccsrc/runtime/device/convert_tensor_utils.cc"
        "../../../mindspore/ccsrc/runtime/device/context_extends.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_event.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_build_ascend.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_kernel_runtime.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_memory_adapter.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_memory_manager.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_device_address.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_stream_manager.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_memory_pool.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/lic_manager.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/tasksink/rtmodel_zero_copy.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_communication_group.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_device_context.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_device_res_manager.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_kernel_executor.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_graph_executor.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/common/ascend_utils.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/common/platform_info_util.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_somas.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_graph_optimization.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/hal/hardware/ms_collective_topo.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/hal/hardware/cpu_memory_pool.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/hal/device/cpu_device_address.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/hal/device/cpu_hash_table.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/optimizer/softmax_grad_fusion.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/factory/ms_factory.h"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/sparse_apply_adam_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/sparse_apply_ftrl_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/sparse_apply_lazy_adam_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/sparse_apply_proximal_adagrad_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/unique_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/unique_with_pad_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/adam_delta_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/fused_ada_factor_cpu_kernel.cc"
        "../../../mindspore/ccsrc/kernel/akg/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/akg/*.cc"
        "../../../mindspore/ccsrc/plugin/device/gpu/kernel/akg/*.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/akg/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/rts/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/hccl/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/bisheng/bisheng_kernel_build.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/bisheng/custom_bisheng_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/bisheng/bisheng_kernel_factory.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/bisheng/bisheng_kernel_metadata.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/kernel_query.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_attr_to_input_registry.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_input_to_attr_registry.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_kernel_metadata.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/rts/rt_kernel_info.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/tbe/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/*.cc"
        "../../../mindspore/ccsrc/backend/common/graph_kernel/*.cc"
        "../../../mindspore/ccsrc/common/expander/*.cc"
        "../../../mindspore/ccsrc/backend/common/session/anf_runtime_algorithm.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_session.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_auto_monad.cc"
        "../../../mindspore/ccsrc/backend/common/session/kernel_graph.cc"
        "../../../mindspore/ccsrc/backend/common/session/kernel_graph_mgr.cc"
        "../../../mindspore/ccsrc/backend/common/session/exec_order_builder.cc"
        "../../../mindspore/ccsrc/backend/common/session/session_basic.cc"
        "../../../mindspore/ccsrc/backend/common/session/executor.cc"
        "../../../mindspore/ccsrc/backend/common/session/executor_manager.cc"
        "../../../mindspore/ccsrc/backend/common/session/session_factory.cc"
        "../../../mindspore/ccsrc/backend/common/session/kernel_build_client.cc"
        "../../../mindspore/ccsrc/backend/operator/*.cc"
        "../../../mindspore/ccsrc/ps/*.cc"
        "../../../mindspore/ccsrc/fl/*.cc"
        "../../../mindspore/ccsrc/distributed/cluster/actor_route_table_service.cc"
        "../../../mindspore/ccsrc/distributed/cluster/actor_route_table_proxy.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/rpc/rpc_recv_kernel.cc"
        "../../../mindspore/ccsrc/distributed/persistent/*.cc"
        "../../../mindspore/ccsrc/distributed/rpc/tcp/*.cc"
        "../../../mindspore/ccsrc/distributed/rpc/rdma/*.cc"
        "../../../mindspore/ccsrc/distributed/cluster/topology/*.cc"
        "../../../mindspore/ccsrc/distributed/embedding_cache/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/profiler/*.cc"
        "../../../mindspore/ccsrc/debug/profiler/profiling.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.c"
        "../../../mindspore/ccsrc/kernel/kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/akg/akg_kernel_metadata.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/ascend_kernel_mod.cc"
        "../../../mindspore/ccsrc/backend/common/optimizer/helper.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ext_info_handle.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_util.cc"
        )
if(${ENABLE_RDMA} STREQUAL "OFF")
    list(REMOVE_ITEM MINDSPORE_SRC_LIST
            "../../../mindspore/ccsrc/distributed/rpc/rdma/rdma_client.cc"
            "../../../mindspore/ccsrc/distributed/rpc/rdma/rdma_server.cc")
endif()

list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/ps/util.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/create_node_helper.cc"
        "../../../mindspore/ccsrc/ps/scheduler.cc"
        "../../../mindspore/ccsrc/fl/server/kernel/sgd_kernel.cc"
        "../../../mindspore/ccsrc/fl/server/kernel/apply_momentum_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/gpu/optimizer/batch_norm_add_relu_fusion.cc"
        "../../../mindspore/ccsrc/plugin/device/gpu/optimizer/post_batch_norm_add_relu_fusion.cc"
        "../../../mindspore/ccsrc/plugin/device/gpu/optimizer/batch_norm_add_relu_grad_fusion.cc"
        "../../../mindspore/ccsrc/plugin/device/gpu/optimizer/batch_norm_relu_fusion.cc"
        "../../../mindspore/ccsrc/plugin/device/gpu/optimizer/batch_norm_relu_grad_fusion.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/tbe/tbe_kernel_compile.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/tbe/tbe_version.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/tbe/tiling/op_tiling_adapter.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/akg/akg_cpu_kernel_mod.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/akg/akg_cpu_kernel_build.cc"
        "../../../mindspore/ccsrc/runtime/graph_scheduler/rpc_node_scheduler.cc"
        "../../../mindspore/ccsrc/runtime/graph_scheduler/actor/rpc/send_actor.cc"
        "../../../mindspore/ccsrc/runtime/graph_scheduler/actor/rpc/recv_actor.cc"
        "../../../mindspore/ccsrc/runtime/graph_scheduler/actor/rpc/mux_send_actor.cc"
        "../../../mindspore/ccsrc/runtime/graph_scheduler/actor/rpc/mux_recv_actor.cc"
        "../../../mindspore/ccsrc/runtime/graph_scheduler/actor/rpc/rpc_actor.cc"
        "../../../mindspore/ccsrc/runtime/graph_scheduler/embedding_cache_scheduler.cc"
        "../../../mindspore/ccsrc/runtime/graph_scheduler/actor/embedding_cache/embedding_cache_prefetch_actor.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/profiler/parallel_strategy_profiling.cc")

if(ENABLE_SECURITY)
    list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/debug/profiler/profiling.cc"
            "../../../mindspore/ccsrc/plugin/device/ascend/hal/profiler/memory_profiling.cc"
            "../../../mindspore/ccsrc/plugin/device/ascend/hal/profiler/ascend_profiling.cc"
            "../../../mindspore/ccsrc/plugin/device/ascend/hal/profiler/options.cc"
            "../../../mindspore/ccsrc/debug/data_dump/dump_json_parser.cc")
endif()

add_library(_ut_stub_and_common_obj OBJECT ${UT_STUB_AND_COMMON_SRCS})
add_library(_ut_mindspore_obj STATIC ${MINDSPORE_SRC_LIST})
add_library(_ut_ut_obj OBJECT ${UT_SRCS} ${EXTEND_SRC_LIST})
add_library(_ut_api_and_operators_obj OBJECT ${UT_API_AND_OPERATORS_SRCS})

foreach(number RANGE 1 ${CORE_OBJECT_COUNT})
    list(APPEND CORE_OBJECT_LIST $<TARGET_OBJECTS:core_obj_${number}>)
endforeach()

set(_ut_common_objects ${CORE_OBJECT_LIST} $<TARGET_OBJECTS:_ut_stub_and_common_obj>
        $<TARGET_OBJECTS:core_proto_obj> $<TARGET_OBJECTS:mindrt_mid>
        $<TARGET_OBJECTS:common_shared_lib_obj> $<TARGET_OBJECTS:_mindspore_utils_obj>
        $<TARGET_OBJECTS:_mindspore_common_obj> $<TARGET_OBJECTS:mindspore_c_api_obj>)
set(ut_objects ${_ut_common_objects} $<TARGET_OBJECTS:_ut_ut_obj>)
set(ut_api_and_operators_objects ${_ut_common_objects} $<TARGET_OBJECTS:_ut_api_and_operators_obj>)

if(ENABLE_MINDDATA)
    set(_ut_minddata_objects $<TARGET_OBJECTS:mindrecord_obj>
            ${dataengine_submodules} $<TARGET_OBJECTS:md_log_adapter_obj>)
    set(ut_objects ${ut_objects} ${_ut_minddata_objects})
    add_library(_ut_minddata_ut_obj OBJECT ${UT_MINDDATA_SRCS})
    add_dependencies(_ut_minddata_ut_obj engine-cache-server)
    set(minddata_ut_obj ${_ut_common_objects} $<TARGET_OBJECTS:_ut_minddata_ut_obj> ${_ut_minddata_objects})
    add_executable(ut_minddata_tests ${minddata_ut_obj})
endif()
add_executable(ut_tests ${ut_objects})
add_executable(ut_api_operators_tests ${ut_api_and_operators_objects})

include_directories("${CMAKE_BINARY_DIR}/plugin/device/ascend/kernel/aicpu")
file(GLOB_RECURSE PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/proto/*.proto")
ms_protobuf_generate(PROTOSRCS PROTOHDRS ${PROTO_IN})

file(GLOB_RECURSE PROTO_DUMP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/dump/proto/*.proto")
ms_protobuf_generate(DUMP_PROTOSRCS PROTOHDRS ${PROTO_DUMP})

list(APPEND MINDSPORE_PROTO_LIST ${PROTOSRCS})
list(APPEND MINDSPORE_PROTO_LIST ${PREDICT_PROTOSRCS})
list(APPEND MINDSPORE_PROTO_LIST ${DUMP_PROTOSRCS})

if(MINDSPORE_PROTO_LIST)
    add_library(proto_input_ut STATIC ${MINDSPORE_PROTO_LIST})
    set_target_properties(proto_input_ut PROPERTIES COMPILE_FLAGS "-Wno-unused-variable -Wno-array-bounds")
endif()

add_library(backend_static STATIC
        $<TARGET_OBJECTS:_mindspore_debug_obj>
        $<TARGET_OBJECTS:_mindspore_common_debug_obj>
        $<TARGET_OBJECTS:_mindspore_debug_profiler_obj>
        $<TARGET_OBJECTS:_mindspore_ps_obj>
        $<TARGET_OBJECTS:_mindspore_distributed_obj>
        $<TARGET_OBJECTS:_mindspore_kernel_obj>
        $<TARGET_OBJECTS:_mindspore_backend_common_mem_reuse_obj>
        $<TARGET_OBJECTS:_mindspore_backend_common_optimizer_obj>
        $<TARGET_OBJECTS:_mindspore_backend_common_pass_obj>
        $<TARGET_OBJECTS:_mindspore_backend_common_session_obj>
        $<TARGET_OBJECTS:_mindspore_backend_common_somas_obj>
        $<TARGET_OBJECTS:_mindspore_backend_graph_compiler_obj>
        $<TARGET_OBJECTS:_mindspore_runtime_collective_obj>
        $<TARGET_OBJECTS:_mindspore_runtime_device_obj>
        $<TARGET_OBJECTS:_mindspore_runtime_graph_scheduler_obj>
        $<TARGET_OBJECTS:_mindspore_runtime_hardware_obj>
        $<TARGET_OBJECTS:_mindspore_runtime_pynative_obj>
        $<TARGET_OBJECTS:_mindspore_runtime_data_queue_obj>)
set(_order_sensitive_libs mindspore backend_static proto_input _ut_mindspore_obj)
target_link_libraries(ut_tests PRIVATE -Wl,--start-group ${_order_sensitive_libs} -Wl,--end-group)
target_link_libraries(ut_api_operators_tests PRIVATE -Wl,--start-group ${_order_sensitive_libs} -Wl,--end-group)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(_dl_libs mindspore::gtest mindspore::event mindspore::event_pthreads
            mindspore::event_openssl mindspore::ssl mindspore::crypto ${PYTHON_LIBRARIES} pthread util dl)
    target_link_libraries(ut_tests PRIVATE ${_dl_libs})
    target_link_libraries(ut_api_operators_tests PRIVATE ${_dl_libs})
    if(ENABLE_MINDDATA)
        set(_dl_minddata_libs mindspore::sqlite mindspore::jpeg_turbo mindspore::turbojpeg
                mindspore::opencv_core mindspore::opencv_imgcodecs mindspore::opencv_imgproc mindspore::tinyxml2
                mindspore::sentencepiece mindspore::sentencepiece_train mindspore::icuuc mindspore::icudata
                mindspore::icui18n mindspore::z)
        target_link_libraries(ut_tests PRIVATE ${_dl_minddata_libs})
        target_link_libraries(ut_api_operators_tests PRIVATE ${_dl_minddata_libs})
        target_link_libraries(ut_minddata_tests PRIVATE -Wl,--start-group ${_order_sensitive_libs} -Wl,--end-group
                ${_dl_minddata_libs} ${_dl_libs})
    endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    if(ENABLE_MINDDATA)
        target_link_libraries(ut_tests PRIVATE  mindspore::z)
        target_link_libraries(ut_minddata_tests PRIVATE  mindspore::z)
    endif()
else()
    target_link_libraries(ut_tests PRIVATE mindspore::gtest ${PYTHON_LIBRARIES})
    if(ENABLE_MINDDATA)
        target_link_libraries(ut_minddata_tests PRIVATE mindspore::gtest ${PYTHON_LIBRARIES})
    endif()
endif()
if(USE_GLOG)
    target_link_libraries(ut_tests PRIVATE mindspore::glog)
    target_link_libraries(ut_api_operators_tests PRIVATE mindspore::glog)
    if(ENABLE_MINDDATA)
        target_link_libraries(ut_minddata_tests PRIVATE mindspore::glog)
    endif()
endif()

target_link_libraries(ut_tests PRIVATE securec mindspore::grpc++ mindspore::protobuf)
target_link_libraries(ut_api_operators_tests PRIVATE securec mindspore::grpc++ mindspore::protobuf)
if(ENABLE_MINDDATA)
    target_link_libraries(ut_minddata_tests PRIVATE securec mindspore::grpc++ mindspore::protobuf)
endif()
