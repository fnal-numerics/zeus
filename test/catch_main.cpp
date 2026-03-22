#include <catch2/catch_session.hpp>

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

  bool
  configure_ctest_gpu_runtime()
  {
    const char* resource = std::getenv("CTEST_RESOURCE_GROUP_0_GPUS");
    if (resource == nullptr) {
      return true;
    }

    const char* id_pos = std::strstr(resource, "id:");
    if (id_pos == nullptr) {
      return true;
    }

    id_pos += 3;
    const char* id_end = std::strpbrk(id_pos, ",;");
    const std::string gpu_id = id_end == nullptr ?
                                 std::string(id_pos) :
                                 std::string(id_pos, id_end - id_pos);

    if (gpu_id.empty()) {
      return true;
    }

    char* parse_end = nullptr;
    const long ordinal = std::strtol(gpu_id.c_str(), &parse_end, 10);
    if (parse_end == gpu_id.c_str() || *parse_end != '\0' || ordinal < 0) {
      std::cerr << "Unable to parse GPU id from CTest resource allocation: "
                << resource << std::endl;
      return false;
    }

    const cudaError_t status = cudaSetDevice(static_cast<int>(ordinal));
    if (status != cudaSuccess) {
      std::cerr << "Failed to bind to CTest-assigned GPU " << gpu_id << ": "
                << cudaGetErrorString(status) << std::endl;
      return false;
    }

    return true;
  }

} // namespace

int
main(int argc, char* argv[])
{
  if (!configure_ctest_gpu_runtime()) {
    return 2;
  }

  return Catch::Session().run(argc, argv);
}
