#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Approx;
#include "cuda_buffer.cuh"

TEST_CASE("Allocation of small buffer succeeds", "[cuda_buffer]") {
    cuda_buffer buf(4);
    REQUIRE(buf.size() == 4);
    REQUIRE(buf.data() != nullptr);
}

TEST_CASE("Constructor from host array + copy_to_host round-trips", "[cuda_buffer]") {
    std::array<double,3> host = {{1.1,2.2,3.3}};
    cuda_buffer buf(host);
    auto out = buf.copy_to_host();
    REQUIRE(out == std::vector<double>(host.begin(), host.end()));
}

