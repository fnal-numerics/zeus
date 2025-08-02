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

TEST_CASE("copy_to_host(vector&) overload", "[cuda_buffer]") {
    std::array<double,2> host = {{4.4,5.5}};
    cuda_buffer buf(host);
    std::vector<double> out;
    int status = buf.copy_to_host(out);
    REQUIRE(status == 0);
    REQUIRE(out == std::vector<double>(host.begin(), host.end()));
}

TEST_CASE("Copy constructor performs deep copy", "[cuda_buffer]") {
    std::array<double,2> host = {{6.6,7.7}};
    cuda_buffer a(host);
    cuda_buffer b(a);
    // modify original on device
    std::vector<double> modified = a.copy_to_host();
    modified[0] = 9.9;
    // b should still have original values
    auto vb = b.copy_to_host();
    REQUIRE(vb[0] == host[0]);
    REQUIRE(vb[1] == host[1]);
}

TEST_CASE("Copy assignment (self and distinct) is safe", "[cuda_buffer]") {
    cuda_buffer a(5);
    a = a;             // self-assign
    REQUIRE(a.size() == 5);
    cuda_buffer b(3);
    b = a;
    REQUIRE(b.size() == 5);
    REQUIRE(a.size() == 5);
}

