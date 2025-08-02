#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Approx;
#include "cuda_buffer.cuh"

TEST_CASE("Allocation of small buffer succeeds", "[cuda_buffer]") {
    cuda_buffer buf(4);
    REQUIRE(buf.size() == 4);
    REQUIRE(buf.data() != nullptr);
}

TEST_CASE("Zero-length buffer behaves correctly", "[cuda_buffer]") {
    cuda_buffer buf(0);
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.data() == nullptr);

    // vector-return
    auto v = buf.copy_to_host();
    REQUIRE(v.empty());

    // vector& overload
    std::vector<double> out;
    REQUIRE(buf.copy_to_host(out) == 0);
    REQUIRE(out.empty());

    // raw-pointer overload with n=0
    REQUIRE(buf.copy_to_host(nullptr, 0) == 0);
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

TEST_CASE("raw-pointer copy_to_host size-mismatch yields error", "[cuda_buffer]") {
    std::array<double,2> host = {{8.8,9.9}};
    cuda_buffer buf(host);
    double small[1];
    REQUIRE(buf.copy_to_host(small, 1) != 0);
}

TEST_CASE("move constructor transfers ownership", "[cuda_buffer]") {
    std::array<double,3> host = {{1.0,2.0,3.0}};
    cuda_buffer a(host);
    cuda_buffer b(std::move(a));

    REQUIRE(b.size() == 3);
    REQUIRE(a.size() == 0);
    REQUIRE(a.data() == nullptr);

    auto out = b.copy_to_host();
    REQUIRE(out == std::vector<double>(host.begin(), host.end()));
}

TEST_CASE("move assignment transfers ownership", "[cuda_buffer]") {
    std::array<double,2> host = {{5.5,6.6}};
    cuda_buffer a(host);
    cuda_buffer b(10);
    b = std::move(a);

    REQUIRE(b.size() == 2);
    REQUIRE(a.size() == 0);
    REQUIRE(a.data() == nullptr);

    auto out = b.copy_to_host();
    REQUIRE(out == std::vector<double>(host.begin(), host.end()));
}

TEST_CASE("repeated allocate/free does not crash", "[cuda_buffer][destructor]") {
    // Create and destroy a buffer 1000 times
    for(int i = 0; i < 1000; ++i) {
        cuda_buffer buf(1024);
        REQUIRE(buf.size() == 1024);
        REQUIRE(buf.data() != nullptr);
        // destructor runs at end of each iteration
    }
}

TEST_CASE("destructor actually frees device memory", "[cuda_buffer][destructor]") {
    size_t free_before, total;
    // query free/total device memory before
    auto st = cudaMemGetInfo(&free_before, &total);
    REQUIRE(st == cudaSuccess);

    {
        // allocate ~8 MB
        cuda_buffer buf(1024 * 1024);
        REQUIRE(buf.data() != nullptr);
        // destructor will run at the end of this scope
    }

    size_t free_after;
    st = cudaMemGetInfo(&free_after, &total);
    REQUIRE(st == cudaSuccess);

    // after destruction, free memory should be at least as large as before
    REQUIRE(free_after >= free_before);
}

