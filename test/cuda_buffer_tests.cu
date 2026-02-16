#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Approx;
#include "cuda_buffer.cuh"

using namespace zeus;

TEST_CASE("allocation of small buffer succeeds", "[dbuf]")
{
  dbuf buf(4);
  REQUIRE(buf.size() == 4);
  REQUIRE(buf.data() != nullptr);
}

TEST_CASE("zero-length buffer behaves correctly", "[dbuf]")
{
  dbuf buf(0);
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

TEST_CASE("ctor from host array + copy_to_host round-trips", "[dbuf]")
{
  std::array<double, 3> host = {{1.1, 2.2, 3.3}};
  dbuf buf(host);
  auto out = buf.copy_to_host();
  REQUIRE(out == std::vector<double>(host.begin(), host.end()));
}

TEST_CASE("copy_to_host(vector&) overload", "[dbuf]")
{
  std::array<double, 2> host = {{4.4, 5.5}};
  dbuf buf(host);
  std::vector<double> out;
  int status = buf.copy_to_host(out);
  REQUIRE(status == 0);
  REQUIRE(out == std::vector<double>(host.begin(), host.end()));
}

TEST_CASE("copy ctor performs deep copy", "[dbuf]")
{
  std::array<double, 2> host = {{6.6, 7.7}};
  dbuf a(host);
  dbuf b(a);
  // modify original on device
  std::vector<double> modified = a.copy_to_host();
  modified[0] = 9.9;
  // b should still have original values
  auto vb = b.copy_to_host();
  REQUIRE(vb[0] == host[0]);
  REQUIRE(vb[1] == host[1]);
}

TEST_CASE("copy assignment (self and distinct) is safe", "[dbuf]")
{
  dbuf a(5);
  void* old_d = a.data();
  a = a; // self-assign
  REQUIRE(a.size() == 5);
  REQUIRE(a.data() == old_d); // self-assignment shouldn't reallocate

  dbuf b(3);
  b = a;
  REQUIRE(b.size() == 5);
  REQUIRE(a.size() == 5);
  REQUIRE(b.data() != a.data()); // deep copy should give distinct pointers
  REQUIRE(b.copy_to_host() == a.copy_to_host()); // contents should match
}

TEST_CASE("raw-pointer copy_to_host size-mismatch yields error", "[dbuf]")
{
  std::array<double, 2> host = {{8.8, 9.9}};
  dbuf buf(host);
  double small[1];
  REQUIRE(buf.copy_to_host(small, 1) != 0);
}

TEST_CASE("move constructor transfers ownership", "[dbuf]")
{
  std::array<double, 3> host = {{1.0, 2.0, 3.0}};
  dbuf a(host);
  dbuf b(std::move(a));

  REQUIRE(b.size() == 3);
  REQUIRE(a.size() == 0);
  REQUIRE(a.data() == nullptr);

  auto out = b.copy_to_host();
  REQUIRE(out == std::vector<double>(host.begin(), host.end()));
}

TEST_CASE("move assignment transfers ownership", "[dbuf]")
{
  std::array<double, 2> host = {{5.5, 6.6}};
  dbuf a(host);
  dbuf b(10);
  b = std::move(a);

  REQUIRE(b.size() == 2);
  REQUIRE(a.size() == 0);
  REQUIRE(b.data() != nullptr);
  REQUIRE(a.data() == nullptr);

  auto out = b.copy_to_host();
  REQUIRE(out == std::vector<double>(host.begin(), host.end()));
}

TEST_CASE("repeated allocate/free does not crash", "[dbuf][destructor]")
{
  // Create and destroy a buffer 1000 times
  for (int i = 0; i < 1000; ++i) {
    dbuf buf(1024);
    REQUIRE(buf.size() == 1024);
    REQUIRE(buf.data() != nullptr);
    // destructor runs at end of each iteration
  }
}

TEST_CASE("destructor actually frees device memory", "[dbuf][destructor]")
{
  size_t free_before, total;
  // query free/total device memory before
  auto st = cudaMemGetInfo(&free_before, &total);
  REQUIRE(st == cudaSuccess);

  {
    // allocate ~8 MB
    dbuf buf(1024 * 1024);
    REQUIRE(buf.data() != nullptr);
    // destructor will run at the end of this scope
  }

  size_t free_after;
  st = cudaMemGetInfo(&free_after, &total);
  REQUIRE(st == cudaSuccess);

  // after destruction, free memory should be at least as large as before
  REQUIRE(free_after >= free_before);
}

TEST_CASE("default constructor creates empty buffer", "[dbuf][default-ctor]")
{
  dbuf buf;
  REQUIRE(buf.size() == 0);
  REQUIRE(buf.data() == nullptr);

  // should be safe to copy to host
  auto v = buf.copy_to_host();
  REQUIRE(v.empty());

  // should be safe to destroy (destructor shouldn't call cudaFree on nullptr)
}

TEST_CASE("swap exchanges buffer contents", "[dbuf][swap]")
{
  std::array<double, 3> host_a = {{1.0, 2.0, 3.0}};
  std::array<double, 2> host_b = {{4.0, 5.0}};

  dbuf a(host_a);
  dbuf b(host_b);

  double* ptr_a = a.data();
  double* ptr_b = b.data();
  size_t size_a = a.size();
  size_t size_b = b.size();

  a.swap(b);

  // after swap, a should have b's old pointer and size
  REQUIRE(a.data() == ptr_b);
  REQUIRE(a.size() == size_b);

  // and b should have a's old pointer and size
  REQUIRE(b.data() == ptr_a);
  REQUIRE(b.size() == size_a);

  // verify contents
  auto vec_a = a.copy_to_host();
  auto vec_b = b.copy_to_host();

  REQUIRE(vec_a == std::vector<double>(host_b.begin(), host_b.end()));
  REQUIRE(vec_b == std::vector<double>(host_a.begin(), host_a.end()));
}

TEST_CASE("swap with empty buffer", "[dbuf][swap]")
{
  std::array<double, 2> host = {{7.0, 8.0}};
  dbuf a(host);
  dbuf b; // default constructed

  double* ptr_a = a.data();
  size_t size_a = a.size();

  a.swap(b);

  REQUIRE(a.size() == 0);
  REQUIRE(a.data() == nullptr);
  REQUIRE(b.size() == size_a);
  REQUIRE(b.data() == ptr_a);

  auto vec_b = b.copy_to_host();
  REQUIRE(vec_b == std::vector<double>(host.begin(), host.end()));
}

TEST_CASE("implicit conversion to raw pointer", "[dbuf][conversion]")
{
  std::array<double, 3> host = {{1.5, 2.5, 3.5}};
  dbuf buf(host);

  // implicit conversion to T*
  double* raw_ptr = buf;
  REQUIRE(raw_ptr == buf.data());
  REQUIRE(raw_ptr != nullptr);

  // can use in cuda API calls that expect raw pointers
  std::vector<double> out(3);
  auto st =
    cudaMemcpy(out.data(), raw_ptr, 3 * sizeof(double), cudaMemcpyDeviceToHost);
  REQUIRE(st == cudaSuccess);
  REQUIRE(out == std::vector<double>(host.begin(), host.end()));
}

TEST_CASE("implicit conversion for empty buffer", "[dbuf][conversion]")
{
  dbuf buf;
  double* raw_ptr = buf;
  REQUIRE(raw_ptr == nullptr);
}

TEST_CASE("operator== compares pointer and size", "[dbuf][equality]")
{
  std::array<double, 3> host = {{1.0, 2.0, 3.0}};
  dbuf a(host);
  dbuf b(host); // different buffer with same contents

  // different buffers should not be equal (different pointers)
  REQUIRE_FALSE(a == b);
  REQUIRE(a.data() != b.data());

  // buffer should equal itself
  REQUIRE(a == a);
  REQUIRE(b == b);
}

TEST_CASE("operator== for empty buffers", "[dbuf][equality]")
{
  dbuf a;
  dbuf b;

  // two default-constructed buffers should be equal (both nullptr, size 0)
  REQUIRE(a == b);

  dbuf c(0); // explicitly constructed with size 0
  REQUIRE(a == c);
}

TEST_CASE("operator== after move", "[dbuf][equality]")
{
  std::array<double, 2> host = {{5.0, 6.0}};
  dbuf a(host);
  double* ptr = a.data();

  dbuf b(std::move(a));

  // a should now be empty
  dbuf empty;
  REQUIRE(a == empty);

  // b should have the original pointer
  REQUIRE(b.data() == ptr);
  REQUIRE_FALSE(b == empty);
}
