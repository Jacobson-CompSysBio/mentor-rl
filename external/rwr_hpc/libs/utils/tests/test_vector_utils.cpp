#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <utils/vector_utils.hpp>

TEST(TestVectorUtils, HasDuplicatesReturnsFalseOnEmptyInput) {
  std::vector<double> data = {};

  EXPECT_FALSE(utils::has_duplicates(data));
}

TEST(TestVectorUtils, HasDuplicatesReturnsFalseWhenNoDups) {
  std::vector<int> data = {1,2,3,4,5};

  EXPECT_FALSE(utils::has_duplicates(data));
}

TEST(TestVectorUtils, HasDuplicatesHandlesStringCases) {
  std::vector<std::string> data = {"foo","Foo","FOO"};

  EXPECT_FALSE(utils::has_duplicates(data));
}

TEST(TestVectorUtils, HasDuplicatesReturnsTrueWhenDups) {
  std::vector<double> data = {1.0, 2.0, 2.0, -5.023424};

  EXPECT_TRUE(utils::has_duplicates(data));
}

TEST(TestVectorUtils, IdxOfMinElementReturnsZeroOnEmptyInput) {
  std::vector<double> data = {};
  std::size_t idx = utils::idx_of_min_element(data);

  EXPECT_EQ(idx, 0);
}

TEST(TestVectorUtils, IdxOfMinElementReturnsCorrectIndex) {
  std::vector<int> data = {5,8,2,1,9,10,11,12123,234};
  std::size_t idx = utils::idx_of_min_element(data);

  EXPECT_EQ(idx, 3);
}

TEST(TestVectorUtils, IdxOfMinElementReturnsIndexOfFirstMinIfDuplicated) {
  std::vector<int> data = {5,8,1,2,9,10,1,1,234};
  std::size_t idx = utils::idx_of_min_element(data);

  EXPECT_EQ(idx, 2);
}

TEST(TestVectorUtils, IdxOfMaxElementReturnsZeroOnEmptyInput) {
  std::vector<double> data = {};
  std::size_t idx = utils::idx_of_max_element(data);

  EXPECT_EQ(idx, 0);
}

TEST(TestVectorUtils, IdxOfMaxElementReturnsCorrectIndex) {
  std::vector<int> data = {5,8,2,1,9,10,11,12123,234};
  std::size_t idx = utils::idx_of_max_element(data);

  EXPECT_EQ(idx, 7);
}

TEST(TestVectorUtils, IdxOfMaxElementReturnsIndexOfFirstMinIfDuplicated) {
  std::vector<int> data = {234,8,1,2,9,10,1,1,234};
  std::size_t idx = utils::idx_of_max_element(data);

  EXPECT_EQ(idx, 0);
}

TEST(TestVectorUtils, RemoveElementsReturnsEmptyVectorWhenX_andY_AreEmpty) {
  std::vector<int> x = {};
  std::vector<int> y;

  utils::remove_elements(x,y);

  EXPECT_TRUE(x.empty());
}

TEST(TestVectorUtils, RemoveElementsReturnsEmptyVectorWhenX_isEmpty) {
  std::vector<int> x = {};
  std::vector<int> y = {1,2,3};

  utils::remove_elements(x,y);

  EXPECT_TRUE(x.empty());
}

TEST(TestVectorUtils, RemoveElementsReturnsX_WhenY_isEmpty) {
  std::vector<int> x = {4,5,6};
  std::vector<int> expected_result = {4,5,6};
  std::vector<int> y = {};

  utils::remove_elements(x,y);

  EXPECT_EQ(x, expected_result);
}

TEST(TestVectorUtils, RemoveElementsNoMatches) {
  std::vector<std::string> x = {"A","B","C"};
  std::vector<std::string> expected_result = x;
  std::vector<std::string> y = {"1","2","3"};

  utils::remove_elements(x, y);

  EXPECT_EQ(x, expected_result);
}

TEST(TestVectorUtils, RemoveElementsSomeMatches) {
  std::vector<std::string> x = {"A","B","C","D","E","F"};
  std::vector<std::string> expected_result = {"B","C","D","E","F"};
  std::vector<std::string> y = {"1","2","3","A"};

  utils::remove_elements(x, y);

  EXPECT_EQ(x, expected_result);
}

TEST(TestVectorUtils, RemoveElementsAllMatches) {
  std::vector<std::string> x = {"A","B","C"};
  std::vector<std::string> y = {"A","B","C"};

  utils::remove_elements(x, y);

  EXPECT_TRUE(x.empty());
}

TEST(TestVectorUtils, SafeCastVectorThrowsOnCastingNanToIntegral) {
  std::vector<double> input = {1.0, 2.0, std::nan("")};

  ASSERT_THAT(
    [&](){auto result = utils::safe_cast_vector<int>(input); },
    testing::ThrowsMessage<std::runtime_error>("Element 2: Cannot convert NaN to integral type")
  );
}

TEST(TestVectorUtils, SafeCastVectorThrowsOnCastingInfToIntegral) {
  std::vector<double> input = {1.0, 2.0, std::numeric_limits<double>::infinity()};

  ASSERT_THAT(
    [&](){auto result = utils::safe_cast_vector<int>(input); },
    testing::ThrowsMessage<std::runtime_error>("Element 2: Cannot convert infinity to integral type")
  );
}

TEST(TestVectorUtils, SafeCastVectorThrowsOnCastingNegativeToUnsigned) {
  std::vector<int> input = {1,-2,3};

  ASSERT_THAT(
    [&](){auto result = utils::safe_cast_vector<unsigned long>(input); },
    testing::ThrowsMessage<std::runtime_error>("Element 1: Negative value -2 cannot be cast to unsigned type")
  );
}

TEST(TestVectorUtils, SafeCastVectorThrowsOnOverflow) {
  std::vector<unsigned long> input = {1,std::numeric_limits<unsigned long>::max()};

  ASSERT_THAT(
    [&](){auto result = utils::safe_cast_vector<int>(input); },
    testing::ThrowsMessage<std::runtime_error>("Element 1: Value 18446744073709551615 out of range for target type [-2.14748e+09, 2.14748e+09]")
  );
}


TEST(TestVectorUtils, SafeCastVectorThrowsOnUnderflow) {
  std::vector<long> input = {1,std::numeric_limits<long>::min()};

  ASSERT_THAT(
    [&](){auto result = utils::safe_cast_vector<int>(input); },
    testing::ThrowsMessage<std::runtime_error>("Element 1: Value -9223372036854775808 out of range for target type [-2.14748e+09, 2.14748e+09]")
  );
}

TEST(TestVectorUtils, SafeCastVectorCorrectlyCasts32To64) {
  std::vector<int> input = {1,2,3};
  std::vector<unsigned long> expected_output = {1UL, 2UL, 3UL};

  std::vector<unsigned long> actual_output = utils::safe_cast_vector<unsigned long>(input);

  EXPECT_EQ(actual_output, expected_output);
}

TEST(TestVectorUtils, SafeCastVectorCorrectlyCasts64To32) {
  std::vector<unsigned long> input = {1UL, 2UL, 3UL};
  std::vector<int> expected_output = {1, 2, 3};

  auto actual_output = utils::safe_cast_vector<int>(input);

  EXPECT_EQ(actual_output, expected_output);
}

TEST(TestVectorUtils, SafeCastVectorCorrectlyCastsDoubleToUnsignedLon) {
  std::vector<double> input = {1.1,2.5,3.3};
  std::vector<unsigned long> expected_output = {1UL, 2UL, 3UL};

  std::vector<unsigned long> actual_output = utils::safe_cast_vector<unsigned long>(input);

  EXPECT_EQ(actual_output, expected_output);
}

TEST(TestVectorUtils, ConcateReturnsEmptyStringOnEmptyInput) {
  std::vector<std::string> data;
  std::string expected_result = "";
  
  std::string actual_result = utils::concate(data);

  EXPECT_EQ(expected_result, actual_result);
}

TEST(TestVectorUtils, ConcateReturnsCorrectStringDefaultSep) {
  std::vector<std::string> data = {"ABC","123"};
  std::string expected_result = "ABC_123";

  std::string actual_result = utils::concate(data);
  EXPECT_EQ(expected_result, actual_result);
}

TEST(TestVectorUtils, ConcateReturnsCorrectStringNondefaultSep) {
  std::vector<std::string> data = {"ABC","123","ABC","456"};
  std::string expected_result = "ABC-123-ABC-456";

  std::string actual_result = utils::concate(data, "-");
  EXPECT_EQ(expected_result, actual_result);
}
