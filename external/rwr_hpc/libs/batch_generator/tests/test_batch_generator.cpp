#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <batch_generator/BatchGen.hpp>

TEST(TestBatchGen, ConstructorThrowsOnMinGreaterThanMax) {
  ASSERT_THAT(
    [&](){BatchGen<int>(5, 3, 2); },
    testing::ThrowsMessage<std::invalid_argument>("BatchGen - Invalid range or batch size.")
  );
}

TEST(TestBatchGen, ConstructorThrowsOnBatchSizeNegative) {
    ASSERT_THAT(
    [&](){BatchGen<int>(4, 30, -1); },
    testing::ThrowsMessage<std::invalid_argument>("BatchGen - Invalid range or batch size.")
  );
}

TEST(TestBatchGen, ValidResultsForTypeSize_t) {
  BatchGen<std::size_t> gen(0, 13, 5);

  EXPECT_FALSE(gen.done());
  auto pair = gen.next();

  EXPECT_EQ(pair.first, 0);
  EXPECT_EQ(pair.second, 4);

  EXPECT_FALSE(gen.done());
  pair = gen.next();

  EXPECT_EQ(pair.first, 5);
  EXPECT_EQ(pair.second, 9);

  EXPECT_FALSE(gen.done());
  pair = gen.next();

  EXPECT_EQ(pair.first, 10);
  EXPECT_EQ(pair.second, 13);

  EXPECT_TRUE(gen.done());
  pair = gen.next();

  EXPECT_EQ(pair.first, 14);
  EXPECT_EQ(pair.second, 14);

  gen.reset();

  EXPECT_FALSE(gen.done());
  pair = gen.next();

  EXPECT_EQ(pair.first, 0);
  EXPECT_EQ(pair.second, 4);
}

TEST(TestBatchGen, ValidResultsForTypeInt) {
  BatchGen<int> gen(-3, 9, 4);

  EXPECT_FALSE(gen.done());
  auto pair = gen.next();

  EXPECT_EQ(pair.first, -3);
  EXPECT_EQ(pair.second, 0);

  EXPECT_FALSE(gen.done());
  pair = gen.next();

  EXPECT_EQ(pair.first, 1);
  EXPECT_EQ(pair.second, 4);

  EXPECT_FALSE(gen.done());
  pair = gen.next();

  EXPECT_EQ(pair.first, 5);
  EXPECT_EQ(pair.second, 8);

  EXPECT_FALSE(gen.done());
  pair = gen.next();

  EXPECT_EQ(pair.first, 9);
  EXPECT_EQ(pair.second, 9);

  EXPECT_TRUE(gen.done());
  pair = gen.next();

  EXPECT_EQ(pair.first, 10);
  EXPECT_EQ(pair.second, 10);

  gen.reset();

  EXPECT_FALSE(gen.done());
  pair = gen.next();

  EXPECT_EQ(pair.first, -3);
  EXPECT_EQ(pair.second, 0);
}
