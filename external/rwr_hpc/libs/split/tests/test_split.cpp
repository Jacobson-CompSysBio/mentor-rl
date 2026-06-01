#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <split/split.hpp>

TEST(TestSplit, ThrowsWhenCurWorkerEqualToNumWorkers) {
  const unsigned long cur_worker = 5;
  const unsigned long num_workers = cur_worker;
  const unsigned long num_tasks = 30;
  unsigned long start, stop;
  ASSERT_THAT(
    [&](){split::split_tasks_among_workers(cur_worker, num_workers, num_tasks, start, stop); },
    testing::ThrowsMessage<std::invalid_argument>("split_tasks_among_workers - cur_worker must be less than num_workers")
  );
}

TEST(TestSplit, ThrowsWhenCurWorkerGreaterThanNumWorkers) {
  const unsigned long cur_worker = 9;
  const unsigned long num_workers = 8;
  const unsigned long num_tasks = 30;
  unsigned long start, stop;
  ASSERT_THAT(
    [&](){split::split_tasks_among_workers(cur_worker, num_workers, num_tasks, start, stop); },
    testing::ThrowsMessage<std::invalid_argument>("split_tasks_among_workers - cur_worker must be less than num_workers")
  );
}

TEST(TestSplit, ReturnsFalseForZeroTasks) {
  const unsigned long num_workers = 8;
  const unsigned long num_tasks = 0;
  unsigned long start, stop;
  for (unsigned long cur_worker = 0; cur_worker < num_workers; ++cur_worker) {
    bool valid_split = split::split_tasks_among_workers(
                        cur_worker,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
    
    EXPECT_FALSE(valid_split);
  }
}

TEST(TestSplit, CalculatesValuesWhenFewTasksThanWorkers) {
  const unsigned long num_workers = 8;
  const unsigned long num_tasks = 5;
  unsigned long start, stop;
  
  bool valid_split = split::split_tasks_among_workers(
                        0UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
 
  EXPECT_TRUE(valid_split);
  EXPECT_EQ(start, 0);
  EXPECT_EQ(stop, 0);

  valid_split = split::split_tasks_among_workers(
                        1UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
 
  EXPECT_TRUE(valid_split);
  EXPECT_EQ(start, 1);
  EXPECT_EQ(stop, 1);

  valid_split = split::split_tasks_among_workers(
                        2UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
 
  EXPECT_TRUE(valid_split);
  EXPECT_EQ(start, 2);
  EXPECT_EQ(stop, 2);

  valid_split = split::split_tasks_among_workers(
                        3UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
 
  EXPECT_TRUE(valid_split);
  EXPECT_EQ(start, 3);
  EXPECT_EQ(stop, 3);

  valid_split = split::split_tasks_among_workers(
                        4UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
 
  EXPECT_TRUE(valid_split);
  EXPECT_EQ(start, 4);
  EXPECT_EQ(stop, 4);

  valid_split = split::split_tasks_among_workers(
                        5UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
 
  EXPECT_FALSE(valid_split);


  valid_split = split::split_tasks_among_workers(
                        6UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
 
  EXPECT_FALSE(valid_split);

  valid_split = split::split_tasks_among_workers(
                        7UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
 
  EXPECT_FALSE(valid_split);

}

TEST(TestSplit, CalculatesValuesWhenTaskSplitEvenly) {
  const unsigned long num_workers = 8;
  const unsigned long num_tasks = 32;
  unsigned long start, stop;
  for (unsigned long cur_worker = 0; cur_worker < num_workers; ++cur_worker) {
    bool valid_split = split::split_tasks_among_workers(
                        cur_worker,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
    
    EXPECT_TRUE(valid_split);
    EXPECT_EQ(start, cur_worker * 4);
    EXPECT_EQ(stop, cur_worker * 4 + 3);
  }
}

TEST(TestSplit, CalculatesValuesWhenTaskSplitUnevenly) {
  const unsigned long num_workers = 8;
  const unsigned long num_tasks = 35;
  unsigned long start, stop;
  
  bool valid_split = split::split_tasks_among_workers(
                        0UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
    
  EXPECT_TRUE(valid_split);
  EXPECT_EQ(start, 0);
  EXPECT_EQ(stop, 4);

  valid_split = split::split_tasks_among_workers(
                        1UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
    
  EXPECT_TRUE(valid_split);
  EXPECT_EQ(start, 5);
  EXPECT_EQ(stop, 9);

  valid_split = split::split_tasks_among_workers(
                        2UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
    
  EXPECT_TRUE(valid_split);
  EXPECT_EQ(start, 10);
  EXPECT_EQ(stop, 14);

  valid_split = split::split_tasks_among_workers(
                        3UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
    
  EXPECT_TRUE(valid_split);
  EXPECT_EQ(start, 15);
  EXPECT_EQ(stop, 18);

  valid_split = split::split_tasks_among_workers(
                        4UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
    
  EXPECT_TRUE(valid_split);
  EXPECT_EQ(start, 19);
  EXPECT_EQ(stop, 22);

  valid_split = split::split_tasks_among_workers(
                        5UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
    
  EXPECT_TRUE(valid_split);
  EXPECT_EQ(start, 23);
  EXPECT_EQ(stop, 26);

  valid_split = split::split_tasks_among_workers(
                        6UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
    
  EXPECT_TRUE(valid_split);
  EXPECT_EQ(start, 27);
  EXPECT_EQ(stop, 30);

  valid_split = split::split_tasks_among_workers(
                        7UL,
                        num_workers,
                        num_tasks,
                        start,
                        stop);
    
  EXPECT_TRUE(valid_split);
  EXPECT_EQ(start, 31);
  EXPECT_EQ(stop,  34);
}
