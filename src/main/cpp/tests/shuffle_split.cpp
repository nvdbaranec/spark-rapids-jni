/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include "shuffle_split.hpp"

struct ShuffleSplitTests : public cudf::test::BaseFixture {};

void run_split(cudf::table_view const& tbl, std::vector<cudf::size_type> const& splits)
{    
    auto [split_data, split_metadata] = cudf::spark_rapids_jni::shuffle_split(tbl,
                                                                              splits,
                                                                              cudf::get_default_stream(),
                                                                              rmm::mr::get_current_device_resource());
    auto result = shuffle_assemble(split_metadata,
                                  {static_cast<int8_t*>(split_data.partitions->data()), split_data.partitions->size()},
                                  split_data.offsets,
                                  cudf::get_default_stream(),
                                  rmm::mr::get_current_device_resource());
    CUDF_TEST_EXPECT_TABLES_EQUAL(tbl, *result);
}

TEST_F(ShuffleSplitTests, Simple)
{  
  cudf::size_type const num_rows = 10000;
  auto iter = thrust::make_counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<int> col0(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<float> col1(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<short> col2(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int8_t> col3(iter, iter + num_rows);


  // 4 columns split once
  {
    cudf::table_view tbl{{static_cast<cudf::column_view>(col0),
                          static_cast<cudf::column_view>(col1),
                          static_cast<cudf::column_view>(col2),
                          static_cast<cudf::column_view>(col3)}};
    run_split(tbl, {10});
  }
  /*
  {
    size_type const num_rows = 10000;    
    auto iter = thrust::make_counting_iterator(0);
    cudf::test::fixed_width_column_wrapper<int> col(iter, iter + num_rows);
    cudf::test::fixed_width_column_wrapper<float> col1(iter, iter + num_rows);
    cudf::test::fixed_width_column_wrapper<short> col2(iter, iter + num_rows);
    cudf::test::fixed_width_column_wrapper<int8_t> col3(iter, iter + num_rows);
    
    std::vector<size_type> splits;
    splits.reserve(num_rows - 1);
    for(size_t idx = 0; idx<num_rows-1; idx++){
      splits.push_back(static_cast<size_type>(idx+1));
    }
    */
   /*
  std::vector<size_type> splits{10};

    cudf::table_view tv{{static_cast<cudf::column_view>(col),
                        static_cast<cudf::column_view>(col1)}};
                        //static_cast<cudf::column_view>(col2),
                        //static_cast<cudf::column_view>(col3)}};
    auto [split_data, split_metadata] = cudf::spark_rapids_jni::shuffle_split(tv,
                                                                              splits,
                                                                              cudf::get_default_stream(),
                                                                              rmm::mr::get_current_device_resource());

    print_span(cudf::device_span<size_t const>{split_data.offsets});

    auto result = shuffle_assemble(split_metadata,
                                  {static_cast<int8_t*>(split_data.partitions->data()), split_data.partitions->size()},
                                  split_data.offsets,
                                  cudf::get_default_stream(),
                                  rmm::mr::get_current_device_resource());

    CUDF_TEST_EXPECT_TABLES_EQUAL(tv, *result);
  }
  */
}

