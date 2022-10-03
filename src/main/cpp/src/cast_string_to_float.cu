/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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


#include <cub/warp/warp_reduce.cuh>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/convert/convert_floats.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

using namespace cudf;

__device__ __inline__ bool is_digit(char c)
{
  return c >= '0' && c <= '9';
}

template<typename T, size_type block_size>
class string_to_float {
public:
  __device__ 
  string_to_float(T* _out, 
                  bitmask_type* _validity, 
                  int32_t *_ansi_except, 
                  size_type* _valid_count, 
                  const char* const _chars, 
                  offset_type const* _offsets, 
                  int _warp_id,
                  uint64_t const*const _ipow) 
    : out(_out),
      validity(_validity),
      ansi_except(_ansi_except),
      valid_count(_valid_count),
      chars(_chars),
      offsets(_offsets),
      tid(threadIdx.x + (blockDim.x * blockIdx.x)),
      warp_id(tid / 32),
      row(warp_id),
      warp_lane(tid % 32),
      row_start(offsets[row]),
      len(offsets[row+1] - row_start),
      ipow(_ipow)
  {    
  }

  __device__ void operator()()
  {
    bstart = 0;                             // start position of the current batch
    blen = min(32, len);                    // length of the batch
    bpos = 0;                               // current position within the current batch of chars for the warp  
    c = warp_lane < blen ? chars[row_start + warp_lane] : 0;

    // printf("(%d): bstart(%d), blen(%d), bpos(%d), tpos(%d), c(%c)\n", tid, bstart, blen, bpos, tpos, c);

    // check for leading nan
    if(check_for_nan()){
      compute_validity(valid, except);
      return;
    }

    // check for + or -
    int sign = check_for_sign();

    // check for inf / infinity
    if(check_for_inf()){
      if(warp_lane == 0){
        out[row] = sign > 0? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      }
      compute_validity(valid, except);
      return;
    }

    // parse the remainder as floating point. 
    auto const [digits, exp_base] = parse_digits();
    if(!valid){
      compute_validity(valid, except);
      return;
    }

    // 0 / -0.
    if(digits == 0){      
      if(warp_lane == 0){
        out[row] = sign * static_cast<double>(0);
      }      
      compute_validity(valid, except);
      return;
    }

    // parse any manual exponent
    auto const manual_exp = parse_manual_exp();
    if(!valid){
      compute_validity(valid, except);
      return;
    }
        
    // construct the final float value
    if(warp_lane == 0){
      // base value
      double digitsf = sign * static_cast<double>(digits);

      // exponent
      int exp_ten = exp_base + manual_exp;

      // final value
      if (exp_ten > std::numeric_limits<double>::max_exponent10){
        out[row] = sign > 0 ? std::numeric_limits<double>::infinity()
                            : -std::numeric_limits<double>::infinity();      
      } else {
        // make sure we don't produce a subnormal number. 
        // - a normal number is one where the leading digit of the floating point rep is not zero. 
        //      eg:   0.0123  represented as  1.23e-2
        //
        // - a denormalized number is one where the leading digit of the floating point rep is zero.
        //      eg:   0.0123 represented as   0.123e-1
        //
        // - a subnormal number is a denormalized number where if you tried to normalize it, the exponent
        //   required would be smaller then the smallest representable exponent. 
        // 
        // https://en.wikipedia.org/wiki/Denormal_number
        //

        double const exponent = exp10(static_cast<double>(std::abs(exp_ten)));
        double const result = exp_ten < 0 ? digitsf / exponent : digitsf * exponent;    

        /*
        if(warp_lane == 0){
          printf("row(%d), %lf, %d, %lf\n", row, digitsf, exp_ten, exponent);
        }
        */
        
        out[row] = result;
      }
    } 
    compute_validity(valid, except);
  }

private:
  // returns true if we encountered 'nan'
  // potentially changes:  valid/except
  __device__ bool check_for_nan()
  {
    auto const nan_mask = __ballot_sync(0xffffffff, (warp_lane == 0 && (c == 'N' || c == 'n')) ||
                                                    (warp_lane == 1 && (c == 'A' || c == 'a')) ||
                                                    (warp_lane == 2 && (c == 'N' || c == 'n')));
    if(nan_mask == 0x7){    
      // if we start with 'nan', then even if we have other garbage character, this is a null row.
      //
      // if we're in ansi mode and this is not -precisely- nan, report that so that we can throw
      // an exception later.
      valid = false;
      except = len != 3;
      return true;
    }
    return false;
  }

  // returns 1 or -1 to indicate sign
  __device__ int check_for_sign()
  {
    auto const sign_mask = __ballot_sync(0xffffffff, warp_lane == 0 && (c == '+' || c == '-'));
    int sign = 1;
    if(sign_mask){
      // NOTE: warp lane 0 is the only thread that ever reads `sign`, so technically it would be 
      // valid to just check if(c == '-'), but that would leave other threads with an incorrect value.
      // if this code ever changes, that could lead to hard-to-find bugs.
      if(__ballot_sync(0xffffffff, warp_lane == 0 && c == '-')){
        sign = -1;
      }
      bpos++;
      c = __shfl_down_sync(0xffffffff, c, 1);
    }
    return sign;
  }

  // returns true if we encountered an inf
  // potentially changes:  valid
  __device__ bool check_for_inf()
  {
    // check for inf or infinity
    auto const inf_mask = __ballot_sync(0xffffffff, (warp_lane == 0 && (c == 'I' || c == 'i')) ||
                                                    (warp_lane == 1 && (c == 'N' || c == 'n')) ||
                                                    (warp_lane == 2 && (c == 'F' || c == 'f')) );
    if(inf_mask == 0x7){
      bpos += 3;
      c = __shfl_down_sync(0xffffffff, c, 3);
      
      // if we're at the end
      if(bpos == len){
        return true;
      }

      // see if we have the whole word
      auto const infinity_mask = __ballot_sync(0xffffffff, (warp_lane == 0 && (c == 'I' || c == 'i')) ||
                                                           (warp_lane == 1 && (c == 'N' || c == 'n')) ||
                                                           (warp_lane == 2 && (c == 'I' || c == 'i')) ||
                                                           (warp_lane == 3 && (c == 'T' || c == 't')) ||
                                                           (warp_lane == 4 && (c == 'Y' || c == 'y')));
      if(infinity_mask == 0x1f){
        // if we're at the end
        if(bpos == len){
          return true;
        }
      }

      // if we reach here for any reason, it means we have "inf" or "infinity" at the start of the string but
      // also have additional characters, making this whole thing bogus/null
      valid = false;
      return true;
    }
    return false;
  }

  // parse the actual digits.  returns 64 bit digit holding value and exponent
  __device__ thrust::pair<uint64_t, int> parse_digits()
  { 
    typedef cub::WarpReduce<uint64_t> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage;
        
    // what we will need to compute the exponent
    uint64_t digits = 0;
    int real_digits = 0;        // total # of digits we've got stored in 'digits'
    int truncated_digits = 0;   // total # of digits we've had to truncate off
    // the # of total digits is (real_digits + truncated_digits)
    bool decimal = false;       // whether or not we have a decimal
    int decimal_pos = 0;        // absolute decimal pos 

    int count = 0;
    
    constexpr int max_safe_digits = 19;
    do {    
      int num_chars = min(max_safe_digits, blen - bpos);    
      /*
      if(warp_lane == 0){
        printf("NC: %d (%d, %d, %d)\n", num_chars, blen, bstart, bpos);
      }
      */

      // if our current sum is 0 and we don't have a decimal, strip leading
      // zeros.  handling cases such as
      // 0000001
      if(!decimal && digits == 0){
        auto const zero_mask = __ballot_sync(0xffffffff, warp_lane < num_chars && c != '0');
        auto const nz_pos = __ffs(zero_mask) - 1;
        if(nz_pos > 0){
          num_chars -= nz_pos;
          bpos += nz_pos;
          c = __shfl_down_sync(0xffffffff, c, nz_pos);
        }
      }

      // handle a decimal point    
      auto const decimal_mask = __ballot_sync(0xffffffff, warp_lane < num_chars && c == '.');    
      if(decimal_mask){
        // if we have more than one decimal, this is an invalid value
        if(decimal || __popc(decimal_mask) > 1){
          valid = false;
          except = true;
          return {0, 0};
        }   
        auto const dpos = __ffs(decimal_mask)-1;    // 0th bit is reported as 1 by __ffs
        decimal_pos = (dpos + real_digits);            
        decimal = true;

        // strip the decimal char out
        if(warp_lane >= dpos){
          c = __shfl_down_sync(~((1 << dpos) - 1), c, 1);
        }
        num_chars--;
      }
      
      // handle any chars that are not actually digits
      //     
      auto const non_digit_mask = __ballot_sync(0xffffffff, warp_lane < num_chars && !is_digit(c));
      auto const first_non_digit = __ffs(non_digit_mask);      
      /*
      if(first_non_digit && warp_lane == 0){
        printf("FND: %d\n", first_non_digit);
      }
      if(first_non_digit){
        printf("(%d)%c\n", tid, c);
      } 
      */     
      num_chars = min(num_chars, first_non_digit > 0 ? first_non_digit - 1 : num_chars);    

      // our local digit
      uint64_t const digit = warp_lane < num_chars ? static_cast<uint64_t>(c - '0') * ipow[(num_chars - warp_lane) - 1] : 0;
            
      // we may have to start truncating because we'd go past the 64 bit limit by adding the new digits
      //
      // max uint64_t is 20 digits, so any 19 digit number is valid.
      // 2^64:  18,446,744,073,709,551,616
      //         9,999,999,999,999,999,999
      //
      // if the 20th digit would push us past that limit, we have to start truncating.
      // max_holding:  1,844,674,407,370,955,160
      // so     1,844,674,407,370,955,160 + 9    -> 18,446,744,073,709,551,609  -> ok
      //        1,844,674,407,370,955,160 + 1X   -> 18,446,744,073,709,551,61X  -> potentially rolls past the limit
      //
      constexpr uint64_t max_holding = (std::numeric_limits<uint64_t>::max() - 9) / 10;
      // if we're already past the max_holding, just truncate.
      // eg:    9,999,999,999,999,999,999
      if(digits > max_holding){
        /*
        if(warp_lane == 0){
          printf("A\n");
        } 
        */
        truncated_digits += num_chars;
      } 
      else {
        // add as many digits to the running sum as we can.
        int const safe_count = min(max_safe_digits - real_digits, num_chars);
        /*
        if(warp_lane == 0){
          printf("SC: %d, %d\n", safe_count, num_chars); 
        }
        */
        if(safe_count > 0){
          // only lane 0 will have the real value so we need to shfl it to the rest of the threads.
          digits = (digits * ipow[safe_count]) + __shfl_sync(0xffffffff, WarpReduce(temp_storage).Sum(digit, safe_count), 0);
          real_digits += safe_count;

          /*
          if(warp_lane == 0){
            printf("B: real_digits(%d)\n", real_digits);
          }
          */
        }

        // if we have more digits
        if(safe_count < num_chars){
          // we're already past max_holding so we have to start truncating
          if(digits > max_holding){
            /*
            if(warp_lane == 0){
              printf("C\n");
            }
            */
            truncated_digits += num_chars - safe_count;
          }
          // we may be able to add one more digit.
          else {
            auto const last_digit = static_cast<uint64_t>(__shfl_sync(0xffffffff, c, safe_count) - '0');
            if((digits * 10) + last_digit <= max_holding){
              // we can add this final digit
              digits = (digits * 10) + last_digit;

              /*
              if(warp_lane == 0){
                printf("D\n");
              }
              */
              truncated_digits += num_chars - (safe_count - 1);
            }
            // everything else gets truncated
            else {
              /*
              if(warp_lane == 0){
                printf("E\n");
              }
              */
              truncated_digits += num_chars - safe_count;            
            }         
          }
        }
      } 
      bpos += num_chars + (decimal_mask > 0);

      /*
      if(warp_lane == 0){
        printf("EXPT: %d (%d, %d, %d, %d)\n", exp_ten, decimal ? 1 : 0, num_chars, decimal_mask, decimal_pos);
      }
      */           

      /*
      if(warp_lane == 0){
        printf("A: bpos(%d), blen(%d), bstart(%d), len(%d)\n", bpos, blen, bstart, len);
      } 
      */     

      // read the next batch of chars.
      if(bpos == blen){
        bstart += blen;
        // nothing left to read?
        if(bstart == len){
          break;
        }
        // read the next batch
        bpos = 0;
        blen = min(32, len - bstart);            
        char c = warp_lane < blen ? chars[row_start + bstart + warp_lane] : 0;
        /*
        if(warp_lane == 0){
          printf("B: bpos(%d), blen(%d), bstart(%d), len(%d)\n", bpos, blen, bstart, len);
        }
        */
      }     
      else {
        //printf("A(%d)%c\n", tid, c);
        /*
        if(warp_lane == 0){
          printf("bpos: %d\n", bpos);
        }
        */
        c = __shfl_down_sync(0xffffffff, c, num_chars);
        //printf("B(%d)%c\n", tid, c);

        // if we encountered a non-digit, we're done
        if(first_non_digit){
          break;
        }
      }
    } while(1);
        
    // 0 / -0.
    if(digits == 0){
      return {0, 0};
    }

    // the total amount of actual digits
    auto const total_digits = real_digits + truncated_digits;

    // exponent
                  // any truncated digits are effectively just trailing zeros    
    int exp_ten = (truncated_digits 
                  // if we've got a decimal, shift left by it's position
                  - (decimal ? (total_digits - decimal_pos) : 0));                  

    //printf("Digits:, %d + %d = %d\n", real_digits, truncated_digits, total_digits);
    return {digits, exp_ten};    
  }

  // parse manually specified exponent.
  // potentially changes: valid
  __device__ int parse_manual_exp()
  {
    typedef cub::WarpReduce<uint64_t> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage;

    // if we still have chars left, the only thing legal now is a manual exponent. 
    // eg:  E-10
    //
    int manual_exp = 0;
    /*
    if(warp_lane == 0){
      printf("eee: %d, %d, %d\n", bstart, bpos, len);
    }
    */
    if(bpos < blen){
      // read some trailing chars.

      auto const exp_mask = __ballot_sync(0xffffffff, (warp_lane == 0 && (c == 'E' || c == 'e')));
      // something invalid
      if(!exp_mask){
        valid = false;
        return 0;
      }
      auto const exp_sign_mask = __ballot_sync(0xffffffff, (warp_lane == 1 && (c == '-' || c == '+')));
      auto const exp_sign = exp_sign_mask ? __ballot_sync(0xffffffff, warp_lane == 1 && c == '-') ? -1 : 1 : 1;      
      c = __shfl_down_sync(0xffffffff, c, exp_sign_mask ? 2 : 1);
            
      // the largest valid exponent for a double is 4 digits (3 for floats). if we have more than that, this is an invalid number.
      auto const num_exp_digits = (len - (bstart + bpos)) - (exp_sign_mask ? 2 : 1);
      /*
      if(warp_lane == 0){    
        printf("ned: %d\n", num_exp_digits);
      } 
      printf("ned(%d): %c\n", warp_lane, c);
      */
      if(num_exp_digits > 4){
        valid = false;
        return 0;
      }
      uint64_t const digit = warp_lane < num_exp_digits ? static_cast<uint64_t>(c - '0') * ipow[(num_exp_digits - warp_lane) - 1] : 0;
      manual_exp = WarpReduce(temp_storage).Sum(digit, num_exp_digits) * exp_sign;
      // printf("Manual EXP: %d\n", manual_exp);
    }
    
    return manual_exp;    
  }

  // sets validity bits, updates outgoing validity count for the block and potentially sets the outgoing ansi_except
  // field
  __device__ void compute_validity(bool const valid, bool const except = false)
  {
    // compute null count for the block. each warp processes one string, so lane 0
    // from each warp contributes 1 bit of validity  
    size_type const block_valid_count = cudf::detail::single_lane_block_sum_reduce<block_size, 0>(valid ? 1 : 0);
    // 0th thread in each block updates the validity count and (optionally) the ansi_except flag
    if (threadIdx.x == 0) {
      atomicAdd(valid_count, block_valid_count); 

      if(ansi_except && except){
        atomicOr(ansi_except, 1);
      }
    }

    // 0th thread in each warp updates the validity
    size_type const row_id = warp_id;
    if(threadIdx.x % 32 == 0){
      // uses atomics
      cudf::set_bit(validity, row_id);    
    }
  }

  T* out;
  bitmask_type* validity;
  int32_t *ansi_except;
  size_type* valid_count;
  const char* const chars;
  offset_type const* offsets;
  size_type const tid;
  size_type const warp_id;
  size_type const row;
  size_type const warp_lane;
  size_type const row_start;
  size_type const len;
  uint64_t const*const ipow;

  // shared/modified by the various parsing functions  
  size_type bstart;   // batch start within the entire string
  size_type bpos;     // position with current batch
  size_type blen;     // batch length;
  char c;             // current character
  bool valid = true;
  bool except = false;
};

template<typename T, size_type block_size>
__global__ void string_to_float_kernel(T* out,
                                       bitmask_type* validity,
                                       int32_t *ansi_except,
                                       size_type* valid_count,
                                       const char* const chars,
                                       offset_type const* offsets,
                                       size_type num_rows)
{
  size_type const tid = threadIdx.x + (blockDim.x * blockIdx.x);
  size_type const warp_id = tid / 32;
  size_type const row = warp_id;
  if(row >= num_rows){
    return;
  }

  __shared__ uint64_t ipow[19];
  if(threadIdx.x == 0){
    ipow[0] = 1;
    ipow[1] = 10;
    ipow[2] = 100;
    ipow[3] = 1000;
    ipow[4] = 10000;
    ipow[5] = 100000;
    ipow[6] = 1000000;
    ipow[7] = 10000000;
    ipow[8] = 100000000;
    ipow[9] = 1000000000;
    ipow[10] = 10000000000;
    ipow[11] = 100000000000;
    ipow[12] = 1000000000000;
    ipow[13] = 10000000000000;
    ipow[14] = 100000000000000;
    ipow[15] = 1000000000000000;
    ipow[16] = 10000000000000000;
    ipow[17] = 100000000000000000;
    ipow[18] = 1000000000000000000;
  }
  __syncthreads();

  // convert
  string_to_float<T, block_size> convert(out, validity, ansi_except, valid_count, chars, offsets, warp_id, ipow);
  convert();
}

void process_one(column_view const& in, 
                 column_view const& expected,                                      
                 rmm::cuda_stream_view stream,
                 rmm::mr::device_memory_resource* mr)
{      
  strings_column_view scv(in);

  auto out = cudf::make_numeric_column(data_type{type_id::FLOAT32}, in.size(), mask_state::UNINITIALIZED, stream, mr);

  using ScalarType = cudf::scalar_type_t<size_type>;
  auto valid_count = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  static_cast<ScalarType*>(valid_count.get())->set_value(0);

  constexpr auto warps_per_block = 8;
  constexpr auto rows_per_block = warps_per_block;
  auto const num_blocks = cudf::util::div_rounding_up_safe(in.size(), rows_per_block);  
  string_to_float_kernel<float, warps_per_block * 32><<<num_blocks, warps_per_block * 32>>>(
    out->mutable_view().begin<float>(), 
    out->mutable_view().null_mask(),
    nullptr,
    static_cast<ScalarType*>(valid_count.get())->data(),    
    scv.chars().begin<char>(),
    scv.offsets().begin<offset_type>(),
    in.size());

  stream.synchronize();

  printf("result:\n");
  cudf::test::print(*out);
  printf("\nexpected:\n");
  cudf::test::print(expected);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*out, expected);
}

void casting_test_simple()
{
  {            
    cudf::test::strings_column_wrapper in{"-1.8946e-10",
                                          "0001", "0000.123", "123", "123.45", "45.123", "-45.123", "0.45123", "-0.45123",
                                          "999999999999999999999",
                                          "99999999999999999999",
                                          "9999999999999999999",
                                          "18446744073709551609",
                                          "18446744073709551610",
                                          "18446744073709551619999999999999",
                                          "-18446744073709551609",
                                          "-18446744073709551610",
                                          "-184467440737095516199999999999997"
                                          };
    
    std::vector<bool> valids{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    auto expected2 = cudf::strings::to_floats(strings_column_view(in), cudf::data_type{type_id::FLOAT32});
    expected2->set_null_mask(cudf::test::detail::make_null_mask(valids.begin(), valids.end()));

    process_one(in,
                *expected2,
                rmm::cuda_stream_default, 
                rmm::mr::get_current_device_resource());
  }
}