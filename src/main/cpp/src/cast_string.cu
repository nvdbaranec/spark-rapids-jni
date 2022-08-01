#include <cub/warp/warp_reduce.cuh>
#include <cudf/column/column.hpp>

using namespace cudf;

/*
template<size_type block_size>
void compute_validity(size_type* valid_count, size_type const tid, bool const valid)
{
  // compute null count for the block. each warp processes one string, so lane 0
  // from each warp contributes 1 bit of validity
  size_type const block_valid_count = cudf::detail::single_lane_block_sum_reduce<block_size, 0>(valid ? 1 : 0);
  if (tid == 0) { atomicAdd(valid_count, block_valid_count); }
}
*/

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
  size_type const warp_lane = tid % 32; 
  size_type const row_start = offsets[row];
  size_type const len = offsets[row+1] - row_start;   
  
  size_type bstart = 0;                             // start position of the current batch
  size_type blen = min(32, len);                    // length of the batch
  size_type bpos = 0;                               // current position within the current batch of chars for the warp  
  char c = warp_lane < blen ? chars[row_start + warp_lane] : 0;

  size_type tpos = warp_lane;                       // current thread position relative to bpos  

  // printf("(%d): bstart(%d), blen(%d), bpos(%d), tpos(%d), c(%c)\n", tid, bstart, blen, bpos, tpos, c);

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

  // a valid string can -only- start with:
  // nan
  // OR
  // +/-
  // inf
  // infinity
  // digits

  // check for leading nan
  auto const nan_mask = __ballot_sync(0xffffffff, (tpos == 0 && (c == 'N' || c == 'n')) ||
                                                  (tpos == 1 && (c == 'A' || c == 'a')) ||
                                                  (tpos == 2 && (c == 'N' || c == 'n')));
  if(nan_mask == 0x7){    
    // if we start with 'nan', then even if we have other garbage character, this is a null row.
    //
    // if we're in ansi mode and this is not -precisely- nan, report that so that we can throw
    // an exception later.
    if(warp_lane == 0 && ansi_except && len != 3){
      atomicOr(ansi_except, 1);
    }
    //compute_validity(tid, false);    
    return;
  }
    
  // check for + or -
  auto const sign_mask = __ballot_sync(0xffffffff, tpos == 0 && (c == '+' || c == '-'));
  int sign = 1;
  if(sign_mask){
    bpos++;
    tpos--;
    sign = c == '+' ? 1 : -1;
  }

  // check for inf or infinity
  auto const inf_mask = __ballot_sync(0xffffffff, (tpos == 0 && (c == 'I' || c == 'i')) ||
                                                  (tpos == 1 && (c == 'N' || c == 'n')) ||
                                                  (tpos == 2 && (c == 'F' || c == 'f')) );
  if(inf_mask == 0x7){
    bpos += 3;
    tpos -= 3;
    
    // if we're at the end
    if(bpos == len){
      if(warp_lane == 0){
        out[row] = sign > 0? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      }
      //compute_validity(tid, true);    
      return;
    }

    // see if we have the whole word
    auto const infinity_mask = __ballot_sync(0xffffffff, (tpos == 0 && (c == 'I' || c == 'i')) ||
                                                         (tpos == 1 && (c == 'N' || c == 'n')) ||
                                                         (tpos == 2 && (c == 'I' || c == 'i')) ||
                                                         (tpos == 3 && (c == 'T' || c == 't')) ||
                                                         (tpos == 4 && (c == 'Y' || c == 'y')));
    if(infinity_mask == 0x1f){
      // if we're at the end
      if(bpos == len){
        if(warp_lane == 0){
          out[row] = sign > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
        }
        //compute_validity(tid, true);
        return;
      }
    }

    // if we reach here for any reason, it means we have "inf" or "infinity" at the start of the string but
    // also have additional characters, making this whole thing bogus/null
    if(warp_lane == 0 && ansi_except){
      atomicOr(ansi_except, 1);
    }
    //compute_validity(tid, false);
    return;
  }

  // parse the remainder as (potentially) valid floating point. 

  // shuffle remaining chars down so lane 0 has the first unprocessed digit of the batch
  c = __shfl_down_sync(0xffffffff, c, bpos);
  
  typedef cub::WarpReduce<uint64_t> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage;
      
  int total_digits = 0;
  uint64_t digits = 0;
  int exp_ten = 0;
  int decimal_pos = 0;
  bool decimal = false;
  int count = 0;
  constexpr int max_safe_digits = 19;
  bool truncating = false;  
  do {    
    int num_chars = min(max_safe_digits, blen - (bstart + bpos));
    if(warp_lane == 0){
      printf("NC: %d (%d, %d, %d)\n", num_chars, blen, bstart, bpos);
    }

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

    // # of digits to the left and right of the decimal for this iteration
    int ldigits = decimal ? 0 : num_chars;
    int rdigits = num_chars - ldigits;

    // handle a decimal point    
    auto const decimal_mask = __ballot_sync(0xffffffff, warp_lane < num_chars && c == '.');    
    if(decimal_mask){
      // if we have more than one decimal, this is an invalid value
      if(decimal || __popc(decimal_mask) > 1){
        //
      }   
      auto const dpos = __ffs(decimal_mask)-1;    // 0th bit is reported as 1 by __ffs
      decimal_pos = (dpos + total_digits);            
      decimal = true;      

      // strip the decimal char out
      if(warp_lane >= dpos){
        c = __shfl_down_sync(~((1 << dpos) - 1), c, 1);
      }
      num_chars--;
      ldigits = dpos;
      rdigits = num_chars - ldigits;
    }    

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
    // so     1,844,674,407,370,955,160 ++ 9    -> 18,446,744,073,709,551,609  -> ok
    //        1,844,674,407,370,955,160 ++ 1X   -> 18,446,744,073,709,551,61X  -> potentially rolls past the limit
    //
    constexpr uint64_t max_holding = (std::numeric_limits<uint64_t>::max() - 9) / 10;
    // if we're already past the max_holding, just truncate.
    // eg:    9,999,999,999,999,999,999
    if(digits > max_holding){
      if(warp_lane == 0){
        printf("A\n");
      }      
      exp_ten += ldigits;
    } 
    else {
      // add as many digits to the running sum as we can.
      int const safe_count = min(max_safe_digits - total_digits, num_chars);
      if(safe_count > 0){
        // only lane 0 will have the real value so we need to shfl it to the rest of the threads.
        digits = (digits * ipow[safe_count]) + __shfl_sync(0xffffffff, WarpReduce(temp_storage).Sum(digit, safe_count), 0);
        total_digits += safe_count;

        if(warp_lane == 0){
          printf("B: total_digits(%d)\n", total_digits);
        }
      }

      // if we have more digits
      if(safe_count < num_chars){
        // we're already past max_holding so we have to start truncating
        if(digits > max_holding){
          if(warp_lane == 0){
            printf("C\n");
          }
        } 
        // we may be able to add one more digit.
        else {
          auto const last_digit = static_cast<uint64_t>(__shfl_sync(0xffffffff, c, safe_count) - '0');
          if((digits * 10) + last_digit <= max_holding){
            // we can add this final digit
            digits = (digits * 10) + last_digit;

            if(warp_lane == 0){
              printf("D\n");
            }
          }
          // everything else gets truncated

          if(warp_lane == 0){
            printf("E\n");
          }
        }
      }
    } 
    bpos += num_chars + (decimal_mask > 0);        
    // adjust the exponent        
    if(decimal){
      printf("EA\n");
      // move left for every digit to the right of the decimal
      exp_ten += ((decimal ? -1 : 0) * num_chars) + (decimal_mask > 0 ? decimal_pos : 0);
    } 
    // if we are to the left of the decimal, we're just truncating extra digits and increasing
    // the exponent instead.
    else if(digits > max_holding){    
      printf("EB\n");
      exp_ten += num_chars;
    }    
    if(warp_lane == 0){
      printf("EXPT: %d (%d, %d, %d, %d)\n", exp_ten, decimal ? 1 : 0, num_chars, decimal_mask, decimal_pos);
    }
    
    /*
    if(warp_lane == 0){   
      printf("EO: %d -> %d (%d, %d, %d)\n", exp_ten - (decimal * num_chars), exp_ten, decimal, num_chars);
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
    } else {
      c = __shfl_down_sync(0xffffffff, c, bpos);
    }
  } while(1);
  
  if(warp_lane == 0){
    // 0 / -0
    if(digits == 0){
      out[row] = sign * static_cast<double>(0);
      return;
    }

    // base value
    double digitsf = sign * static_cast<double>(digits);
    
    // exponent    
    printf("ET: %d, %d\n", exp_ten, decimal_pos);
    if (exp_ten > std::numeric_limits<double>::max_exponent10){
      out[row] = sign > 0 ? std::numeric_limits<double>::infinity()
                          : -std::numeric_limits<double>::infinity();
      return;
    }

    // make sure we don't produce a subnormal number. 
    // - a normal number is one where the leading digit of the floating point rep not zero. 
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

    if(warp_lane == 0){
      printf("row(%d), %lf, %d\n", row, digitsf, exp_ten);
    }

    double const exponent = exp10(static_cast<double>(std::abs(exp_ten)));
    double const result = exp_ten < 0 ? digitsf / exponent : digitsf * exponent;  
    
    out[row] = result;
  }    

  // compute_validity(tid, true);
}