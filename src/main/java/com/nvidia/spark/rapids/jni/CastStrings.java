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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;

/** Utility class for converting between column major and row major data */
public class CastStrings {
  static {
    NativeDepsLoader.loadNativeDeps();
  }
    
  public static ColumnVector stringToFloat(ColumnView vec, boolean ansiMode, DType type) {
    return new ColumnVector(stringToFloat(vec.getNativeView(), ansiMode, type.getTypeId().getNativeId()));
  }

  private static native long stringToFloat(long nativeColumnView, boolean ansi_enabled, int dtype);
}
