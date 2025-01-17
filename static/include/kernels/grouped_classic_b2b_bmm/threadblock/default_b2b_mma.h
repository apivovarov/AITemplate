// @lint-ignore-every LICENSELINT
// clang-format off

/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.

    This file is copied from https://github.com/NVIDIA/cutlass/tree/master/examples/13_two_tensor_op_fusion.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"

#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"
#include "cutlass/transform/threadblock/predicated_vector_access_iterator.h"
#include "cutlass/transform/threadblock/vector_iterator.h"
#include "cutlass/transform/warp/vector_fragment_iterator.h"

#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h"

#include "grouped_classic_b2b_bmm/threadblock/b2b_mma_pipelined.h"
#include "grouped_classic_b2b_bmm/threadblock/b2b_mma_multistage.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Layout type for B1 matrix operand
    typename LayoutB1_,
    /// Element type for C matrix
    typename ElementC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape0_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape1_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape0_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape1_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Apply upper triangular causal mask after first gemm
    bool CausalMaskAfterGemm0,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Staging the accumulators in shared memory.
    bool SmemAccumulator = false>
struct DefaultB2bMma;

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major output with 2-stage pipeline
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Layout type for B1 matrix operand
    typename LayoutB1,
    /// Element type forAC matrix operand
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape0,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape1,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape0,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape1,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Apply upper triangular causal mask after first gemm
    bool CausalMaskAfterGemm0,
    /// Epilogue output operator
    typename EpilogueOutputOp>
struct DefaultB2bMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, LayoutB1, ElementC, ElementAccumulator, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag,
                  ThreadblockShape0, ThreadblockShape1,
                  WarpShape0, WarpShape1,
                  InstructionShape, 2, Operator, CausalMaskAfterGemm0, EpilogueOutputOp, false> {
  // Define the MmaCore components
  using MmaCore0 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape0, WarpShape0, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
      arch::OpClassTensorOp, 2, Operator>;
  using MmaCore1 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape1, WarpShape1, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB1, ElementAccumulator, layout::RowMajor,
      arch::OpClassTensorOp, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA0 =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore0::Shape::kM, MmaCore0::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore0::IteratorThreadMapA, kAlignmentA>;

  // Define iterators over tiles from the B operand
  using IteratorB0 =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore0::Shape::kK, MmaCore0::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore0::IteratorThreadMapB, kAlignmentB>;

  // Use fragment iterator for A operand
  using AccumulatorLayout = cutlass::layout::ColumnMajor;
  using FragmentIteratorA1 =
      cutlass::gemm::warp::MmaTensorOpFragmentIterator<
          cutlass::MatrixShape<MmaCore1::WarpShape::kM, MmaCore1::InstructionShape::kK>, //warp shape
          cutlass::MatrixShape<MmaCore0::WarpShape::kM, MmaCore0::WarpShape::kN>, //accumulator shape
          MmaCore1::Shape::kK, //kBlocksColumn
          ElementAccumulator, ElementA, AccumulatorLayout, InstructionShape, EpilogueOutputOp>;

  using ElementScaleBias = typename EpilogueOutputOp::ElementCompute;
  using LayoutScaleBias = layout::RowMajor; //vector layout doesn't really matter
  static int const kElementsPerAccess = 2;
  using IteratorAccumulatorScaleBias =
    cutlass::transform::threadblock::VectorIterator<
      cutlass::transform::threadblock::PredicatedVectorAccessIterator<
          cutlass::MatrixShape<ThreadblockShape0::kM, ThreadblockShape0::kN>,
          cutlass::MatrixShape<WarpShape1::kM, WarpShape1::kK>,
          ElementScaleBias, LayoutScaleBias, kElementsPerAccess>
    >;

  // Warp-level iterators to load scale and bias vectors
  using FragmentIteratorA1ScaleBias = cutlass::transform::warp::VectorFragmentIterator<
      MatrixShape<1, IteratorAccumulatorScaleBias::Fragment::kElements>, ElementScaleBias,
      LayoutScaleBias, InstructionShape, kElementsPerAccess>;

  // Define iterators over tiles from the B operand
  using IteratorB1 =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore1::Shape::kK, MmaCore1::Shape::kN>,
          ElementB, LayoutB1, 0, typename MmaCore1::IteratorThreadMapB, kAlignmentB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockB2bMma = cutlass::gemm::threadblock::B2bMmaPipelined<
      typename MmaCore0::Shape, IteratorA0, typename MmaCore0::SmemIteratorA,
      IteratorB0, typename MmaCore0::SmemIteratorB,
      typename MmaCore1::Shape, FragmentIteratorA1,
      IteratorAccumulatorScaleBias, FragmentIteratorA1ScaleBias,
      IteratorB1, typename MmaCore1::SmemIteratorB,
      ElementC, ElementAccumulator, layout::RowMajor,
      EpilogueOutputOp,
      typename MmaCore0::MmaPolicy, typename MmaCore1::MmaPolicy>;

};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major output for multi-stage
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Layout type for B1 matrix operand
    typename LayoutB1,
    /// Element type for output
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape0,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape1,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape0,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape1,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Apply upper triangular causal mask after first gemm
    bool CausalMaskAfterGemm0,
    /// Epilogue output operator
    typename EpilogueOutputOp>
struct DefaultB2bMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, LayoutB1, ElementC, ElementAccumulator, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag,
                  ThreadblockShape0, ThreadblockShape1,
                  WarpShape0, WarpShape1,
                  InstructionShape, Stages, Operator, CausalMaskAfterGemm0, EpilogueOutputOp, false> {
  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;


  // Define the MmaCore components
  using MmaCore0 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape0, WarpShape0, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, Operator, false, CacheOpA, CacheOpB>;
  using MmaCore1 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape1, WarpShape1, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB1, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  using ThreadMapA0 = typename MmaCore0::IteratorThreadMapA;
  using AccessTypeA0 = cutlass::Array<ElementA, kAlignmentA>;
  using IteratorA0 =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape0::kM, ThreadblockShape0::kK>,
          ElementA, LayoutA, 1, ThreadMapA0, AccessTypeA0>;

  // Define iterators over tiles from the B operand
  using ThreadMapB0 = typename MmaCore0::IteratorThreadMapB;
  using AccessTypeB0 = cutlass::Array<ElementB, kAlignmentB>;
  using IteratorB0 =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape0::kK, ThreadblockShape0::kN>,
          ElementB, LayoutB, 0, ThreadMapB0, AccessTypeB0>;

  // Use fragment iterator for A operand
  using AccumulatorLayout = cutlass::layout::ColumnMajor;
  // FragmentIteratorA1 should just load A1 fragments from the intermediate
  // accumulator tile without modification, so LinearCombination is used to
  // apply a no-op to the accumulator tile.
  using LinearCombinationOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    EpilogueOutputOp::kCount,
    ElementAccumulator,
    ElementC,
    cutlass::epilogue::thread::ScaleType::Nothing
  >;
  using FragmentIteratorA1 =
      cutlass::gemm::warp::MmaTensorOpFragmentIterator<
          cutlass::MatrixShape<MmaCore1::WarpShape::kM, MmaCore1::InstructionShape::kK>, //warp shape
          cutlass::MatrixShape<MmaCore0::WarpShape::kM, MmaCore0::WarpShape::kN>, //accumulator shape
          MmaCore1::Shape::kK, //kBlocksColumn
          ElementAccumulator, ElementA, AccumulatorLayout, InstructionShape, LinearCombinationOutputOp>;

  /// Define iterators over tiles from scale/bias vectors
  using ElementScaleBias = typename EpilogueOutputOp::ElementCompute;
  using LayoutScaleBias = layout::RowMajor; //vector layout doesn't really matter
  static int const kElementsPerAccess = 2;
  using IteratorAccumulatorScaleBias =
    cutlass::transform::threadblock::VectorIterator<
      cutlass::transform::threadblock::PredicatedVectorAccessIterator<
          cutlass::MatrixShape<ThreadblockShape0::kM, ThreadblockShape0::kN>,
          cutlass::MatrixShape<WarpShape1::kM, WarpShape1::kK>,
          ElementScaleBias, LayoutScaleBias, kElementsPerAccess>
    >;

  // Warp-level iterators to load scale and bias vectors
  using FragmentIteratorA1ScaleBias = cutlass::transform::warp::VectorFragmentIterator<
      MatrixShape<1, IteratorAccumulatorScaleBias::Fragment::kElements>, ElementScaleBias,
      LayoutScaleBias, InstructionShape, kElementsPerAccess>;


  // Define iterators over tiles from the B operand
  using ThreadMapB1 = typename MmaCore1::IteratorThreadMapB;
  using AccessTypeB1 = cutlass::Array<ElementB, kAlignmentB>;
  using IteratorB1 =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape1::kK, ThreadblockShape1::kN>,
          ElementB, LayoutB1, 0, ThreadMapB1, AccessTypeB1>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockB2bMma = cutlass::gemm::threadblock::B2bMmaMultistage<
      typename MmaCore0::Shape, IteratorA0, typename MmaCore0::SmemIteratorA,
      MmaCore0::kCacheOpA,
      IteratorB0, typename MmaCore0::SmemIteratorB, MmaCore0::kCacheOpB,
      typename MmaCore1::Shape, FragmentIteratorA1,
      IteratorAccumulatorScaleBias, FragmentIteratorA1ScaleBias,
      IteratorB1, typename MmaCore1::SmemIteratorB, MmaCore1::kCacheOpB,
      ElementC, ElementAccumulator, layout::RowMajor,
      EpilogueOutputOp,
      typename MmaCore0::MmaPolicy, typename MmaCore1::MmaPolicy, Stages,
      CausalMaskAfterGemm0, typename MmaCore0::WarpShape>;

};


////////////////////////////////////////////////////////////////////////////////


} // namespace threadblock
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
