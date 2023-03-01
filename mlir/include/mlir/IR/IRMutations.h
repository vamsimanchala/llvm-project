//===- IRMutations.h - IR Mutations Utilities ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file several utility methods for snapshotting the current IR to
// produce new debug locations.
//
//===----------------------------------------------------------------------===//

#ifndef THIRD_PARTY_LLVM_LLVM_PROJECT_MLIR_INCLUDE_MLIR_IR_IRMUTATIONS_H_
#define THIRD_PARTY_LLVM_LLVM_PROJECT_MLIR_INCLUDE_MLIR_IR_IRMUTATIONS_H_

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace mlir {

using OpOrValueLocId = std::pair<StringRef, Location>;

/// Base class to hold a single mutation object. This can be extended by
/// different classes to represent specific mnutation types and hold any
/// mutation specific information or metedata. For example- In a mutation object
/// to used to record OpArgMutation, it may be useful to know index of the
/// mutated openrand.
class MutationBase {
 public:
  MutationBase();

  virtual ~MutationBase() = default;

  MutationBase(std::string transform_tag);

  void addPrevOpId(const OpOrValueLocId &op_or_value_loc_id);

  void addCurOpId(const OpOrValueLocId &op_or_value_loc_id);

  std::vector<OpOrValueLocId> getPrevOpIds();

  std::vector<OpOrValueLocId> getCurOpIds();

  std::string getTransformTag();

  void setTransformTag(std::string transform_tag);

  void print();

 private:
  virtual void printInternal() = 0;

  std::vector<OpOrValueLocId> prev_op_ids;
  std::vector<OpOrValueLocId> cur_op_ids;
  std::string transform_tag;
};

class FusedMut : public MutationBase {
 private:
  void printInternal() override { llvm::outs() << " are fused"; };
};

class UnFusedMut : public MutationBase {
 private:
  void printInternal() override { llvm::outs() << " is un-fused"; };
};

class InlineMut : public MutationBase {
 private:
  void printInternal() override { llvm::outs() << " is inlined"; };
};

class MoveMut : public MutationBase {
 private:
  void printInternal() override { llvm::outs() << " is moved"; };
};

class DeleteMut : public MutationBase {
 private:
  void printInternal() override { llvm::outs() << " is deleted"; };
};

class IdentityMut : public MutationBase {
 private:
  void printInternal() override { llvm::outs() << " remained unmutated"; };
};

class ConversionMut : public MutationBase {
 private:
  void printInternal() override { llvm::outs() << " converted"; };
};

/// Utility class to create the specific Mutation objects
class MutationFactory {
 public:
  static std::unique_ptr<MutationBase>
  createIdentityMutation(const OpOrValueLocId &op_id);

  static std::unique_ptr<MutationBase>
  createDeleteMutation(const OpOrValueLocId &op_id);

  static std::unique_ptr<MutationBase>
  createMoveMutation(StringRef op_name, Location from, Location to);

  static std::unique_ptr<MutationBase>
  createInlineMutation(StringRef op_name, Location callee, Location caller);

  static std::unique_ptr<MutationBase>
  createConversionMutation(StringRef op_name_from, StringRef op_name_to,
                           Location loc);

  static std::unique_ptr<MutationBase>
  createUnFusedMutation(const OpOrValueLocId &op_id_from,
                        const OpOrValueLocId &op_id_to);

  static std::unique_ptr<MutationBase>
  createFusedMutation(const std::vector<OpOrValueLocId> &fused_from,
                      const OpOrValueLocId &fused_to);
};

/// Utility class that can be used to get the list of MutationBase objects after
/// a pass, pipeline or a rewrite
class IRMutationManager {
 public:
  IRMutationManager() = default;
  void reset(Operation *op_previous, const std::string& transform_tag);

  ~IRMutationManager();

  std::vector<std::unique_ptr<MutationBase>>
  getMutations(Operation *op_current);

 private:
  void clear();

  std::string transform_tag;
  mlir::IRMapping ir_mapping;
  DenseMap<Location, Operation *> loc_to_op_map;
  DenseMap<Operation *, Location> op_to_loc_map;
};

} // namespace mlir

#endif // THIRD_PARTY_LLVM_LLVM_PROJECT_MLIR_INCLUDE_MLIR_IR_IRMUTATIONS_H_
