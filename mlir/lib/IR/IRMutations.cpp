
#include "mlir/IR/IRMutations.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
/// MutationBase
//===----------------------------------------------------------------------===//
MutationBase::MutationBase() { transform_tag = ""; }

MutationBase::MutationBase(std::string transform_tag)
    : transform_tag(transform_tag) {
  prev_op_ids.clear();
  cur_op_ids.clear();
}

void MutationBase::addPrevOpId(const OpOrValueLocId &op_or_value_loc_id) {
  prev_op_ids.push_back(op_or_value_loc_id);
}

void MutationBase::addCurOpId(const OpOrValueLocId &op_or_value_loc_id) {
  cur_op_ids.push_back(op_or_value_loc_id);
}

std::vector<OpOrValueLocId> MutationBase::getPrevOpIds() { return prev_op_ids; }

std::vector<OpOrValueLocId> MutationBase::getCurOpIds() { return cur_op_ids; }

std::string MutationBase::getTransformTag() { return transform_tag; }

void MutationBase::setTransformTag(std::string transform_tag) {
  this->transform_tag = transform_tag;
}

void MutationBase::print() {
  for (size_t i = 0; i < prev_op_ids.size(); i++) {
    if (i > 0 && i == prev_op_ids.size() - 1) {
      llvm::outs() << " and ";
    } else if (i > 0) {
      llvm::outs() << " , ";
    }
    llvm::outs() << prev_op_ids[i].first << " at " << prev_op_ids[i].second;
  }

  if (!prev_op_ids.empty()) {
    printInternal();
  } else {
    llvm::outs() << "\n\n";
  }

  for (size_t i = 0; i < cur_op_ids.size(); i++) {
    llvm::outs() << " to form ";

    if (i > 0 && i == cur_op_ids.size() - 1) {
      llvm::outs() << " and ";
    } else if (i > 0) {
      llvm::outs() << " , ";
    }
    llvm::outs() << cur_op_ids[i].first << " at " << cur_op_ids[i].second;
  }
  llvm::outs() << "\n\n";
}

//===----------------------------------------------------------------------===//
/// MutationFactory
//===----------------------------------------------------------------------===//
std::unique_ptr<MutationBase>
MutationFactory::createIdentityMutation(const OpOrValueLocId &op_id) {
  std::unique_ptr<MutationBase> mutation(new IdentityMut);
  mutation->addPrevOpId(op_id);
  return mutation;
}

std::unique_ptr<MutationBase>
MutationFactory::createDeleteMutation(const OpOrValueLocId &op_id) {
  std::unique_ptr<MutationBase> mutation(new DeleteMut);
  mutation->addPrevOpId(op_id);
  return mutation;
}

std::unique_ptr<MutationBase>
MutationFactory::createMoveMutation(StringRef op_name, Location from,
                                    Location to) {
  std::unique_ptr<MutationBase> mutation(new MoveMut);
  mutation->addPrevOpId(std::make_pair(op_name, from));
  mutation->addCurOpId(std::make_pair(op_name, to));
  return mutation;
}

std::unique_ptr<MutationBase>
MutationFactory::createInlineMutation(StringRef op_name, Location callee,
                                      Location caller) {
  std::unique_ptr<MutationBase> mutation(new InlineMut);
  mutation->addPrevOpId(std::make_pair(op_name, callee));
  mutation->addCurOpId(std::make_pair(op_name, caller));
  return mutation;
}

std::unique_ptr<MutationBase>
MutationFactory::createConversionMutation(StringRef op_name_from,
                                          StringRef op_name_to, Location loc) {
  std::unique_ptr<MutationBase> mutation(new ConversionMut);
  mutation->addPrevOpId(std::make_pair(op_name_from, loc));
  mutation->addCurOpId(std::make_pair(op_name_to, loc));
  return mutation;
}

std::unique_ptr<MutationBase>
MutationFactory::createUnFusedMutation(const OpOrValueLocId &op_id_from,
                                       const OpOrValueLocId &op_id_to) {
  std::unique_ptr<MutationBase> mutation(new UnFusedMut);
  mutation->addPrevOpId(op_id_from);
  mutation->addCurOpId(op_id_to);
  return mutation;
}

std::unique_ptr<MutationBase> MutationFactory::createFusedMutation(
    const std::vector<OpOrValueLocId> &fused_from,
    const OpOrValueLocId &fused_to) {
  std::unique_ptr<MutationBase> mutation(new FusedMut);
  for (auto &&op_id : fused_from) {
    mutation->addPrevOpId(op_id);
  }
  mutation->addCurOpId(fused_to);
  return mutation;
}

//===----------------------------------------------------------------------===//
/// IRMutationManager
//===----------------------------------------------------------------------===//
void IRMutationManager::reset(Operation *op, const std::string &tag) {
  clear();
  Operation *op_previous = op->clone(ir_mapping);
  // Map the locations to Ops from the original version of the
  // module. The locations in this module are guaranteed to be unique as they
  // are re-numbered just befor this function call.
  op_previous->walk([&](Operation *opIt) {
    loc_to_op_map.insert({opIt->getLoc(), opIt});
    op_to_loc_map.insert({opIt, opIt->getLoc()});
  });
}

IRMutationManager::~IRMutationManager() { clear(); }

std::vector<std::unique_ptr<MutationBase>>
IRMutationManager::getMutations(Operation *op_current) {
  llvm::outs() << "Printing IR Mutations occured in- " << transform_tag << "\n";
  // Create a map of locations to ops for the newly added Op in the
  // module. The locations in this module are need not be unique as there
  // could have beed some module transformations
  DenseMap<Location, std::vector<Operation *>> loc_to_op_map_after;
  std::vector<Operation *> newly_inserted_ops;
  std::vector<Operation *> old_mutated_ops;

  std::vector<std::unique_ptr<MutationBase>> mutations;

  op_current->walk([&](Operation *opIt) {
    // If an op exists BEFORE and AFTER, it can still have mutaions, like
    // arguments and resulsts
    if (Operation *op = ir_mapping.lookupOrNull(opIt)) {
      bool is_op_mutated = false;
      // The information related to 1.1 and 1.2 is not available directly from
      // the Source-Location information. However, we should be able to
      // evaluate these mutations as we have access to versions of Op/Module
      // before and after the transform and we should be able to apply
      // equivalance comparisions like this easily-
      // llvm-project/mlir/lib/IR/OperationSupport.cpp;l=704-783
      // OperationEquivalence::isEquivalentTo

      // 1.1. Check if arguments are mutated

      // 1.2. Check if Op result is mutated

      // 1.3. Check if Locations match
      if (!is_op_mutated && op->getLoc() == opIt->getLoc()) {
        mutations.push_back(MutationFactory::createIdentityMutation(
            std::make_pair(op->getName().getStringRef(), op->getLoc())));
      }
      // 1.4. Check if Locations do not match. If the locations do not match,
      // it means the op has been moved
      if (!is_op_mutated && op->getLoc() != opIt->getLoc()) {
        mutations.push_back(MutationFactory::createMoveMutation(
            op->getName().getStringRef(), op->getLoc(), opIt->getLoc()));
      }
      // If the Op is present BEFORE and AFTER, delete it from the
      // loc_to_op_map_before and op_to_loc_map maps, as these maps
      // will be used to track the deleted Ops and other kinds of mutations
      op_to_loc_map.erase(op);
    } else {
      if (loc_to_op_map_after.find(opIt->getLoc()) ==
          loc_to_op_map_after.end()) {
        loc_to_op_map_after.insert({opIt->getLoc(), {}});
      }

      loc_to_op_map_after[opIt->getLoc()].push_back(opIt);
      newly_inserted_ops.push_back(opIt);
    }
  });

  for (Operation *opIt : newly_inserted_ops) {
    // If the Op didn't exist BEFORE and exists AFTER the transform, it is
    // probably newly introduce but still may have been derived from an Op in
    // BEFORE.
    if (loc_to_op_map_after[opIt->getLoc()].size() > 1 &&
        loc_to_op_map.find(opIt->getLoc()) != loc_to_op_map.end()) {
      Operation *op = loc_to_op_map[opIt->getLoc()];
      // 2.1. If more than one Op in AFTER have the same location as an Op
      // in BEFORE, its probably an outcome of an unroll action.
      mutations.push_back(MutationFactory::createUnFusedMutation(
          std::make_pair(op->getName().getStringRef(), op->getLoc()),
          std::make_pair(opIt->getName().getStringRef(), opIt->getLoc())));
      old_mutated_ops.push_back(op);
    } else if (loc_to_op_map.find(opIt->getLoc()) != loc_to_op_map.end()) {
      // 2.2. Check if the Op in AFTER has the same location as an Op in
      // BEFORE. Its probably a result of 1->1 conversion pattern. Ex. TF ->
      // TFL Delete the converted Op from the loc_to_op_map and
      // op_to_loc_map maps
      Operation *op = loc_to_op_map[opIt->getLoc()];
      mutations.push_back(MutationFactory::createConversionMutation(
          op->getName().getStringRef(), opIt->getName().getStringRef(),
          opIt->getLoc()));
      old_mutated_ops.push_back(op);
    } else if (auto fused_loc = opIt->getLoc().dyn_cast<FusedLoc>()) {
      // 2.3. Check if the Op in AFTER is a result of fusion. Get the list of
      // fused locations in BEFORE. Delete the fused Ops from the
      // loc_to_op_map and op_to_loc_map maps
      std::vector<OpOrValueLocId> op_ids;
      for (size_t loc_idx = 0; loc_idx < fused_loc.getLocations().size();
           ++loc_idx) {
        Operation *op = loc_to_op_map[fused_loc.getLocations()[loc_idx]];
        op_ids.push_back(
            std::make_pair(op->getName().getStringRef(), op->getLoc()));
        old_mutated_ops.push_back(op);
      }
      mutations.push_back(MutationFactory::createFusedMutation(
          op_ids,
          std::make_pair(opIt->getName().getStringRef(), opIt->getLoc())));

    } else if (auto callsite_loc = opIt->getLoc().dyn_cast<CallSiteLoc>()) {
      // 2.3. Check if the Op in AFTER is a result of inlining. Get the
      // original inlined location in BEFORE. Delete the fused Ops from the
      // loc_to_op_map and op_to_loc_map maps
      Operation *op = loc_to_op_map[callsite_loc.getCallee()];
      mutations.push_back(MutationFactory::createInlineMutation(
          opIt->getName().getStringRef(), callsite_loc.getCallee(),
          callsite_loc.getCaller()));
      old_mutated_ops.push_back(op);
    }
    for (Operation *m_op : old_mutated_ops) {
      op_to_loc_map.erase(m_op);
    }
  }

  if (!op_to_loc_map.empty()) {
    // There are some Ops BEFORE that have not been accounted for in AFTER,
    // consider them deleted?
    for (auto deleted_op_loc_pair : op_to_loc_map) {
      Operation *deleted_op = deleted_op_loc_pair.first;
      mutations.push_back(MutationFactory::createDeleteMutation(std::make_pair(
          deleted_op->getName().getStringRef(), deleted_op->getLoc())));
    }
  }
  return mutations;
}

void IRMutationManager::clear() {
  ir_mapping.clear();
  loc_to_op_map.clear();
  op_to_loc_map.clear();
}
