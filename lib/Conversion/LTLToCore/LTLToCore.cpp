//===- LTLToCore.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts LTL and Verif operations to Core operations
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LTLToCore.h"
#include "circt/Conversion/HWToSV.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"

#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"

namespace circt {
#define GEN_PASS_DEF_LOWERLTLTOCORE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

struct LTLImplicationConversion
    : public OpConversionPattern<ltl::ImplicationOp> {
  using OpConversionPattern<ltl::ImplicationOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::ImplicationOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // The logical rule: A -> B becomes (!A || B)
    // The operands of the original op are available in the 'adaptor'.
    Value a = op.getAntecedent(); // Left-hand side (A)
    Value b = op.getConsequent(); // Right-hand side (B)

    // Create !A. In hardware, this is typically an XOR with a constant 1.
    Location loc = op.getLoc();
    Value constOne =
        rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(), 1);
    Value notA = rewriter.create<comb::XorOp>(loc, a, constOne);

    // Create (!A || B)
    Value orResult = rewriter.create<comb::OrOp>(loc, notA, b);

    // Replace the original ltl.implication op with the result of the OrOp.
    rewriter.replaceOp(op, orResult);

    return success();
  }
};

struct AssertOpConversion : public OpConversionPattern<verif::AssertOp> {
  using OpConversionPattern<verif::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The `adaptor` provides the operands of the original op *after* they
    // have been converted by other patterns. In this case,
    // adaptor.getProperty() will be the `i1` result from your
    // LTLImplicationConversion.
    Value newProperty = adaptor.getProperty();

    // If the type is already what we want (i1), there's nothing to do.
    // This check is important to avoid infinite recursion if the op is already
    // legal. if (newProperty.getType() == op.getProperty().getType())
    //   return failure();
    llvm::outs() << "Converting AssertOp: " << op << "\n";
    // Value newProperty =
    // op.getProperty().getDefiningOp<mlir::UnrealizedConversionCastOp>().getInputs().front();
    llvm::outs() << "newProperty: " << newProperty << "\n";
    // Create a new `verif.AssertOp` with the same attributes but with the
    // new, converted `i1` property.
    rewriter.replaceOpWithNewOp<verif::AssertOp>(
        op, newProperty, /*enable=*/Value(), op.getLabelAttr());

    return success();
  }
};

struct CombinationalConverter
    : public OpConversionPattern<llhd::CombinationalOp> {
  using OpConversionPattern<llhd::CombinationalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::CombinationalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs() << "cloning: " << op << "\n";

    rewriter.inlineBlockBefore(&op.getBody().front(), op, {});
    op.erase();
    return success();
  }
};
struct YieldConverter : public OpConversionPattern<llhd::YieldOp> {
  using OpConversionPattern<llhd::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs() << "erasing: " << op << "\n";
    op.erase();
    return success();
  }
};
struct DelayConverter : public OpConversionPattern<llhd::DelayOp> {
  using OpConversionPattern<llhd::DelayOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::DelayOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs() << "erasing: " << op << "\n";
    if (op.getDelay().getEpsilon() != 0) {
      op->emitError() << "Delay with non-zero epsilon is not supported";
      return failure();
    }
    Value clock;
    for (auto arg : op->getBlock()->getArguments()) {
      if (isa<seq::ClockType>(arg.getType())) {
        clock = arg;
        break;
      }
    }
    op.erase();
    return success();
  }
};
struct HasBeenResetOpConversion : OpConversionPattern<verif::HasBeenResetOp> {
  using OpConversionPattern<verif::HasBeenResetOp>::OpConversionPattern;

  // HasBeenReset generates a 1 bit register that is set to one once the reset
  // has been raised and lowered at at least once.
  LogicalResult
  matchAndRewrite(verif::HasBeenResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i1 = rewriter.getI1Type();
    // Generate the constant used to set the register value
    Value constZero = seq::createConstantInitialValue(
        rewriter, op->getLoc(), rewriter.getIntegerAttr(i1, 0));

    // Generate the constant used to negate the reset value
    Value constOne = hw::ConstantOp::create(rewriter, op.getLoc(), i1, 1);

    // Create a backedge for the register to be used in the OrOp
    circt::BackedgeBuilder bb(rewriter, op.getLoc());
    circt::Backedge reg = bb.get(rewriter.getI1Type());

    // Generate an or between the reset and the register's value to store
    // whether or not the reset has been active at least once
    Value orReset =
        comb::OrOp::create(rewriter, op.getLoc(), adaptor.getReset(), reg);

    // This register should not be reset, so we give it dummy reset and resetval
    // operands to fit the build signature
    Value reset, resetval;

    // Finally generate the register to set the backedge
    reg.setValue(seq::CompRegOp::create(
        rewriter, op.getLoc(), orReset,
        rewriter.createOrFold<seq::ToClockOp>(op.getLoc(), adaptor.getClock()),
        rewriter.getStringAttr("hbr"), reset, resetval, constZero,
        InnerSymAttr{} // inner_sym
        ));

    // We also need to consider the case where we are currently in a reset cycle
    // in which case our hbr register should be down-
    // Practically this means converting it to (and hbr (not reset))
    Value notReset = comb::XorOp::create(rewriter, op.getLoc(),
                                         adaptor.getReset(), constOne);
    rewriter.replaceOpWithNewOp<comb::AndOp>(op, reg, notReset);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Lower LTL To Core pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerLTLToCorePass
    : public circt::impl::LowerLTLToCoreBase<LowerLTLToCorePass> {
  LowerLTLToCorePass() = default;
  void runOnOperation() override;
};
} // namespace

// Simply applies the conversion patterns defined above
void LowerLTLToCorePass::runOnOperation() {
  mlir::TypeConverter converter;

  // Set target dialects: We don't want to see any ltl or verif that might
  // come from an AssertProperty left in the result
  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<sv::SVDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  // target.addLegalDialect<ltl::LTLDialect>();
  target.addIllegalDialect<verif::VerifDialect>();
  target.addDynamicallyLegalOp<verif::AssertOp>([&](verif::AssertOp op) {
    // The operation is legal if the type of its property operand
    // is already what the type converter would produce.
    // In this case, we are converting properties to i1.
    return converter.isLegal(op.getProperty().getType());
  });
  target.addDynamicallyLegalOp<verif::AssumeOp>([&](verif::AssumeOp op) {
    // The operation is legal if the type of its property operand
    // is already what the type converter would produce.
    // In this case, we are converting properties to i1.
    return converter.isLegal(op.getProperty().getType());
  });
  target.addIllegalOp<verif::HasBeenResetOp>();
  target.addIllegalDialect<llhd::LLHDDialect>();
  target.addLegalDialect<mlir::cf::ControlFlowDialect>();
  // Create type converters, mostly just to convert an ltl property to a bool

  // Convert the ltl property type to a built-in type
  converter.addConversion([](IntegerType type) { return type; });
  converter.addConversion([](ltl::PropertyType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  converter.addConversion([](ltl::SequenceType type) {
    return IntegerType::get(type.getContext(), 1);
  });

  // Basic materializations
  converter.addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return Value();
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs[0])
            ->getResult(0);
      });

  converter.addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return Value();
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs[0])
            ->getResult(0);
      });

  // Create the operation rewrite patters
  RewritePatternSet patterns(&getContext());
  patterns.add<HasBeenResetOpConversion>(converter, patterns.getContext());
  patterns.add<LTLImplicationConversion>(converter, patterns.getContext());
  patterns.add<AssertOpConversion>(converter, patterns.getContext());
  patterns.add<CombinationalConverter>(converter, patterns.getContext());
  patterns.add<YieldConverter>(converter, patterns.getContext());
  patterns.add<DelayConverter>(converter, patterns.getContext());

  OpBuilder builder(&getContext());

  // Apply the conversions
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createLowerLTLToCorePass() {
  return std::make_unique<LowerLTLToCorePass>();
}
