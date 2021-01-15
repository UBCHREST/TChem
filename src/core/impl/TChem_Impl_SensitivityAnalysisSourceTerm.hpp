/* =====================================================================================
TChem version 2.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

This file is part of TChem. TChem is open source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the licese is also
provided under the main directory

Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */
#ifndef __TCHEM_IMPL_SENSITIVITY_ANALYSIS_SOURCE_HPP__
#define __TCHEM_IMPL_SENSITIVITY_ANALYSIS_SOURCE_HPP__

#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_Impl_EnthalpySpecMs.hpp"
#include "TChem_Impl_MolarConcentrations.hpp"
#include "TChem_Impl_ReactionRates.hpp"
#include "TChem_Impl_RhoMixMs.hpp"
#include "TChem_Util.hpp"
#include "TChem_Impl_ForcingAndJacobianMatrix.hpp"

#define TCHEM_ENABLE_SERIAL_TEST_OUTPUT
namespace TChem {
namespace Impl {

struct SensitivityAnalysisSourceTerm
{
  template<typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd)
  {
    const ordinal_type workspace_size = (kmcd.nSpec *kmcd.nSpec
      + kmcd.nSpec*kmcd.nReac + ForcingAndJacobianMatrix::getWorkSpaceSize(kmcd));
    return workspace_size;
  }

  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& p,
    const RealType1DViewType& vals, /// (kmcd.nSpec)
    const RealType1DViewType& alpha,
    // outputs
    const RealType1DViewType& source, // rhs 
    /// workspace
    const RealType1DViewType& facL,
    const RealType1DViewType& facF,
    const RealType2DViewType& L,
    const RealType2DViewType& F,
    const RealType1DViewType& w,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {

    /// 1. compute L and F matrices
    ForcingAndJacobianMatrix::team_invoke(member,
                                       p,
                                       vals,
                                       alpha,
                                       facL,
                                       facF,
                                       L,
                                       F,
                                       w,
                                       kmcd);
    member.team_barrier();

    // Here








  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& p,
    const RealType1DViewType& vals, /// (kmcd.nSpec)
    const RealType1DViewType& alpha,
    /// output
    const RealType1DViewType& source, /// (kmcd.nSpec + 1)
    const RealType1DViewType& facL,
    const RealType1DViewType& facF,
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    // const real_type zero(0);

    ///
    auto w = (real_type*)work.data();

    auto L = Kokkos::View<real_type**,
                             Kokkos::LayoutRight,
                             typename WorkViewType::memory_space>(
      w, kmcd.nSpec, kmcd.nSpec);
    w += kmcd.nSpec *kmcd.nSpec;

    auto F = Kokkos::View<real_type**,
                             Kokkos::LayoutRight,
                             typename WorkViewType::memory_space>(
      w, kmcd.nSpec, kmcd.nReac);

    w += kmcd.nSpec*kmcd.nReac;

    const ordinal_type workspace_used(w - work.data()),
      workspace_extent(work.extent(0));
    if (workspace_used > workspace_extent) {
      Kokkos::abort("Error: workspace used is larger than it is provided\n");
    }

    auto workFL = RealType1DViewType(w, workspace_extent - workspace_used);

    team_invoke_detail(member,
                       p,
                       vals,
                       alpha,
                       source,
                       /// workspace
                       facL,
                       facF,
                       L,
                       F,
                       workFL,
                       kmcd);

//
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("SensitivityAnalysisSourceTerm.team_invoke.test.out", "a+");
      fprintf(fs, ":: SensitivityAnalysisSourceTerm::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, nReac %3d, t %e, p %e\n",
              kmcd.nSpec,
              kmcd.nReac,
              vals(0),
              p);
      fprintf(fs, "mass fraction :: input\n");
      for (ordinal_type sp = 1; sp < kmcd.nSpec+1; sp++) {
        fprintf(fs,"sp %d % e \n", sp, vals(sp) );
      }
      fprintf(fs, "L :: output\n");
      for (int i = 0; i < int(L.extent(0)); ++i){
        for (int j = 0; j < int(L.extent(1)); j++) {
          fprintf(fs,"%e ",L(i,j) );
        }
        fprintf(fs,"\n");
      }

      fprintf(fs, "F :: output\n");
      for (int i = 0; i < int(F.extent(0)); ++i){
        for (int j = 0; j < int(F.extent(1)); j++) {
          fprintf(fs,"%e ",F(i,j) );
        }
        fprintf(fs,"\n");
      }

    }
#endif
  }
};



} // namespace Impl
} // namespace TChem

#endif
