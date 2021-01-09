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

namespace TChem {
namespace Impl {

struct SensitivityAnalysisSourceTerm
{
  template<typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd)
  {
    const ordinal_type workspace_size = (5 * kmcd.nSpec + 8 * kmcd.nReac);
    return workspace_size;
  }

  ///
  ///  \param t  : temperature [K]
  ///  \param Xc : array of \f$N_{spec}\f$ doubles \f$((XC_1,XC_2,...,XC_N)\f$:
  ///              molar concentrations XC \f$[kmol/m^3]\f$
  ///  \return omega : array of \f$N_{spec}\f$ molar reaction rates
  ///  \f$\dot{\omega}_i\f$ \f$\left[kmol/(m^3\cdot s)\right]\f$
  ///
  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& p,
    const RealType1DViewType& vals, /// (kmcd.nSpec)
    // outputs
    const RealType1DViewType& source,
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
                                       facL,
                                       facF,
                                       L,
                                       F,
                                       w,
                                       kmcd);

    member.team_barrier();
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
      w, kmcd.nSpec, kmcd.nReac);
    w += kmcd.nReac *kmcd.nReac;

    auto F = Kokkos::View<real_type**,
                             Kokkos::LayoutRight,
                             typename WorkViewType::memory_space>(
      w, kmcd.nSpec, kmcd.nSpec);

    w += kmcd.nSpec*kmcd.nSpec;

    auto workFL = RealType1DViewType(w, kmcd.nSpec);
    w += kmcd.nSpec;

      team_invoke_detail(member,
                       p,
                       vals,
                       source,
                       /// workspace
                       facL,
                       facF,
                       L,
                       F,
                       workFL,
                       kmcd);
  }
};

} // namespace Impl
} // namespace TChem

#endif
