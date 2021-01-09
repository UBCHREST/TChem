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
#ifndef __TCHEM_IMPL_FORCING_AND_JACOBIAN_MATRIX_HPP__
#define __TCHEM_IMPL_FORCING_AND_JACOBIAN_MATRIX_HPP__

#include "TChem_Util.hpp"

#include "TChem_Impl_IgnitionZeroD_Problem.hpp"
#include "TChem_Impl_ForcingMatrix_Problem.hpp"

namespace TChem {
namespace Impl {

struct ForcingAndJacobianMatrix
{
  template<typename KineticModelConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd)
  {
    using problem_type =
      TChem::Impl::IgnitionZeroD_Problem<KineticModelConstDataType>;
    problem_type problem;
    problem._kmcd = kmcd;

    // return TimeIntegrator::getWorkSpaceSize(problem);
  }

  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    const real_type& pressure,      /// pressure
    const RealType1DViewType& vals, /// mass fraction (kmcd.nSpec)
    const RealType1DViewType& facL, /// numerica jacobian percentage
    const RealType1DViewType& facF, /// numerica jacobian percentage
    //output
    const RealType2DViewType& L,
    const RealType2DViewType& F,
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {
    real_type one(1);

    using problem_type_jac =
      TChem::Impl::IgnitionZeroD_Problem<KineticModelConstDataType>;
    problem_type_jac problem_jac;

    /// problem workspace
    const ordinal_type problem_workspace_size =
      problem_type_jac::getWorkSpaceSize(kmcd);
    auto wptr = work.data();
    auto pw = typename problem_type_jac::real_type_1d_view_type(
      wptr, problem_workspace_size);
    wptr += problem_workspace_size;

    /// initialize problem
    problem_jac._p = pressure; // pressure
    problem_jac._work = pw;    // problem workspace array
    problem_jac._kmcd = kmcd;  // kinetic model
    problem_jac._fac = facL;    // fac for numerical jacobian

    //computes jacobian
    problem_jac.computeJacobian(member,vals,L);

    using problem_type_forcing =
      TChem::Impl::ForcingMatrix_Problem<KineticModelConstDataType>;
    problem_type_forcing problem_forcing;

    /// problem workspace
    const ordinal_type problem_workspace_size_forcing =
      problem_type_forcing::getWorkSpaceSize(kmcd);

    //
    auto pwL = typename problem_type_forcing::real_type_1d_view_type(
      wptr, problem_workspace_size_forcing);
    wptr += problem_workspace_size_forcing;

    //
    auto alpha = RealType1DViewType(wptr, kmcd.nReac);
    wptr += kmcd.nReac;

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, kmcd.nSpec),
                         [&](const ordinal_type& i) { alpha(i) = one; });
    /// error check
    const ordinal_type workspace_used(wptr - work.data()),
      workspace_extent(work.extent(0));
    if (workspace_used > workspace_extent) {
      Kokkos::abort("Error: workspace used is larger than it is provided\n");
    }

    /// initialize problem
    problem_forcing._p = pressure; // pressure
    problem_forcing._temp = vals(0);
    for (ordinal_type i = 0; i < kmcd.nSpec; i++) {
      problem_forcing._Ys(i) = vals(1+i);
    }
    problem_forcing._work = pwL;    // problem workspace array
    problem_forcing._kmcd = kmcd;  // kinetic model
    problem_forcing._fac = facF;    // fac for numerical jacobian

    /// problem workspace
    const ordinal_type problem_workspace_size_F =
      problem_type_forcing::getWorkSpaceSize(kmcd);


    problem_jac.computeJacobian(member, alpha, F);

  }

};

} // namespace Impl
} // namespace TChem

#endif
