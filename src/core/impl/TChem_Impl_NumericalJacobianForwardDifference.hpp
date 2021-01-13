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
#ifndef __TCHEM_IMPL_NUMERICAL_JACOBIAN_FORWARD_DIFFERENCE_HPP__
#define __TCHEM_IMPL_NUMERICAL_JACOBIAN_FORWARD_DIFFERENCE_HPP__
#define TCHEM_ENABLE_SERIAL_TEST_OUTPUT
namespace TChem {
namespace Impl {
///
/// J_{ij} = { df_i/dx_j }
///

struct NumericalJacobianForwardDifference
{
  template<typename MemberType,
           typename ProblemType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    const ProblemType& problem,
    const real_type& fac_min,
    const real_type& fac_max,
    const RealType1DViewType& fac,
    const RealType1DViewType& x,
    const RealType1DViewType& f_0,
    const RealType1DViewType& f_h,
    const RealType2DViewType& J)
  {
    const real_type eps = ats<real_type>::epsilon();
    const real_type eps_1_2 = ats<real_type>::sqrt(eps);     // U
    const real_type eps_1_4 = ats<real_type>::sqrt(eps_1_2); // bu
    const real_type eps_1_8 = ats<real_type>::sqrt(eps_1_4);
    const real_type eps_3_4 = eps_1_2 * eps_1_4; // bl
    const real_type eps_7_8 = eps / (eps_1_8);   // br
    const real_type zero(0), one(1), two(2);
    const real_type eps_2_1_2 = ats<real_type>::sqrt(two*eps);     // U
    const real_type fac_min_use = fac_min <= zero ? (eps_3_4) : fac_min;
    const real_type fac_max_use = fac_max <= zero ? (eps_2_1_2) : fac_max;


    const ordinal_type m = J.extent(0);
    const ordinal_type n = J.extent(1);


    /// initialization fac if necessary
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n),
                         [&](const ordinal_type& i) {
                           fac(i) = (fac(i) == zero ? eps_1_2 : fac(i));
                         });
    /// compute f_0
    problem.computeFunction(member, x, f_0);

    /// loop over columns
    for (ordinal_type j = 0; j < n; ++j) {
      /// keep x at i
      const real_type x_at_j = x(j);

      /// force fac between facmin and famax
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        const real_type fac_at_j = fac(j);
        fac(j) = (fac_at_j < fac_min_use
                    ? fac_min_use
                    : fac_at_j > fac_max_use ? fac_max_use : fac_at_j);
      });



      /// x scale value in case that x(j) is zero
      const real_type xs = x_at_j; //(x_at_j != zero ? x_at_j : one);
      const real_type h = ats<real_type>::abs(fac(j) * xs) + eps;

      /// modify x vector
      member.team_barrier();
      Kokkos::single(Kokkos::PerTeam(member), [&]() { x(j) = x_at_j + h; });

      /// compute f_h
      member.team_barrier();
      problem.computeFunction(member, x, f_h);

      /// roll back the input vector
      member.team_barrier();
      Kokkos::single(Kokkos::PerTeam(member), [&]() { x(j) = x_at_j; });

      /// compute jacobian at ith column
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, m),
        [&](const ordinal_type& i) { J(i, j) = (f_h(i) - f_0(i)) / h; });

      /// find location k
      ordinal_type k(0);
      {
        using reducer_value_type =
          typename Kokkos::MaxLoc<real_type, ordinal_type>::value_type;
        reducer_value_type value;
        Kokkos::MaxLoc<real_type, ordinal_type> reducer_value(value);
        Kokkos::parallel_reduce(
          Kokkos::TeamVectorRange(member, m),
          [&](const ordinal_type& i, reducer_value_type& update) {
            const real_type val = ats<real_type>::abs(f_h(i) - f_0(i));
            if (val > update.val) {
              update.val = val;
              update.loc = i;
            }
          },
          reducer_value);
        member.team_barrier();
        k = value.loc;
      }

      const real_type diff = ats<real_type>::abs(f_h(k) - f_0(k));
      const real_type abs_f_h_at_k = ats<real_type>::abs(f_h(k));
      const real_type abs_f_0_at_k = ats<real_type>::abs(f_0(k));
      const real_type scale =
        abs_f_h_at_k > abs_f_0_at_k ? abs_f_h_at_k : abs_f_0_at_k;
      const real_type check =
        abs_f_h_at_k < abs_f_0_at_k ? abs_f_h_at_k : abs_f_0_at_k;

      if (check == zero) {
        /// fac(i) is accepted and compute jacobian with fac change
      } else {
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          if (diff > eps_1_4 * scale) {
            /// truncation error is dominant; decrease fac
            fac(j) *= eps_1_2;
          } else if ((eps_7_8 * scale < diff) && (diff < eps_3_4 * scale)) {
            /// round off error is dominant; increase fac
            fac(j) /= eps_1_2;
          } else if (diff < eps_7_8 * scale) {
            /// round off error is dominant; increase fac rapidly
            fac(j) = ats<real_type>::sqrt(fac(j));
          } else {
            /// fac is not changed
          }
        });
        member.team_barrier();
      }

    }

//
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("NumericalJacobianForwardDifference.team_invoke.test.out", "a+");
      fprintf(fs, ":: NumericalJacobianForwardDifference::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs, "m %d n %d \n", m, n);
      fprintf(fs, "x \n");
      for (ordinal_type sp = 0; sp < x.extent(0); sp++) {
        fprintf(fs,"i %d % e \n", sp, x(sp) );
      }
      fprintf(fs, "fac \n");
      for (ordinal_type sp = 0; sp < fac.extent(0); sp++) {
        fprintf(fs,"i %d % e \n", sp, fac(sp) );
      }
      fprintf(fs, ":::: output\n");
      fprintf(fs, "f_0 \n");
      for (ordinal_type sp = 0; sp < f_0.extent(0); sp++) {
        fprintf(fs,"i %d % e \n", sp, f_0(sp) );
      }
      fprintf(fs, "f_h \n");
      for (ordinal_type sp = 0; sp < f_h.extent(0); sp++) {
        fprintf(fs,"i %d % e \n", sp, f_h(sp) );
      }
      fprintf(fs, "J\n");
      for (int i = 0; i < int(J.extent(0)); ++i){
        for (int j = 0; j < int(J.extent(1)); j++) {
          fprintf(fs,"%e ",J(i,j) );
        }
        fprintf(fs,"\n");
      }





    }
#endif

  }

  template<typename MemberType,
           typename ProblemType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(const MemberType& member,
                                                 const ProblemType& problem,
                                                 const real_type& fac_min,
                                                 const real_type& fac_max,
                                                 const RealType1DViewType& fac,
                                                 const RealType1DViewType& x,
                                                 const RealType2DViewType& J,
                                                 const RealType1DViewType& work)
  {
    real_type* wptr = work.data();
    const ordinal_type m = problem.getNumberOfEquations();
    RealType1DViewType f_0(wptr, m);
    wptr += f_0.span();
    RealType1DViewType f_h(wptr, m);



    team_invoke_detail(member, problem, fac_min, fac_max, fac, x, f_0, f_h, J);
  }
};

} // namespace Impl
} // namespace TChem

#endif
