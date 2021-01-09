#include "TChem_SensitivityAnalysisSourceTerm.hpp"

namespace TChem {

  template<typename PolicyType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstType>

  void
  SensitivityAnalysisSourceTerm_TemplateRun( /// input
    const std::string& profile_name,
    const RealType1DViewType& dummy_1d,
    /// team size setting
    const PolicyType& policy,
    const RealType2DViewType& state,
    const RealType2DViewType& facL,
    const RealType2DViewType& facF,
    const RealType2DViewType& SourceTerm,
    const KineticModelConstType& kmcd
  )
  {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;

    const ordinal_type level = 1;
    const ordinal_type per_team_extent = SensitivityAnalysisSourceTerm::getWorkSpaceSize(kmcd);
    const ordinal_type m = kmcd.nSpec + 1;
    Kokkos::parallel_for(
      profile_name,
      policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const ordinal_type i = member.league_rank();
        const RealType1DViewType facL_at_i =
          Kokkos::subview(facL, i, Kokkos::ALL());
        //
        const RealType1DViewType facF_at_i =
          Kokkos::subview(facF, i, Kokkos::ALL());
        const RealType1DViewType facM_at_i =
          Kokkos::subview(facF, i, Kokkos::ALL());
        const RealType1DViewType state_at_i =
          Kokkos::subview(state, i, Kokkos::ALL());

        const RealType1DViewType SourceTerm_at_i =
          Kokkos::subview(SourceTerm, i, Kokkos::ALL());

        Scratch<RealType1DViewType> work(member.team_scratch(level),
                                        per_team_extent);

        //
        Impl::StateVector<RealType1DViewType> sv_at_i(kmcd.nSpec, state_at_i);

        TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                          "Error: input state vector is not valid");

        //
        auto wptr = work.data();
        const RealType1DViewType vals(wptr, m);
        wptr += m;
        const RealType1DViewType ww(wptr,
                                    work.extent(0) - (wptr - work.data()));
        //
        const auto temperature = sv_at_i.Temperature();
        const auto pressure = sv_at_i.Pressure();
        const auto Ys = sv_at_i.MassFractions();
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                             [&](const ordinal_type& i) {
                               vals(i) = i == 0 ? temperature : Ys(i - 1);
                             });

        Impl::SensitivityAnalysisSourceTerm ::team_invoke(member, pressure, vals,
           SourceTerm_at_i, facL_at_i, facF_at_i, work, kmcd);

      });
    Kokkos::Profiling::popRegion();
  }



void
SensitivityAnalysisSourceTerm::runDeviceBatch( /// input
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_2d_view& state,
  /// output
  const real_type_2d_view& SourceTerm,
  const real_type_2d_view& facL,
  const real_type_2d_view& facF,
  /// const data from kinetic model
  const KineticModelConstDataDevice& kmcd)
{

  SensitivityAnalysisSourceTerm_TemplateRun( /// input
    "TChem::TChem_SensitivityAnalysisSourceTerm::runDeviceBatch",
    real_type_1d_view(),
    /// team size setting
    policy,
    state,
    SourceTerm,
    facL,
    facF,
    kmcd);

}

void
SensitivityAnalysisSourceTerm::runHostBatch( /// input
  typename UseThisTeamPolicy<host_exec_space>::type& policy,
  const real_type_2d_view_host& state,
  /// output
  const real_type_2d_view_host& SourceTerm,
  const real_type_2d_view_host& facL,
  const real_type_2d_view_host& facF,
  /// const data from kinetic model
  const KineticModelConstDataHost& kmcd)
{

  SensitivityAnalysisSourceTerm_TemplateRun( /// input
    "TChem::SourceTermToyProblem::runHostBatch",
    real_type_1d_view_host(),
    /// team size setting
    policy,
    state,
    SourceTerm,
    facL,
    facF,
    kmcd);

}


} // namespace TChem
