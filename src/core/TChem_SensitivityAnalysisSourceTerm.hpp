#ifndef __TCHEM_IMPL_SENSITIVITY_ANALYSIS_SOURCE_THERM_HPP__
#define __TCHEM_IMPL_SENSITIVITY_ANALYSIS_SOURCE_THERM_HPP__

#include "TChem_Impl_SensitivityAnalysisSourceTerm.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace TChem {

struct SensitivityAnalysisSourceTerm
{

  template<typename KineticModelConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd)
  {
    return Impl::SensitivityAnalysisSourceTerm::getWorkSpaceSize(kmcd) + kmcd.nSpec + 1;
  }

  static void runDeviceBatch( /// input
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_2d_view& state,
    const real_type_1d_view& alpha,
    /// output
    const real_type_2d_view& SourceTerm,
    const real_type_2d_view& facL,
    const real_type_2d_view& facF,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd);
  //
  // static void runHostBatch( /// input
  //   typename UseThisTeamPolicy<host_exec_space>::type& policy,
  //   const real_type_2d_view_host& state,
  //   const real_type_1d_view_host& alpha,
  //   /// output
  //   const real_type_2d_view_host& SourceTerm,
  //   const real_type_2d_view_host& facL,
  //   const real_type_2d_view_host& facF,
  //   /// const data from kinetic model
  //   const KineticModelConstDataHost& kmcd);

};

} // namespace TChem

#endif
