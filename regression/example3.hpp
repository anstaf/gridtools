


#include <gridtools/common/defs.hpp>
#include <gridtools/common/gt_math.hpp>

#include <boost/cstdfloat.hpp> 

#include <array>
#include <cstdint>
#include <vector>


using boost::float32_t; 
using boost::float64_t; 

using py_size_t = std::intptr_t;

struct BufferInfo {
    py_size_t ndim;
    std::vector<py_size_t> shape;
    std::vector<py_size_t> strides;
    void* ptr;
};


namespace gt = ::gridtools;

namespace main_loop____gtx86_0466c98f64_pyext {

void run(const std::array<gt::uint_t, 3>& domain,
         const BufferInfo& bi_h_var, const std::array<gt::uint_t, 3>& h_var_origin, 
         const BufferInfo& bi_rh_adj, const std::array<gt::uint_t, 3>& rh_adj_origin, 
         const BufferInfo& bi_rh_rain, const std::array<gt::uint_t, 3>& rh_rain_origin, 
         const BufferInfo& bi_graupel, const std::array<gt::uint_t, 3>& graupel_origin, 
         const BufferInfo& bi_ice, const std::array<gt::uint_t, 3>& ice_origin, 
         const BufferInfo& bi_rain, const std::array<gt::uint_t, 3>& rain_origin, 
         const BufferInfo& bi_snow, const std::array<gt::uint_t, 3>& snow_origin, 
         const BufferInfo& bi_qaz, const std::array<gt::uint_t, 3>& qaz_origin, 
         const BufferInfo& bi_qgz, const std::array<gt::uint_t, 3>& qgz_origin, 
         const BufferInfo& bi_qiz, const std::array<gt::uint_t, 3>& qiz_origin, 
         const BufferInfo& bi_qlz, const std::array<gt::uint_t, 3>& qlz_origin, 
         const BufferInfo& bi_qrz, const std::array<gt::uint_t, 3>& qrz_origin, 
         const BufferInfo& bi_qsz, const std::array<gt::uint_t, 3>& qsz_origin, 
         const BufferInfo& bi_qvz, const std::array<gt::uint_t, 3>& qvz_origin, 
         const BufferInfo& bi_tz, const std::array<gt::uint_t, 3>& tz_origin, 
         const BufferInfo& bi_w, const std::array<gt::uint_t, 3>& w_origin, 
         const BufferInfo& bi_t0, const std::array<gt::uint_t, 3>& t0_origin, 
         const BufferInfo& bi_den0, const std::array<gt::uint_t, 3>& den0_origin, 
         const BufferInfo& bi_dz0, const std::array<gt::uint_t, 3>& dz0_origin, 
         const BufferInfo& bi_dp1, const std::array<gt::uint_t, 3>& dp1_origin, 
         const BufferInfo& bi_p1, const std::array<gt::uint_t, 3>& p1_origin, 
         const BufferInfo& bi_m1, const std::array<gt::uint_t, 3>& m1_origin, 
         const BufferInfo& bi_ccn, const std::array<gt::uint_t, 3>& ccn_origin, 
         const BufferInfo& bi_c_praut, const std::array<gt::uint_t, 3>& c_praut_origin, 
         const BufferInfo& bi_m2_rain, const std::array<gt::uint_t, 3>& m2_rain_origin, 
         const BufferInfo& bi_m2_sol, const std::array<gt::uint_t, 3>& m2_sol_origin, 
         int32_t do_sedi_w, 
         int32_t p_nonhydro, 
         int32_t use_ccn, 
         float64_t c_air, 
         float64_t c_vap, 
         float64_t d0_vap, 
         float64_t lv00, 
         float64_t fac_rc, 
         float64_t csacr, 
         float64_t cracs, 
         float64_t cgacr, 
         float64_t cgacs, 
         float64_t acco_00, 
         float64_t acco_01, 
         float64_t acco_02, 
         float64_t acco_03, 
         float64_t acco_10, 
         float64_t acco_11, 
         float64_t acco_12, 
         float64_t acco_13, 
         float64_t acco_20, 
         float64_t acco_21, 
         float64_t acco_22, 
         float64_t acco_23, 
         float64_t csacw, 
         float64_t csaci, 
         float64_t cgacw, 
         float64_t cgaci, 
         float64_t cracw, 
         float64_t cssub_0, 
         float64_t cssub_1, 
         float64_t cssub_2, 
         float64_t cssub_3, 
         float64_t cssub_4, 
         float64_t crevp_0, 
         float64_t crevp_1, 
         float64_t crevp_2, 
         float64_t crevp_3, 
         float64_t crevp_4, 
         float64_t cgfr_0, 
         float64_t cgfr_1, 
         float64_t csmlt_0, 
         float64_t csmlt_1, 
         float64_t csmlt_2, 
         float64_t csmlt_3, 
         float64_t csmlt_4, 
         float64_t cgmlt_0, 
         float64_t cgmlt_1, 
         float64_t cgmlt_2, 
         float64_t cgmlt_3, 
         float64_t cgmlt_4, 
         float64_t ces0, 
         float64_t log_10, 
         float64_t tice0, 
         float64_t t_wfr, 
         float64_t so3, 
         float64_t dt_rain, 
         float64_t zs, 
         float64_t dts, 
         float64_t rdts, 
         float64_t fac_i2s, 
         float64_t fac_g2v, 
         float64_t fac_v2g, 
         float64_t fac_imlt, 
         float64_t fac_l2v);

}  // namespace main_loop____gtx86_0466c98f64_pyext