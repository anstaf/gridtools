


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

namespace transportdelp____gtx86_d7fde2f45f_pyext {

void run(const std::array<gt::uint_t, 3>& domain,
         const BufferInfo& bi_delp, const std::array<gt::uint_t, 3>& delp_origin, 
         const BufferInfo& bi_pt, const std::array<gt::uint_t, 3>& pt_origin, 
         const BufferInfo& bi_utc, const std::array<gt::uint_t, 3>& utc_origin, 
         const BufferInfo& bi_vtc, const std::array<gt::uint_t, 3>& vtc_origin, 
         const BufferInfo& bi_w, const std::array<gt::uint_t, 3>& w_origin, 
         const BufferInfo& bi_rarea, const std::array<gt::uint_t, 3>& rarea_origin, 
         const BufferInfo& bi_delpc, const std::array<gt::uint_t, 3>& delpc_origin, 
         const BufferInfo& bi_ptc, const std::array<gt::uint_t, 3>& ptc_origin, 
         const BufferInfo& bi_wc, const std::array<gt::uint_t, 3>& wc_origin);

}  // namespace transportdelp____gtx86_d7fde2f45f_pyext