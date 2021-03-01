


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

namespace ke_from_bwind____gtx86_c3fe030bc4_pyext {

void run(const std::array<gt::uint_t, 3>& domain,
         const BufferInfo& bi_ke, const std::array<gt::uint_t, 3>& ke_origin, 
         const BufferInfo& bi_ub, const std::array<gt::uint_t, 3>& ub_origin, 
         const BufferInfo& bi_vb, const std::array<gt::uint_t, 3>& vb_origin);

}  // namespace ke_from_bwind____gtx86_c3fe030bc4_pyext