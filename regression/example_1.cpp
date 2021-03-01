

#include "example_1.hpp"

#include <gridtools/stencil_composition/stencil_composition.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <type_traits>

static constexpr int MAX_DIM = 3;

namespace ke_from_bwind____gtx86_c3fe030bc4_pyext {

namespace {

// Backend
using backend_t = gt::backend::x86;

// Storage definitions
template <typename T, int Id, bool... Dims>
struct storage_traits {
  using info_t =
      gt::storage_traits<backend_t>::special_storage_info_t<Id, gt::selector<Dims...>,
                                                            gt::halo<0, 0, 0 /* not used */>>;
  using store_t = gt::storage_traits<backend_t>::data_store_t<T, info_t>;
};

template <typename T, int Id>
struct storage_traits<T, Id, 1, 1, 1> {
  using info_t =
      gt::storage_traits<backend_t>::storage_info_t<Id, 3, gt::halo<0, 0, 0 /* not used */>>;
  using store_t = gt::storage_traits<backend_t>::data_store_t<T, info_t>;
};

template <typename T>
constexpr int count(std::integer_sequence<T>) {
  return 0;
}

template <typename T, T Head, T... Tail>
constexpr int count(std::integer_sequence<T, Head, Tail...>) {
  return Head + count(std::integer_sequence<T, Tail...>());
}

template <class T, T... Ints>
constexpr T get(std::integer_sequence<T, Ints...>, std::size_t i) {
  constexpr T arr[] = {Ints...};
  return arr[i];
}

template <typename T, int Id, int NDIM, bool... Dims>
typename storage_traits<T, Id, Dims...>::store_t
make_data_store(const BufferInfo& bi, const std::array<gt::uint_t, MAX_DIM>& compute_domain_shape,
                const std::array<gt::uint_t, NDIM>& origin, gt::selector<Dims...> mask) {
  // ptr, dims and strides are "outer domain" (i.e., compute domain + halo
  // region). The halo region is only defined through `make_grid` (and
  // currently, in the storage info)
  static_assert(NDIM == count(mask), "Mask dimensions do not match origin");

  gt::array<gt::uint_t, MAX_DIM> dims{};
  gt::array<gt::uint_t, MAX_DIM> strides{};
  T* ptr = static_cast<T*>(bi.ptr);

  int curr_stride = 0;
  for(int i = 0, j = 0; i < MAX_DIM; ++i) {
    if(get(mask, i) != 0) {
      curr_stride = strides[i] = bi.strides[j] / sizeof(T);
      ptr += strides[i] * origin[j];
      dims[i] = compute_domain_shape[i] + 2 * origin[j];
      ++j;
    } else {
      strides[i] = curr_stride;
      dims[i] = 0;
    }
  }

  using storage_info_t = typename storage_traits<T, Id, Dims...>::info_t;
  using data_store_t = typename storage_traits<T, Id, Dims...>::store_t;

  return data_store_t{storage_info_t{dims, strides}, ptr, gt::ownership::external_cpu};
}

// Axis
static constexpr gt::uint_t level_offset_limit = 2;

using axis_t = gt::axis<1, /* NIntervals */
                        gt::axis_config::offset_limit<level_offset_limit>>;

// These halo sizes are used to determine the sizes of the temporaries
static constexpr gt::uint_t halo_size_i = 0;
static constexpr gt::uint_t halo_size_j = 0;
static constexpr gt::uint_t halo_size_k = 0;

// Placeholder definitions

using p_ke = gt::arg<0, typename storage_traits<float64_t, 0, true, true, true>::store_t>;
using p_ub = gt::arg<1, typename storage_traits<float64_t, 1, true, true, true>::store_t>;
using p_vb = gt::arg<2, typename storage_traits<float64_t, 2, true, true, true>::store_t>;

// Constants

// Functors

struct stage__6_func {
  using ke = gt::inout_accessor<0, gt::extent<0, 0, 0, 0, 0, 0>>;
  using ub = gt::in_accessor<1, gt::extent<0, 0, 0, 0, 0, 0>>;
  using vb = gt::in_accessor<2, gt::extent<0, 0, 0, 0, 0, 0>>;

  using param_list = gt::make_param_list<ke, ub, vb>;

  template <typename Evaluation>
  GT_FUNCTION static void
  apply(Evaluation eval,
        gt::interval<gt::level<0, 1, level_offset_limit>, gt::level<1, -1, level_offset_limit>>) {
    eval(ke()) = float64_t{0.5} * (eval(ke()) + (eval(ub()) * eval(vb())));
  }
};

// Grids and halos
gt::halo_descriptor make_halo_descriptor(gt::uint_t compute_domain_shape) {
  return {0, 0, 0, compute_domain_shape - 1, compute_domain_shape};
}

auto make_grid(const std::array<gt::uint_t, MAX_DIM>& compute_domain_shape) {
  return gt::make_grid(make_halo_descriptor(compute_domain_shape[0]),
                       make_halo_descriptor(compute_domain_shape[1]),
                       axis_t(compute_domain_shape[2]));
}

} // namespace

// Run actual computation
void run(const std::array<gt::uint_t, MAX_DIM>& domain, const BufferInfo& bi_ke,
         const std::array<gt::uint_t, 3>& ke_origin, const BufferInfo& bi_ub,
         const std::array<gt::uint_t, 3>& ub_origin, const BufferInfo& bi_vb,
         const std::array<gt::uint_t, 3>& vb_origin) {
  // Initialize data stores from input buffers
  auto ds_ke =
      make_data_store<float64_t, 0, 3>(bi_ke, domain, ke_origin, gt::selector<true, true, true>{});
  auto ds_ub =
      make_data_store<float64_t, 1, 3>(bi_ub, domain, ub_origin, gt::selector<true, true, true>{});
  auto ds_vb =
      make_data_store<float64_t, 2, 3>(bi_vb, domain, vb_origin, gt::selector<true, true, true>{});

  // Run computation and wait for the synchronization of the output stores
  auto gt_computation = gt::make_computation<backend_t>(
      make_grid(domain),
      gt::make_multistage(gt::execute::parallel(),
                          gt::make_stage_with_extent<stage__6_func, gt::extent<0, 0, 0, 0>>(
                              p_ke(), p_ub(), p_vb())));

  gt_computation.run(p_ke() = ds_ke, p_ub() = ds_ub, p_vb() = ds_vb);
  // computation_.sync_bound_data_stores();
}

} // namespace ke_from_bwind____gtx86_c3fe030bc4_pyext