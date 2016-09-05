/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.




  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/structured_grids/call_interfaces.hpp>
#include <tools/verifier.hpp>

using namespace gridtools;
using namespace gridtools::enumtype;
using namespace gridtools::expressions;

namespace call_interface_functors {

    // TODO this should go to a general helper class
    template < typename storage_t, typename F >
    static void fill(storage_t &&out, F f) {
        for (uint_t k = 0; k < out.meta_data().dim(2); ++k)
            for (uint_t i = 0; i < out.meta_data().dim(0); ++i)
                for (uint_t j = 0; j < out.meta_data().dim(1); ++j) {
                    out(i, j, k) = f(i, j, k);
                }
    }

    typedef interval< level< 0, -2 >, level< 1, 1 > > axis;
    typedef interval< level< 0, -1 >, level< 1, -1 > > x_interval;

    struct copy_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    struct call_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< copy_functor, x_interval >::with(eval, in(), out());
        }
    };

    struct call_nested_functor {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< call_functor, x_interval >::with(eval, in(), out());
        }
    };

    struct call_functor_with_offset {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< copy_functor, x_interval >::at< 1, 1, 0 >::with(eval, in(), out());
        }
    };

    struct call_nested_functor_with_offset {
        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = call< call_functor_with_offset, x_interval >::with(eval, in(), out());
        }
    };
}

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

class call_interface : public testing::Test {
  protected:
    const uint_t d1 = 13;
    const uint_t d2 = 9;
    const uint_t d3 = 7;
    const uint_t halo_size = 1;

    typedef gridtools::layout_map< 0, 1, 2 > layout_t;
    typedef gridtools::BACKEND::storage_info< 0, layout_t > meta_t;
    typedef gridtools::BACKEND::storage_type< uint_t, meta_t >::type storage_type;

    meta_t meta_;

    halo_descriptor di;
    halo_descriptor dj;
    gridtools::grid< call_interface_functors::axis > grid;

    verifier verifier_;
    array< array< uint_t, 2 >, 3 > verifier_halos;
    call_interface()
        : meta_(d1, d2, d3), di(halo_size, halo_size, halo_size, d1 - halo_size - 1, d1),
          dj(halo_size, halo_size, halo_size, d2 - halo_size - 1, d2), grid(di, dj),
#if FLOAT_PRECISION == 4
          verifier_(1e-6),
#else
          verifier_(1e-12),
#endif
          verifier_halos{{{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}} {
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;
    }
};

TEST_F(call_interface, call_to_copy_functor) {
    storage_type in(meta_, 0, "in");
    storage_type out(meta_, -5, "out");
    storage_type reference(meta_, -1, "reference");

    typedef arg< 0, storage_type > p_in;
    typedef arg< 1, storage_type > p_out;
    typedef boost::mpl::vector< p_in, p_out > accessor_list;

    aggregator_type< accessor_list > domain(boost::fusion::make_vector(&in, &out));

    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(
            execute< forward >(), gridtools::make_stage< call_interface_functors::call_functor >(p_in(), p_out())));

    call_interface_functors::fill(in, [](uint_t i, uint_t j, uint_t k) { return i + j * 10 + k * 100; });
    call_interface_functors::fill(reference, [](uint_t i, uint_t j, uint_t k) { return i + j * 10 + k * 100; });

    // run stencil
    comp->ready();
    comp->steady();
    comp->run();
#ifdef __CUDACC__
    out.d2h_update();
#endif

    ASSERT_TRUE(verifier_.verify(grid, reference, out, verifier_halos));
}

TEST_F(call_interface, call_to_copy_functor_with_offset) {
    storage_type in(meta_, 0, "in");
    storage_type out(meta_, -5, "out");
    storage_type reference(meta_, -1, "reference");

    typedef arg< 0, storage_type > p_in;
    typedef arg< 1, storage_type > p_out;
    typedef boost::mpl::vector< p_in, p_out > accessor_list;

    aggregator_type< accessor_list > domain(boost::fusion::make_vector(&in, &out));

    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_functor_with_offset >(p_in(), p_out())));

    call_interface_functors::fill(in, [](uint_t i, uint_t j, uint_t k) { return i + j * 10 + k * 100; });
    call_interface_functors::fill(
        reference, [](uint_t i, uint_t j, uint_t k) { return (i + 1) + (j + 1) * 10 + k * 100; });

    // run stencil
    comp->ready();
    comp->steady();
    comp->run();
#ifdef __CUDACC__
    out.d2h_update();
#endif

    ASSERT_TRUE(verifier_.verify(grid, reference, out, verifier_halos));
}

TEST_F(call_interface, nested_call_to_copy_functor) {
    storage_type in(meta_, 0, "in");
    storage_type out(meta_, -5, "out");
    storage_type reference(meta_, -1, "reference");

    typedef arg< 0, storage_type > p_in;
    typedef arg< 1, storage_type > p_out;
    typedef boost::mpl::vector< p_in, p_out > accessor_list;

    aggregator_type< accessor_list > domain(boost::fusion::make_vector(&in, &out));

    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_nested_functor >(p_in(), p_out())));

    call_interface_functors::fill(in, [](uint_t i, uint_t j, uint_t k) { return i + j * 10 + k * 100; });
    call_interface_functors::fill(reference, [](uint_t i, uint_t j, uint_t k) { return i + j * 10 + k * 100; });

    // run stencil
    comp->ready();
    comp->steady();
    comp->run();
#ifdef __CUDACC__
    out.d2h_update();
#endif

    ASSERT_TRUE(verifier_.verify(grid, reference, out, verifier_halos));
}

TEST_F(call_interface, nested_call_to_copy_functor_with_offset) {
    storage_type in(meta_, 0, "in");
    storage_type out(meta_, -5, "out");
    storage_type reference(meta_, -1, "reference");

    typedef arg< 0, storage_type > p_in;
    typedef arg< 1, storage_type > p_out;
    typedef boost::mpl::vector< p_in, p_out > accessor_list;

    aggregator_type< accessor_list > domain(boost::fusion::make_vector(&in, &out));

    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< call_interface_functors::call_nested_functor_with_offset >(p_in(), p_out())));

    call_interface_functors::fill(in, [](uint_t i, uint_t j, uint_t k) { return i + j * 10 + k * 100; });
    call_interface_functors::fill(
        reference, [](uint_t i, uint_t j, uint_t k) { return (i + 1) + (j + 1) * 10 + k * 100; });

    // run stencil
    comp->ready();
    comp->steady();
    comp->run();
#ifdef __CUDACC__
    out.d2h_update();
#endif

    ASSERT_TRUE(verifier_.verify(grid, reference, out, verifier_halos));
}
