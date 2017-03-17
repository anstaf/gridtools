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
/*
 * test_cache_metafunctions.cpp
 *
 *  Created on: Jul 17, 2015
 *      Author: cosuna
 */

#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include "common/defs.hpp"
#include "stencil-composition/empty_extent.hpp"
#include "stencil-composition/backend.hpp"
#include "stencil-composition/caches/cache_metafunctions.hpp"
#include "stencil-composition/interval.hpp"
#include "stencil-composition/stencil-composition.hpp"
#include "common/generic_metafunctions/fusion_map_to_mpl_map.hpp"
#include "stencil-composition/caches/extract_extent_caches.hpp"

using namespace gridtools;
using namespace enumtype;

// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 1, -1 > > x_interval;
struct functor1 {
    typedef accessor< 0, enumtype::in, extent< 0, 0, 0, 0 >, 6 > in;
    typedef accessor< 1, enumtype::inout, extent< 0, 0, 0, 0 >, 5 > buff;
    typedef boost::mpl::vector< in, buff > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {}
};

typedef layout_map< 0, 1 > layout_ij_t;
typedef backend< Host, GRIDBACKEND, Naive >::storage_type< float_type,
    backend< Host, GRIDBACKEND, Naive >::storage_info< 0, layout_ij_t > >::type storage_type;

typedef arg< 0, storage_type > p_in;
typedef arg< 2, storage_type > p_out;
typedef arg< 1, storage_type > p_buff;
typedef arg< 3, storage_type > p_notin;

typedef decltype(gridtools::make_stage< functor1 >(p_in(), p_buff())) esf1_t;
typedef decltype(gridtools::make_stage< functor1 >(p_buff(), p_out())) esf2_t;

typedef boost::mpl::vector2< esf1_t, esf2_t > esf_sequence_t;

typedef detail::cache_impl< IJ, p_in, fill, boost::mpl::void_ > cache1_t;
typedef detail::cache_impl< IJ, p_buff, fill, boost::mpl::void_ > cache2_t;
typedef detail::cache_impl< K, p_out, local, x_interval > cache3_t;
typedef detail::cache_impl< K, p_notin, local, x_interval > cache4_t;
typedef boost::mpl::vector4< cache1_t, cache2_t, cache3_t, cache4_t > caches_t;

TEST(cache_metafunctions, cache_used_by_esfs) {
    typedef caches_used_by_esfs< esf_sequence_t, caches_t >::type caches_used_t;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal< caches_used_t, boost::mpl::vector3< cache1_t, cache2_t, cache3_t > >::value), "WRONG");
    ASSERT_TRUE(true);
}

TEST(cache_metafunctions, extract_extents_for_caches) {
    typedef boost::mpl::vector3< p_in, p_buff, p_out > esf_args_t;
    typedef local_domain< boost::mpl::void_, boost::mpl::void_, esf_args_t, false > local_domain_t;

    typedef boost::mpl::vector2< extent< -1, 2, -2, 1 >, extent< -2, 1, -3, 2 > > extents_t;
    typedef gridtools::interval< gridtools::level< 0, -2 >, gridtools::level< 1, 1 > > axis;

    typedef typename boost::mpl::fold< extents_t,
        extent< 0, 0, 0, 0 >,
        enclosing_extent< boost::mpl::_1, boost::mpl::_2 > >::type max_extent_t;

    typedef iterate_domain_arguments< backend_ids< Cuda, GRIDBACKEND, Block >,
        local_domain_t,
        esf_sequence_t,
        extents_t,
        max_extent_t,
        caches_t,
        block_size< 32, 4, 1 >,
        block_size< 32, 4, 1 >,
        gridtools::grid< axis >,
        boost::mpl::false_,
        notype > iterate_domain_arguments_t;

    typedef extract_extents_for_caches< iterate_domain_arguments_t >::type extents_map_t;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal< extents_map_t,
                                boost::mpl::map3< boost::mpl::pair< cache1_t, extent< -1, 2, -2, 1 > >,
                                                    boost::mpl::pair< cache2_t, extent< -2, 2, -3, 2 > >,
                                                    boost::mpl::pair< cache3_t, extent< -2, 1, -3, 2 > > > >::value),
        "ERROR");
}

TEST(cache_metafunctions, get_cache_storage_tuple) {
    typedef metadata_set< boost::mpl::vector1< pointer< storage_type::storage_info_type > > > metadata_vector_t;
    typedef boost::mpl::vector3< pointer< storage_type >, pointer< storage_type >, pointer< storage_type > > storages_t;
    typedef boost::fusion::result_of::as_vector< storages_t >::type storages_tuple_t;
    typedef boost::mpl::vector3< p_in, p_buff, p_out > esf_args_t;
    typedef local_domain< storages_tuple_t, metadata_vector_t, esf_args_t, false > local_domain_t;

    typedef boost::mpl::vector2< extent< -1, 2, -2, 1 >, extent< -2, 1, -3, 2 > > extents_t;
    typedef typename boost::mpl::fold< extents_t,
        extent< 0, 0, 0, 0 >,
        enclosing_extent< boost::mpl::_1, boost::mpl::_2 > >::type max_extent_t;

    typedef gridtools::interval< gridtools::level< 0, -2 >, gridtools::level< 1, 1 > > axis;

    typedef iterate_domain_arguments< backend_ids< Cuda, GRIDBACKEND, Block >,
        local_domain_t,
        esf_sequence_t,
        extents_t,
        max_extent_t,
        caches_t,
        block_size< 32, 4, 1 >,
        block_size< 32, 4, 1 >,
        gridtools::grid< axis >,
        boost::mpl::false_,
        notype > iterate_domain_arguments_t;

    typedef extract_extents_for_caches< iterate_domain_arguments_t >::type extents_map_t;

    typedef get_cache_storage_tuple< IJ, caches_t, extents_map_t, block_size< 32, 4, 1 >, local_domain_t >::type
        cache_storage_tuple_t;

    // fusion::result_of::at_key<cache_storage_tuple_t, p_in::index_type> does not compile,
    // therefore we convert into an mpl map and do all the metaprogramming operations on that map
    typedef fusion_map_to_mpl_map< cache_storage_tuple_t >::type cache_storage_mpl_map_t;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<
            cache_storage_tuple_t,
            boost::fusion::map<
                boost::fusion::pair< p_in::index_type,
                    cache_storage< block_size< 32, 4, 1 >, extent< -1, 2, -2, 1 >, 1, pointer< storage_type > > >,
                boost::fusion::pair< p_buff::index_type,
                    cache_storage< block_size< 32, 4, 1 >, extent< -2, 2, -3, 2 >, 1, pointer< storage_type > > > > >::
                value),
        "ERROR");
}