#pragma once

#include <boost/mpl/has_key.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/find_if.hpp>
#include "basic_token_execution.h"

namespace gridtools {
    namespace _impl_naive {

        template <typename t_functor_list,
                  typename t_loop_intervals,
                  typename t_functors_map,
                  typename t_range_sizes,
                  typename t_domain_list,
                  typename t_coords>
        struct run_functor {
            t_coords const &coords;
            t_domain_list &domain_list;

            explicit run_functor(t_domain_list & domain_list, t_coords const& coords)
                : coords(coords)
                , domain_list(domain_list)
            {}

            template <typename t_index>
            void operator()(t_index const&) const {
                typedef typename boost::mpl::at<t_range_sizes, t_index>::type range_type;

                typedef typename boost::mpl::at<t_functor_list, t_index>::type functor_type;
                typedef typename boost::fusion::result_of::value_at<t_domain_list, t_index>::type local_domain_type;

                typedef typename boost::mpl::at<t_functors_map, t_index>::type interval_map;

                local_domain_type local_domain = boost::fusion::at<t_index>(domain_list);

#ifndef NDEBUG
                std::cout << "Functor " << functor_type() << std::endl;
                std::cout << "I loop " << coords.i_low_bound() + range_type::iminus::value << " -> "
                          << coords.i_high_bound() + range_type::iplus::value << std::endl;
                std::cout << "J loop " << coords.j_low_bound() + range_type::jminus::value << " -> "
                          << coords.j_high_bound() + range_type::jplus::value << std::endl;
#endif


                typedef typename index_to_level<typename boost::mpl::deref<typename boost::mpl::find_if<t_loop_intervals, boost::mpl::has_key<interval_map, boost::mpl::_1> >::type>::type::first>::type first_hit;
#ifndef NDEBUG
                std::cout << " ******************** " << first_hit() << std::endl;
                std::cout << " ******************** " << coords.template value_at<first_hit>() << std::endl;
#endif

                for (int i = coords.i_low_bound() + range_type::iminus::value;
                     i < coords.i_high_bound() + range_type::iplus::value;
                     ++i)
                    for (int j = coords.j_low_bound() + range_type::jminus::value;
                         j < coords.j_high_bound() + range_type::jplus::value;
                         ++j) {
                        local_domain.move_to(i,j, coords.template value_at<first_hit>());
                        for_each<t_loop_intervals>(_impl::run_f_on_interval<functor_type, interval_map,local_domain_type,t_coords>(local_domain,coords));
                    }
            }

        };
    }

    struct backend_naive {
        static const int BI = 0;
        static const int BJ = 0;
        static const int BK = 0;

        template <typename t_functor_list, // List of functors to execute (in order)
                  typename range_sizes, // computed range sizes to know where to compute functot at<i>
                  typename t_loop_intervals, // List of intervals on which functors are defined
                  typename t_functors_map,  // Map between interval and actual arguments to pass to Do methods
                  typename t_domain, // Domain class (not really useful maybe)
                  typename t_coords, // Coordinate class with domain sizes and splitter coordinates
                  typename t_local_domain_list> // List of local domain to be pbassed to functor at<i>
        static void run(t_domain const& domain, t_coords const& coords, t_local_domain_list &local_domain_list) {

            typedef boost::mpl::range_c<int, 0, boost::mpl::size<t_functor_list>::type::value> iter_range;

            for_each<iter_range>(_impl_naive::run_functor
                                             <
                                             t_functor_list,
                                             t_loop_intervals,
                                             t_functors_map,
                                             range_sizes,
                                             t_local_domain_list,
                                             t_coords
                                             >
                                             (local_domain_list,coords));
            std::cout << std::endl;
        }
    };

} // namespace gridtools
