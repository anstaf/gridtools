#pragma once
#include "../common/basic_utils.h"
#include "../common/gpu_clone.h"

namespace gridtools {

    namespace _impl
    {
        template <int I, typename OtherLayout, int X>
        struct get_stride_
        {
            GT_FUNCTION
            static int get(const int* s) {
                return s[OtherLayout::template at_<I>::value];
            }
        };

        template <int I, typename OtherLayout>
        struct get_stride_<I, OtherLayout, 2>
        {
            GT_FUNCTION
            static int get(const int* ) {
#ifndef __CUDACC__
#ifndef NDEBUG
                //                std::cout << "U" ;//<< std::endl;
#endif
#endif
                return 1;
            }
        };

        template <int I, typename OtherLayout>
        struct get_stride
          : get_stride_<I, OtherLayout, OtherLayout::template at_<I>::value>
        {};
    }

    template < typename Derived,
               typename ValueType,
               typename Layout,
               bool IsTemporary = false
               >
    struct base_storage : public clonable_to_gpu<Derived> {
        typedef Layout layout;
        typedef ValueType value_type;
        typedef value_type* iterator_type;
        typedef value_type const* const_iterator_type;

        int m_dims[3];
        int strides[3];
        int m_size;
        bool is_set;
        //std::string name;

        explicit base_storage(int m_dim1, int m_dim2, int m_dim3,
                         value_type init = value_type(),
                         std::string const& s = std::string("default name") ) {
            m_dims[0] = m_dim1;
            m_dims[1] = m_dim2;
            m_dims[2] = m_dim3;
            strides[0] = layout::template find<2>(m_dims)*layout::template find<1>(m_dims);
            strides[1] = layout::template find<2>(m_dims);
            strides[2] = 1;
            m_size = m_dims[0] * m_dims[1] * m_dims[2];
            //            std::cout << "Size " << m_size << std::endl;
            is_set=true;
            //name = s;
        }

        __device__
        base_storage(base_storage const& other)
            : m_size(other.m_size)
            , is_set(is_set)
        {
            m_dims[0] = other.m_dims[0];
            m_dims[1] = other.m_dims[1];
            m_dims[2] = other.m_dims[2];

            strides[0] = other.strides[0];
            strides[1] = other.strides[1];
            strides[2] = other.strides[2];
        }

        explicit base_storage() {
            is_set=false;
        }

        virtual void h2d_update() {}
        virtual void d2h_update() {}

        void info() const {
            std::cout << m_dims[0] << "x"
                      << m_dims[1] << "x"
                      << m_dims[2] << ", "
                //<< name 
                      << std::endl;
        }

        template <int I>
        GT_FUNCTION
        int stride_along() const {
            return _impl::get_stride<I, layout>::get(strides); /*layout::template at_<I>::value];*/
        }

    protected:
        template <typename derived_t>
        void print(derived_t* that) const {
            //std::cout << "Printing " << name << std::endl;
            std::cout << "(" << m_dims[0] << "x"
                      << m_dims[1] << "x"
                      << m_dims[2] << ")"
                      << std::endl;
            std::cout << "| j" << std::endl;
            std::cout << "| j" << std::endl;
            std::cout << "v j" << std::endl;
            std::cout << "---> k" << std::endl;

            int MI=12;
            int MJ=12;
            int MK=12;

            for (int i = 0; i < std::min(m_dims[0],MI); ++i) {
                for (int j = 0; j < std::min(m_dims[1],MJ); ++j) {
                    for (int k = 0; k < std::min(m_dims[2],MK); ++k) {
                        std::cout << "["/*("
                                          << i << ","
                                          << j << ","
                                          << k << ")"*/
                                  << that->operator()(i,j,k) << "] ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        GT_FUNCTION
        int _index(int i, int j, int k) const {
            int index;
            if (IsTemporary) {
                index =
                    layout::template find<2>(m_dims) * layout::template find<1>(m_dims)
                    * (modulus(layout::template find<0>(i,j,k),layout::template find<0>(m_dims))) +
                    layout::template find<2>(m_dims) * modulus(layout::template find<1>(i,j,k),layout::template find<1>(m_dims)) +
                    modulus(layout::template find<2>(i,j,k),layout::template find<2>(m_dims));
            } else {
                index =
                    layout::template find<2>(m_dims) * layout::template find<1>(m_dims)
                    * layout::template find<0>(i,j,k) +
                    layout::template find<2>(m_dims) * layout::template find<1>(i,j,k) +
                    layout::template find<2>(i,j,k);
            }
            assert(index >= 0);
            assert(index <m_size);
            return index;
        }
    };
    
    template <typename T>
    struct is_temporary_storage {
        typedef boost::false_type type;
    };



} //namespace gridtools