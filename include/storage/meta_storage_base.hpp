#pragma once
#include "base_storage_impl.hpp"
#include "../common/array.hpp"
#include "common/explode_array.hpp"

/**
   @file
   @brief basic file containing the storage meta information container
 */

/**
    @class
    @brief class containing the meta storage information
*/
namespace gridtools {

    /**
     * @brief Type to indicate that the type is not decided yet
     */
    template <typename RegularMetaStorageType>
    struct no_meta_storage_type_yet {
        typedef RegularMetaStorageType type;
        typedef typename type::index_type index_type;
        typedef typename  type::layout layout;
        static const ushort_t space_dimensions=type::space_dimensions;
        static const bool is_temporary = type::is_temporary;
    };

    template<typename T>
    struct is_meta_storage : boost::mpl::false_{};


    /**fwd declaration*/
    template < ushort_t Index
               , typename Layout
               , bool IsTemporary
               , typename ... Tiles
               >
    struct meta_storage_base;

    /**@brief class containing the storage meta information for non temporary storage

       \tparam Index an index used to differentiate the types also when there's only runtime
       differences (e.g. only the storage dimensions differ)
       \tparam Layout the map of the layout in memory
       \tparam IsTemporary boolean flag set to true when the storage is a temporary one
     */
    template < ushort_t Index
               , typename Layout
               , bool IsTemporary
               >
    struct meta_storage_base<Index, Layout, IsTemporary>
    {
        typedef meta_storage_base<Index, Layout , IsTemporary> type;
        typedef Layout layout;
        typedef static_ushort<Index> index_type;

        typedef meta_storage_base<Index, Layout , IsTemporary> basic_type;

        static const bool is_temporary = IsTemporary;
        static const ushort_t n_width = 1;
        static const ushort_t space_dimensions = layout::length;

    protected:

         array<int_t, space_dimensions> m_dims;
         array<int_t, space_dimensions> m_strides;

    public:

        template <ushort_t I, typename L, bool B, typename ... Ts>
        friend std::ostream& operator<<(std::ostream &, meta_storage_base<I,L,B,Ts...> const & );

#ifdef CXX11_ENABLED
        /**
           SFINAE for the case in which all the components of a parameter pack are of integral type
         */
        template <typename ... IntTypes>
        using all_integers=typename boost::enable_if_c<accumulate(logical_and(),  boost::is_integral<IntTypes>::type::value ... ), bool >::type;

        /**
           @brief empty constructor
        */
        constexpr meta_storage_base(){}


#ifdef CXX11_ENABLED
        constexpr meta_storage_base(array<uint_t, space_dimensions> const& a) :
            m_dims(space_dimensions),
            m_strides(explode<
                   array<int_t, (short_t)(space_dimensions)>,
                   _impl::assign_all_strides< (short_t)(space_dimensions), layout>
                   >(a))
        {}
#else
         //TODO This is a bug, we should generate a constructor for array of dimensions space_dimensions
        meta_storage_base(array<uint_t, 3> const& a)
            : m_dims(a)
        {
            m_strides[0]=( ((layout::template at_<0>::value < 0)?1:m_dims[0]) * ((layout::template at_<1>::value < 0)?1:m_dims[1]) * ((layout::template at_<2>::value < 0)?1:m_dims[2]) );
            m_strides[1]=( (m_strides[0]<=1)?0:layout::template find_val<2,uint_t,1>(m_dims)*layout::template find_val<1,uint_t,1>(m_dims) );
            m_strides[2]=( (m_strides[1]<=1)?0:layout::template find_val<2,uint_t,1>(m_dims) );
        }
d
#endif
        template<typename T>
        constexpr meta_storage_base(T const& a)
        {
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::is_void_<T>::value), "Metastorage Constructor not supported");
        }

        // variadic constexpr constructor

        /**
           @brief constructor given the space dimensions

           NOTE: this contructor is constexpr, i.e. the storage metadata information could be used
           at compile-time (e.g. in template metafunctions)
         */
        template <class ... IntTypes
#ifndef __CUDACC__
                  , typename Dummy = all_integers<IntTypes...> //nvcc does not get it
#endif
                  >
        constexpr meta_storage_base(  IntTypes const& ... dims_  ) :
            m_dims(dims_...)
            , m_strides(_impl::assign_all_strides< (short_t)(space_dimensions), layout>::apply( dims_...))
            {
                GRIDTOOLS_STATIC_ASSERT(sizeof...(IntTypes)==space_dimensions, "you tried to initialize a storage with a number of integer arguments different from its number of dimensions. This is not allowed. If you want to fake a lower dimensional storage, you have to add explicitly a \"1\" on the dimension you want to kill. Otherwise you can use a proper lower dimensional storage by defining the storage type using another layout_map.");
            }
#else
        // non variadic non constexpr constructor
        meta_storage_base(  uint_t const& d1, uint_t const& d2, uint_t const& d3 ) :
            m_dims(d1, d2, d3)
            {
                m_strides[0]=( ((layout::template at_<0>::value < 0)?1:d1) * ((layout::template at_<1>::value < 0)?1:d2) * ((layout::template at_<2>::value < 0)?1:d3) );
                m_strides[1]=( (m_strides[0]<=1)?0:layout::template find_val<2,short_t,1>(d1,d2,d3)*layout::template find_val<1,short_t,1>(d1,d2,d3) );
                m_strides[2]=( (m_strides[1]<=1)?0:layout::template find_val<2,short_t,1>(d1,d2,d3) );
            }
#endif

        /**
            @brief constexpr copy constructor

            copy constructor, used e.g. to generate the gpu clone of the storage metadata.
         */
        GT_FUNCTION
        constexpr meta_storage_base( meta_storage_base const&other )
            : m_dims(other.m_dims)
            , m_strides(other.m_strides)
            {
                //GRIDTOOLS_STATIC_ASSERT(is_meta_storage<Other>::type::value, "Type error");
            }

        /** @brief prints debugging information */
         void info() const {
            std::cout << dims<0>() << "x"
                      << dims<1>() << "x"
                      << dims<2>() << ", "
                      << std::endl;
        }

        /**@brief returns the size of the data field*/
        GT_FUNCTION
        constexpr uint_t size() const { //cast to uint_t
            return m_strides[0];
        }

        /** @brief returns the dimension fo the field along I*/
        template<ushort_t I>
        GT_FUNCTION
         constexpr uint_t dims() const {return m_dims[I];}

        /** @brief returns the dimension fo the field along I*/
        GT_FUNCTION
         constexpr uint_t dims(const ushort_t I) const {return m_dims[I];}

        /**@brief returns the storage strides
         */
        GT_FUNCTION
         constexpr int_t const& strides(ushort_t i) const {
            return m_strides[i];
        }

        /**@brief returns the storage strides
         */
        GT_FUNCTION
         constexpr int_t const* strides() const {
            GRIDTOOLS_STATIC_ASSERT(space_dimensions>1, "one dimensional storage");
            return (&m_strides[1]);
        }

#ifdef CXX11_ENABLED

        /**@brief straightforward interface*/
        template <typename ... UInt>
        GT_FUNCTION
        uint_t index(uint_t const& first, UInt const& ... args_) const {
            return _index(strides(), first, args_... );
        }

        struct _impl_index{
            template<typename ... UIntType>
            static uint_t apply(const type& me, UIntType ... args){
                return me.index(args...);
            }
        };

        template<size_t S>
        GT_FUNCTION
        uint_t index(array<uint, S> a) const {
            return explode<uint_t, _impl_index>(a, *this);
        }
#else
        /**@brief straightforward interface*/
        GT_FUNCTION
        uint_t index(uint_t const& i, uint_t const& j, uint_t const&  k) const { return _index(strides(), i, j, k); }
#endif

        //####################################################
        // static functions (independent from the storage)
        //####################################################


        /**@brief return the stride for a specific coordinate, given the vector of strides
           Coordinates 0,1,2 correspond to i,j,k respectively*/
        template<uint_t Coordinate, typename StridesVector>
        GT_FUNCTION
        static constexpr int_t strides(StridesVector const& RESTRICT strides_){
            return ((vec_max<typename layout::layout_vector_t>::value < 0) ? 0:(( layout::template at_<Coordinate>::value == vec_max<typename layout::layout_vector_t>::value ) ? 1 : ((strides_[layout::template at_<Coordinate>::value/*+1*/]))));//POL TODO explain the fact that here there was a +1
        }

        /**@brief returning the index of the memory address corresponding to the specified (i,j,k) coordinates.
           This method depends on the strategy used (either naive or blocking). In case of blocking strategy the
           index for temporary storages is computed in the subclass gridtools::host_tmp_storge
           NOTE: this version will be preferred over the templated overloads
        */
        template<typename StridesVector>
        GT_FUNCTION
        static constexpr uint_t _index(StridesVector const& RESTRICT strides_, uint_t const& i, uint_t const& j, uint_t const&  k) {
            return strides_[0]
                * layout::template find_val<0,uint_t,0>(i,j,k) +
                strides_[1] * layout::template find_val<1,uint_t,0>(i,j,k) +
                layout::template find_val<2,uint_t,0>(i,j,k);
        }

#ifdef CXX11_ENABLED
        /**
           @brief computing index to access the storage in the coordinates passed as parameters.

           This method must be called with integral type parameters, and the result will be a positive integer.
        */
        template <typename StridesVector, typename ... UInt>
        GT_FUNCTION
        constexpr
        static uint_t _index(StridesVector const& RESTRICT strides_, UInt const& ... dims) {
#ifndef __CUDACC__
            GRIDTOOLS_STATIC_ASSERT(accumulate(logical_and(),  boost::is_integral<UInt>::type::value ...), "you have to pass in arguments of uint_t type");
#endif
            return _impl::compute_offset<space_dimensions, layout>::apply(strides_, dims ...);
        }
#endif

        /**
           @brief computing index to access the storage in the coordinates passed as a tuple.

           \param StridesVector the vector of strides, it is a contiguous array of length space_dimenisons-1
           \param tuple is a tuple of coordinates, of type \ref gridtools::offset_tuple

           This method returns signed integers of type int_t (used e.g. in iterate_domain)
        */
        template <typename StridesVector, typename OffsetTuple>
        GT_FUNCTION
        static constexpr int_t _index(StridesVector const& RESTRICT strides_, OffsetTuple  const& tuple) {
            return _impl::compute_offset<space_dimensions, layout>::apply(strides_, tuple);
        }

        template <typename OffsetTuple>
        GT_FUNCTION
        constexpr int_t _index(OffsetTuple  const& tuple) {
            return _impl::compute_offset<space_dimensions, layout>::apply(strides(), tuple);
        }

        /** @brief returns the memory access index of the element with coordinate passed as an array

            \param StridesVector the vector of strides, it is a contiguous array of length space_dimenisons-1
            \param indices array of coordinates

            This method returns a signed int_t  (used e.g. in iterate_domain)*/
        template<typename IntType, typename StridesVector>
        GT_FUNCTION
        static constexpr int_t _index( StridesVector const& RESTRICT strides_, IntType* RESTRICT indices) {

            return  _impl::compute_offset<space_dimensions, layout>::apply(strides_, indices);
        }

        // /** @brief method to increment the memory address index by moving forward one step in the given Coordinate direction

        //     SFINAE: design pattern used to avoid the compilation of the overloaded method which are not used (and which would produce a compilation error)
        // */
        // template <uint_t Coordinate, enumtype::execution Execution, typename StridesVector>
        // GT_FUNCTION
        // static void increment( int_t* RESTRICT index_, StridesVector const& RESTRICT strides_){
        //     BOOST_STATIC_ASSERT(Coordinate < space_dimensions);
        //     if(layout::template at_< Coordinate >::value >=0)//static if
        //     {
        //         increment_policy<Execution>::apply(*index_ , strides<Coordinate>(strides_));
        //     }
        // }

        /** @brief method to increment the memory address index by moving forward a given number of step in the given Coordinate direction
            \tparam Coordinate: the dimension which is being incremented (0=i, 1=j, 2=k, ...)
            \param steps: the number of steps of the increment
            \param index: the output index being set
        */
        template <uint_t Coordinate, typename StridesVector>
        GT_FUNCTION
        static void increment(int_t const& steps_, int_t* RESTRICT index_, StridesVector const& RESTRICT strides_){
#ifdef PEDANTIC
            GRIDTOOLS_STATIC_ASSERT(Coordinate < space_dimensions, "you have a storage in the iteration space whoose dimension is lower than the iteration space dimension. This might not be a problem, since trying to increment a nonexisting dimension has no effect. In case you want this feature comment out this assert.");

#endif
            if( layout::template at_< Coordinate >::value >= 0 )//static if
            {
#ifdef CXX11_ENABLED
                GRIDTOOLS_STATIC_ASSERT(StridesVector::size()==space_dimensions-1, "error: trying to compute the storage index using strides from another storage which does not have the same space dimensions. Are you explicitly incrementing the iteration space by calling base_storage::increment?");
#endif
                    *index_ += strides<Coordinate>(strides_)*steps_;
            }
        }

        /**
           @brief initializing a given coordinate (i.e. multiplying times its stride)

           \param steps_ the input coordinate value
           \param index_ the output index
           \param strides_ the strides array
         */
        template <uint_t Coordinate, typename StridesVector >
        GT_FUNCTION
        static void initialize(uint_t const& steps_, uint_t const& /*block*/, int_t* RESTRICT index_, StridesVector const& RESTRICT strides_){

            if( Coordinate < space_dimensions && layout::template at_< Coordinate >::value >= 0 )//static if
            {
#ifdef CXX11_ENABLED
                GRIDTOOLS_STATIC_ASSERT(StridesVector::size()==space_dimensions-1, "error: trying to compute the storage index using strides from another storages which does not have the same space dimensions. Sre you explicitly initializing the iteration space by calling base_storage::initialize?");
#endif
                    *index_+=strides<Coordinate>(strides_)*steps_;
            }
        }


        /**
           returning 0 in a non blocked storage
        */
        GT_FUNCTION
        uint_t fields_offset(int_t EU_id_i, int_t EU_id_j) const {
            return 0;
        }

    };


}//namespace gridtools