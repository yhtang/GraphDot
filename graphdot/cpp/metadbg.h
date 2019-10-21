#ifndef GRAPHDOT_METADBG_H_
#define GRAPHDOT_METADBG_H_

template<class...>   struct DISPLAY_TYPE_;
template<int>        struct DISPLAY_INT_;
template<bool>       struct DISPLAY_BOOL_;

template<class T> constexpr DISPLAY_TYPE_<T> DISPLAY_TYPE__();
template<int   I> constexpr DISPLAY_INT_ <I> DISPLAY_INT__();
template<bool  B> constexpr DISPLAY_BOOL_<B> DISPLAY_BOOL__();

#define DISPLAY_TYPE( t ) DISPLAY_TYPE__<decltype(t)>()
#define DISPLAY_INT( i )  DISPLAY_INT__<i>()
#define DISPLAY_BOOL( b ) DISPLAY_BOOL__<b>()

#endif
