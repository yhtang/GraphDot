#ifndef GRAPHDOT_CPP_SPATIAL_KDTREE_H_
#define GRAPHDOT_CPP_SPATIAL_KDTREE_H_

#include <cstdint>
#include <vector>
#include <Eigen/Dense>

namespace graphdot {

namespace spatial {

template<class T, std::size_t D>
struct point_t : Eigen::Matrix<T,D,1> {
    using base_t = Eigen::Matrix<T,D,1>;

    using base_t::base_t;

    inline point_t(double initializer) {
        for(std::size_t i = 0 ; i < D ; i++) (*this)(i) = initializer;
    }
};

template<typename T, std::size_t D, std::size_t LeafSize = 16>
class kdtree {
    static_assert( D < 256, "kd-tree is not effective for high dimensional space" );
//protected:
    using point  = point_t<T,D>;
    using dim_t  = std::uint8_t;
    using size_t = std::size_t;
    std::vector<point>  pts;
    std::vector<size_t> idx;

public:
    struct node {
        std::uint8_t d = 0; // direction of bisection, i.e. the axial direction that the bisection plane is perpendicular to
        T lo, hi;           // extreme coordinate along bisection direction
        union { T inner_bound[2]; size_t range[2]; }; // for leaf nodes: range in effect, denote left and right range
                                            // for internal nodes: inner_bound in effect, upper bound of left child + lower bound of right child
        node * parent = nullptr, *child = nullptr, *next = nullptr;

        node() {
            // TODO: try remove the two lines below
            inner_bound[0] = -std::numeric_limits<T>::max();
            inner_bound[1] =  std::numeric_limits<T>::max();
        }
    } root;

public:
    kdtree() = default;
    kdtree( kdtree const & ) = delete;
    kdtree & operator = ( kdtree const & ) = delete;
    ~kdtree() { clear(); }

    template<class InputIter>
    void build_serial( InputIter first, InputIter last ) {

        pts.resize( std::distance( first, last ) );
        std::copy( first, last, pts.begin() );

        idx.resize( pts.size() );
        for ( size_t i = 0; i < idx.size(); ++i ) idx[i] = i;

        allocator.refresh();
        root.child = root.parent = root.next = nullptr;

        __build__( &root, 0, pts.size(), pts.data() );

        std::vector<point> pts_reordered( pts.size() );
        for ( size_t i = 0; i < idx.size(); ++i ) pts_reordered[i] = pts[ idx[i] ];
        std::swap( pts, pts_reordered );

        // build shortcut to the next branch
        auto find_next = [&]( node * n ) {
            node * p = n;
            while ( p->parent && p == p->parent->child + 1 ) p = p->parent; // go up until being a left child
            if ( p->parent ) n->next = p->parent->child + 1;
            else n->next = nullptr;
        };
        depth_first_traversal( &root, find_next );
    }

    void __build__( node * n, size_t l, size_t r, point * points ) {

        // choose dimension with longest span
        point lo = points[ idx[l] ];
        point hi = points[ idx[l] ];
        for ( int i = l; i < r; i++ ) {
            lo = lo.cwiseMin( points[ idx[i] ] );
            hi = hi.cwiseMax( points[ idx[i] ] );
        }
        int d;
        ( hi - lo ).maxCoeff( &d );
        n->d   = d;
        n->lo  = lo[d];
        n->hi  = hi[d];

        if ( r - l > LeafSize ) {
            // bisect by reordering point indices

            // quick sort-like algo to evenly split range with left half < right half
            size_t m, active_left = l, active_right = r;
            do {
                // use a random pivot to bisect current range
                T pivot = points[ idx[ rand( active_left, active_right ) ] ][d];

                auto i = idx.data() + active_left;
                auto j = idx.data() + active_right - 1;

                do {
                    while ( i - idx.data() < r && points[*i][d] <  pivot ) ++i;
                    while ( j - idx.data() > l && points[*j][d] >= pivot ) --j;
                    if ( j - i > 0 ) std::swap( *i, *j );
                } while ( j - i > 1 );

                // if pivot ends up on left, recursively bisect right range, and vice versa
                // until pivot ends up in middle
                m = std::min( i - idx.data(), j - idx.data() ) + 1;
                if ( m == active_left || m == active_right ) {
                    bool same = true;
                    for ( int i = active_left; same && i < active_right; i++ ) same &= ( points[ idx[active_left] ][d] == points[ idx[i] ][d] );
                    if ( same ) break;
                } else if ( m == ( l + r ) / 2 ) break;
                else if ( m < ( l + r ) / 2 ) active_left = m;
                else active_right = m;

            } while ( true );

            for ( int k = l; k < m; k++ ) n->inner_bound[0] = std::max( n->inner_bound[0], points[idx[k]][d] );
            for ( int k = m; k < r; k++ ) n->inner_bound[1] = std::min( n->inner_bound[1], points[idx[k]][d] );
            n->child = allocator.allocate( 2 );
            n->child[0].parent = n->child[1].parent = n;
            __build__( n->child + 0, l, m, points );
            __build__( n->child + 1, m, r, points );
        } else {
            n->range[0] = l;
            n->range[1] = r;
        }
    }

    std::pair<T, size_t> find_nearest( point const & q, std::pair<T, size_t> ig = { std::numeric_limits<T>::max(), -1 } ) const {

        auto [ min_r, min_i ] = ig;
        if ( min_i == -1 ) {
            size_t x = rand( 0, pts.size() );
            min_i = idx[ x ];
            min_r = ( pts[ x ] - q ).norm();
        }

        node const * p = &root;
        T r;
        size_t i;
        do {
            while ( p->child && in_box( q, p, min_r ) ) { // descend to the nearest leave
                if ( q[p->d] - min_r < p->inner_bound[0] ) p = p->child + 0; // trying left tree
                else if ( q[p->d] + min_r > p->inner_bound[1] ) p = p->child + 1; // tryign right tree
                else break; // both end no worth trying
            }
            if ( !p->child && in_box( q, p, min_r ) ) {
                std::tie( r, i ) = nearest( p->range[0], p->range[1], q );
                if ( r < min_r ) {
                    min_r = r;
                    min_i = i;
                }
            }
            do { // march to the next unsearched branch
                p = p->next;
            } while ( p && q[p->parent->d] + min_r <= p->parent->inner_bound[1] );
        } while ( p ); // exit search when back to root

        return std::make_pair( min_r, min_i );
    }

    // return first nmax points within max_r
    size_t find_within( point const & q, T max_r, size_t * __restrict result, size_t nmax ) const {
        node const * p = &root;
        size_t nfound = 0;
        do {
            while ( p->child && in_box( q, p, max_r ) ) { // descend to the nearest leave
                if ( q[p->d] - max_r < p->inner_bound[0] ) p = p->child + 0; // trying left tree
                else if ( q[p->d] + max_r > p->inner_bound[1] ) p = p->child + 1; // tryign right tree
                else break; // both end no worth trying
            }
            if ( !p->child && in_box( q, p, max_r ) ) {
                for ( size_t i = p->range[0]; i < p->range[1]; i++ ) {
                    if ( normsq( pts[ i ] - q ) < max_r * max_r ) {
                        result[nfound++] = idx[i];
                        if ( nfound == nmax ) return nfound;
                    }
                }
            }
            do { // march to the next unsearched branch
                p = p->next;
            } while ( p && q[p->parent->d] + max_r <= p->parent->inner_bound[1] );
        } while ( p ); // exit search when back to root
        return nfound;
    }

    void clear() {
        root.child = root.parent = root.next = nullptr;
        pts.clear();
        idx.clear();
        allocator.refresh();
    }

//protected:
    struct NodeAllocator {
        std::vector<node *> ptrs_free, ptrs_used;
        node * lead = nullptr;
        std::size_t stock = 0;
        const static std::size_t chunk_size = 2048;

        ~NodeAllocator() {
            for ( auto & p : ptrs_used ) delete [] p;
            for ( auto & p : ptrs_free ) delete [] p;
        }

        node * allocate( std::size_t n ) {
            if ( stock < n ) {
                if ( lead ) ptrs_used.push_back( lead );
                if ( !ptrs_free.size() ) {
                    lead = new node[ chunk_size ];
                } else {
                    lead = ptrs_free.back();
                    ptrs_free.pop_back();
                }
                stock = chunk_size;
            }
            node * ret = new( lead + chunk_size - stock ) node[ n ];
            stock -= n;
            return ret;
        }

        void refresh() {
            for ( auto & p : ptrs_used ) ptrs_free.push_back( p );
            ptrs_used.clear();
            if ( lead ) ptrs_free.push_back( lead );
            lead = nullptr;
            stock = 0;
        }
    };

    NodeAllocator allocator;

    inline bool in_box( point const & q, node const * p, T min_r ) const {
        return ( q[p->d] - min_r < p->hi ) & ( q[p->d] + min_r > p->lo ); // non-short-circuit version faster by saving branches
    }

    // small-range brute-force search
    std::pair<T, size_t> nearest( size_t l, size_t r, point q ) const {
        size_t p;
        auto r2min = std::numeric_limits<T>::max();
        for ( size_t i = l; i < r; i++ ) {
            auto r2 = ( pts[ i ] - q ).squaredNorm();
            if ( r2 < r2min ) {
                p = i;
                r2min = r2;
            }
        }
        return std::make_pair( std::sqrt( r2min ), idx[p] );
    }

    template<class VISITOR> void depth_first_traversal( node * n, VISITOR const & visitor ) {
        if ( n->child ) {
            depth_first_traversal( n->child + 0, visitor );
            depth_first_traversal( n->child + 1, visitor );
        }
        visitor( n );
    }

    template<class I, class J> static inline std::common_type_t<I,J> rand( I l, J r ) {
        static uint seed = 0;
        return ( seed = seed * 1664525U + 1013904223U ) % ( r - l ) + l;
    }
};

}

}

#endif
