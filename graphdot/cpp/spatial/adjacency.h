#ifndef GRAPHDOT_CPP_SPATIAL_ADJACENCY_H
#define GRAPHDOT_CPP_SPATIAL_ADJACENCY_H

#include <map>
#include <utility>
#include <tuple>
#include <math/radialbasis.h>
#include <math/distance.h>
#include <misc/periodic_table.h>

namespace graphdot {
namespace spatial {

struct compact_adjacency {

    constexpr static bool hint_is_sparse_ = true;

    compact_adjacency( double zoom, int order ) : zoom( zoom ), order( order ) {
        bonding_distance[ {Element::H, Element::H} ] = 0.74;
        bonding_distance[ {Element::H, Element::C} ] = 1.09;
        bonding_distance[ {Element::H, Element::O} ] = 0.96;
        bonding_distance[ {Element::H, Element::N} ] = 1.01;
        bonding_distance[ {Element::H, Element::F} ] = 0.92;
        bonding_distance[ {Element::H, Element::S} ] = 1.34;
        bonding_distance[ {Element::C, Element::H} ] = 1.09;
        bonding_distance[ {Element::C, Element::C} ] = 1.39;
        bonding_distance[ {Element::C, Element::O} ] = 1.27;
        bonding_distance[ {Element::C, Element::N} ] = 1.34;
        bonding_distance[ {Element::C, Element::F} ] = 1.35;
        bonding_distance[ {Element::C, Element::S} ] = 1.82;
        bonding_distance[ {Element::O, Element::H} ] = 0.96;
        bonding_distance[ {Element::O, Element::C} ] = 1.27;
        bonding_distance[ {Element::O, Element::O} ] = 1.48;
        bonding_distance[ {Element::O, Element::N} ] = 1.23;
        bonding_distance[ {Element::O, Element::F} ] = 1.42;
        bonding_distance[ {Element::O, Element::S} ] = 1.44;
        bonding_distance[ {Element::N, Element::H} ] = 1.01;
        bonding_distance[ {Element::N, Element::C} ] = 1.34;
        bonding_distance[ {Element::N, Element::O} ] = 1.23;
        bonding_distance[ {Element::N, Element::N} ] = 1.26;
        bonding_distance[ {Element::N, Element::F} ] = 1.38;
        bonding_distance[ {Element::N, Element::S} ] = 1.68;
        bonding_distance[ {Element::F, Element::H} ] = 0.92;
        bonding_distance[ {Element::F, Element::C} ] = 1.35;
        bonding_distance[ {Element::F, Element::O} ] = 1.42;
        bonding_distance[ {Element::F, Element::N} ] = 1.38;
        bonding_distance[ {Element::F, Element::F} ] = 1.42;
        bonding_distance[ {Element::F, Element::S} ] = 1.57;
        bonding_distance[ {Element::S, Element::H} ] = 1.34;
        bonding_distance[ {Element::S, Element::C} ] = 1.82;
        bonding_distance[ {Element::S, Element::O} ] = 1.44;
        bonding_distance[ {Element::S, Element::N} ] = 1.68;
        bonding_distance[ {Element::S, Element::F} ] = 1.57;
        bonding_distance[ {Element::S, Element::S} ] = 2.05;
    }

    template<class Atom> auto operator () ( Atom && atom1, Atom && atom2 ) const {
        auto const &[s1, x1] = std::forward<Atom>( atom1 );
        auto const &[s2, x2] = std::forward<Atom>( atom2 );
        auto cutoff = bonding_distance.at( {s1, s2} ) * zoom;
        auto dr = distance( x1, x2 );
        auto w  = dr < cutoff ? rbf::tent( dr, order, cutoff ) : 0.0;
        return std::make_tuple( dr, w );
    }

protected:
    double zoom;
    int order;
    std::map<std::tuple<Element, Element>, double> bonding_distance;
};

struct gaussian_adjacency {

    constexpr static bool hint_is_sparse_ = false;

    gaussian_adjacency( double zoom = 1.0 ) {
        rsqinv[ {Element::H, Element::H} ] = std::pow( 0.74 * zoom, -2.0 );
        rsqinv[ {Element::H, Element::C} ] = std::pow( 1.09 * zoom, -2.0 );
        rsqinv[ {Element::H, Element::O} ] = std::pow( 0.96 * zoom, -2.0 );
        rsqinv[ {Element::H, Element::N} ] = std::pow( 1.01 * zoom, -2.0 );
        rsqinv[ {Element::H, Element::F} ] = std::pow( 0.92 * zoom, -2.0 );
        rsqinv[ {Element::H, Element::S} ] = std::pow( 1.34 * zoom, -2.0 );
        rsqinv[ {Element::C, Element::H} ] = std::pow( 1.09 * zoom, -2.0 );
        rsqinv[ {Element::C, Element::C} ] = std::pow( 1.39 * zoom, -2.0 );
        rsqinv[ {Element::C, Element::O} ] = std::pow( 1.27 * zoom, -2.0 );
        rsqinv[ {Element::C, Element::N} ] = std::pow( 1.34 * zoom, -2.0 );
        rsqinv[ {Element::C, Element::F} ] = std::pow( 1.35 * zoom, -2.0 );
        rsqinv[ {Element::C, Element::S} ] = std::pow( 1.82 * zoom, -2.0 );
        rsqinv[ {Element::O, Element::H} ] = std::pow( 0.96 * zoom, -2.0 );
        rsqinv[ {Element::O, Element::C} ] = std::pow( 1.27 * zoom, -2.0 );
        rsqinv[ {Element::O, Element::O} ] = std::pow( 1.48 * zoom, -2.0 );
        rsqinv[ {Element::O, Element::N} ] = std::pow( 1.23 * zoom, -2.0 );
        rsqinv[ {Element::O, Element::F} ] = std::pow( 1.42 * zoom, -2.0 );
        rsqinv[ {Element::O, Element::S} ] = std::pow( 1.44 * zoom, -2.0 );
        rsqinv[ {Element::N, Element::H} ] = std::pow( 1.01 * zoom, -2.0 );
        rsqinv[ {Element::N, Element::C} ] = std::pow( 1.34 * zoom, -2.0 );
        rsqinv[ {Element::N, Element::O} ] = std::pow( 1.23 * zoom, -2.0 );
        rsqinv[ {Element::N, Element::N} ] = std::pow( 1.26 * zoom, -2.0 );
        rsqinv[ {Element::N, Element::F} ] = std::pow( 1.38 * zoom, -2.0 );
        rsqinv[ {Element::N, Element::S} ] = std::pow( 1.68 * zoom, -2.0 );
        rsqinv[ {Element::F, Element::H} ] = std::pow( 0.92 * zoom, -2.0 );
        rsqinv[ {Element::F, Element::C} ] = std::pow( 1.35 * zoom, -2.0 );
        rsqinv[ {Element::F, Element::O} ] = std::pow( 1.42 * zoom, -2.0 );
        rsqinv[ {Element::F, Element::N} ] = std::pow( 1.38 * zoom, -2.0 );
        rsqinv[ {Element::F, Element::F} ] = std::pow( 1.42 * zoom, -2.0 );
        rsqinv[ {Element::F, Element::S} ] = std::pow( 1.57 * zoom, -2.0 );
        rsqinv[ {Element::S, Element::H} ] = std::pow( 1.34 * zoom, -2.0 );
        rsqinv[ {Element::S, Element::C} ] = std::pow( 1.82 * zoom, -2.0 );
        rsqinv[ {Element::S, Element::O} ] = std::pow( 1.44 * zoom, -2.0 );
        rsqinv[ {Element::S, Element::N} ] = std::pow( 1.68 * zoom, -2.0 );
        rsqinv[ {Element::S, Element::F} ] = std::pow( 1.57 * zoom, -2.0 );
        rsqinv[ {Element::S, Element::S} ] = std::pow( 2.05 * zoom, -2.0 );
    }

    template<class Atom> auto operator () ( Atom && atom1, Atom && atom2 ) const {
        auto const &[s1, x1] = std::forward<Atom>( atom1 );
        auto const &[s2, x2] = std::forward<Atom>( atom2 );
        auto dr = distance( x1, x2 );
        auto w  = std::exp( -0.5 * ipow( dr, 2 ) * rsqinv.at( {s1, s2} ) );
        return std::make_tuple( dr, w );
    }

protected:
    std::map<std::tuple<Element, Element>, double> rsqinv;
};

}
}

#endif
