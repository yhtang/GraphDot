#include <cstdio>
#include <cstring>
#include <limits>
#include <algorithm>
#include <omp.h>

char elements[][4] = {
    "Ac",
    "Ag",
    "Al",
    "Am",
    "Ar",
    "As",
    "At",
    "Au",
    "B",
    "Ba",
    "Be",
    "Bh",
    "Bi",
    "Bk",
    "Br",
    "C",
    "Ca",
    "Cd",
    "Ce",
    "Cf",
    "Cl",
    "Cm",
    "Co",
    "Cr",
    "Cs",
    "Cu",
    "Db",
    "Ds",
    "Dy",
    "Er",
    "Es",
    "Eu",
    "F",
    "Fe",
    "Fm",
    "Fr",
    "Ga",
    "Gd",
    "Ge",
    "H",
    "He",
    "Hf",
    "Hg",
    "Ho",
    "Hs",
    "I",
    "In",
    "Ir",
    "K",
    "Kr",
    "La",
    "Li",
    "Lr",
    "Lu",
    "Md",
    "Mg",
    "Mn",
    "Mo",
    "Mt",
    "N",
    "Na",
    "Nb",
    "Nd",
    "Ne",
    "Ni",
    "No",
    "Np",
    "O",
    "Os",
    "P",
    "Pa",
    "Pb",
    "Pd",
    "Pm",
    "Po",
    "Pr",
    "Pt",
    "Pu",
    "Ra",
    "Rb",
    "Re",
    "Rf",
    "Rg",
    "Rh",
    "Rn",
    "Ru",
    "S",
    "Sb",
    "Sc",
    "Se",
    "Sg",
    "Si",
    "Sm",
    "Sn",
    "Sr",
    "Ta",
    "Tb",
    "Tc",
    "Te",
    "Th",
    "Ti",
    "Tl",
    "Tm",
    "U",
    "Uub",
    "Uuh",
    "Uuo",
    "Uup",
    "Uuq",
    "Uus",
    "Uut",
    "V",
    "W",
    "Xe",
    "Y",
    "Yb",
    "Zn",
    "Zr"
};

int main() {

    //using hash_type = unsigned char;
	using hash_type = unsigned short int;

    //int min_collision = 256;
    int min_scatter   = std::numeric_limits<hash_type>::max() - std::numeric_limits<hash_type>::min();

    int a_max = 256;
    int b_max = 256;
    int c_max = 256;

    hash_type mod = 512;

    for ( int a = 0; a < a_max; ++a ) {
        printf( "a %d\n", a );
        for ( int b = 0; b < b_max; ++b ) {
            //printf( "b %d\n", b );
			#pragma omp parallel for
            for ( int c = 0; c < c_max; ++c ) {
            	for(int b_fallback = 0; b_fallback < 256; ++b_fallback ) {
            		for(int c_fallback = 0; c_fallback < 1; ++c_fallback ) {

                        int used[1 << ( sizeof( hash_type ) * 8 )];
                        int collision = 0;
                        hash_type hash_max = 0, hash_min = std::numeric_limits<hash_type>::max();
                        memset( used, 0, sizeof(used) );

                        for ( int i = 0; i < sizeof( elements ) / 4; ++i ) {
                            hash_type hash = 0;

                            auto l = strlen( elements[i] );
                            hash += elements[i][0] * hash_type(a);
                            hash += ( l > 1 ? elements[i][1] : b_fallback ) * hash_type(b);
                            hash += ( l > 2 ? elements[i][2] : c_fallback ) * hash_type(c);
                            hash = hash % mod;

                            if ( hash > hash_max ) {
                            	hash_max = std::max( hash, hash_max );
                            }
                            if ( hash < hash_min ) {
                            	hash_min = std::min( hash, hash_min );
                            }
                            if ( used[hash] != 0 ) {
                                ++collision;
                            }
                            used[hash] = i;
                        }
                        int scatter = hash_max - hash_min;
                        if ( collision == 0 ) {
                        	if (scatter < min_scatter) {
								#pragma omp critical
                        		if (scatter < min_scatter) {
									min_scatter = scatter;
									printf( "%d %d %d %d %d collision %d scatter %d\n", a, b, c, b_fallback, c_fallback, collision, scatter );
                        		}
                        	}
                        }
            		}
            	}
            }
        }
    }
}
