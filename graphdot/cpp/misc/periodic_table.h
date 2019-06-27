#ifndef GRAPHDOT_MISC_PERIODIC_TABLE_H_
#define GRAPHDOT_MISC_PERIODIC_TABLE_H_

#include <cstdint>
#include <cstring>
#include <string>

#include <misc/hash.h>

namespace graphdot {

enum class Element {
	_  =  0,
	H  =  1, He =  2, Li =  3, Be =  4, B  =  5, C  =  6, N  =  7, O  =  8,
	F  =  9, Ne = 10, Na = 11, Mg = 12, Al = 13, Si = 14, P  = 15, S  = 16,
	Cl = 17, Ar = 18, K  = 19, Ca = 20, Sc = 21, Ti = 22, V  = 23, Cr = 24,
	Mn = 25, Fe = 26, Co = 27, Ni = 28, Cu = 29, Zn = 30, Ga = 31, Ge = 32,
	As = 33, Se = 34, Br = 35, Kr = 36, Rb = 37, Sr = 38, Y  = 39, Zr = 40,
	Nb = 41, Mo = 42, Tc = 43, Ru = 44, Rh = 45, Pd = 46, Ag = 47, Cd = 48,
	In = 49, Sn = 50, Sb = 51, Te = 52, I  = 53, Xe = 54, Cs = 55, Ba = 56,
	La = 57, Ce = 58, Pr = 59, Nd = 60, Pm = 61, Sm = 62, Eu = 63, Gd = 64,
	Tb = 65, Dy = 66, Ho = 67, Er = 68, Tm = 69, Yb = 70, Lu = 71, Hf = 72,
	Ta = 73, W  = 74, Re = 75, Os = 76, Ir = 77, Pt = 78, Au = 79, Hg = 80,
	Tl = 81, Pb = 82, Bi = 83, Po = 84, At = 85, Rn = 86, Fr = 87, Ra = 88,
	Ac = 89, Th = 90, Pa = 91, U  = 92, Np = 93, Pu = 94, Am = 95, Cm = 96,
	Bk = 97, Cf = 98, Es = 99, Fm =100, Md =101, No =102, Lr =103, Rf =104,
	Db =105, Sg =106, Bh =107, Hs =108, Mt =109, Ds =110, Rg =111, Uub=112,
	Uut=113, Uuq=114, Uup=115, Uuh=116, Uus=117, Uuo=118
};

static Element _str2elem_lut[] = {
		Element::Ac , Element::_  , Element::_  , Element::_  , Element::Ag , Element::_  , Element::_  , Element::S  ,
		Element::_  , Element::Al , Element::Am , Element::_  , Element::_  , Element::_  , Element::_  , Element::Ar ,
		Element::As , Element::At , Element::Au , Element::Xe , Element::_  , Element::Ba , Element::_  , Element::_  ,
		Element::_  , Element::Be , Element::_  , Element::_  , Element::Bh , Element::Bi , Element::_  , Element::Bk ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::Br , Element::Yb ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::Ca , Element::_  , Element::_  , Element::Cd ,
		Element::Ce , Element::Cf , Element::_  , Element::_  , Element::_  , Element::U  , Element::_  , Element::Cl ,
		Element::Cm , Element::_  , Element::Co , Element::_  , Element::_  , Element::Cr , Element::Cs , Element::_  ,
		Element::Cu , Element::_  , Element::_  , Element::_  , Element::Db , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::Zn , Element::_  , Element::V  , Element::_  , Element::Zr , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::Ds , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::Dy , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::W  , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::Er , Element::Es , Element::_  , Element::Eu , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::Fe , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::Fm , Element::_  , Element::_  ,
		Element::B  , Element::_  , Element::Fr , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::Ga , Element::_  , Element::_  , Element::Gd , Element::Ge , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::Y  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::C  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::Uub, Element::He , Element::Hf , Element::Hg , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::Ho , Element::Uuh, Element::_  ,
		Element::_  , Element::Hs , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::Uuo, Element::_  , Element::Uup, Element::_  ,
		Element::Uuq, Element::_  , Element::_  , Element::In , Element::Uus, Element::_  , Element::Uut, Element::Ir ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::F  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::Kr , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::La , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::Li , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::H  , Element::_  , Element::Lr , Element::_  , Element::_  , Element::Lu ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::Md , Element::_  , Element::_  ,
		Element::Mg , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::Mn ,
		Element::Mo , Element::I  , Element::_  , Element::_  , Element::_  , Element::Mt , Element::_  , Element::_  ,
		Element::_  , Element::Na , Element::Nb , Element::_  , Element::Nd , Element::Ne , Element::_  , Element::_  ,
		Element::_  , Element::Ni , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::No ,
		Element::Np , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::K  ,
		Element::_  , Element::_  , Element::Os , Element::_  , Element::_  , Element::_  , Element::_  , Element::Pa ,
		Element::Pb , Element::_  , Element::Pd , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::Pm , Element::_  , Element::Po , Element::_  , Element::_  ,
		Element::Pr , Element::_  , Element::Pt , Element::Pu , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::_  , Element::_  , Element::Ra , Element::Rb , Element::_  ,
		Element::_  , Element::Re , Element::Rf , Element::Rg , Element::Rh , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::Rn , Element::_  , Element::N  , Element::_  , Element::_  , Element::_  ,
		Element::_  , Element::Ru , Element::_  , Element::_  , Element::_  , Element::Sb , Element::Sc , Element::_  ,
		Element::Se , Element::_  , Element::Sg , Element::_  , Element::Si , Element::_  , Element::_  , Element::_  ,
		Element::Sm , Element::Sn , Element::_  , Element::O  , Element::_  , Element::Sr , Element::_  , Element::_  ,
		Element::_  , Element::_  , Element::_  , Element::Ta , Element::Tb , Element::Tc , Element::_  , Element::Te ,
		Element::_  , Element::_  , Element::Th , Element::Ti , Element::_  , Element::_  , Element::Tl , Element::Tm ,
		Element::_  , Element::_  , Element::P
};

inline Element stoe( const char symbol[] ) {
	auto l = strlen(symbol);
	unsigned int hash = symbol[0] * 23 + ( l > 1 ? symbol[1] : 204 ) + ( l > 2 ? symbol[2] : 0 ) * 2 - 58;
	return _str2elem_lut[ hash % 512 ];
}

inline Element stoe( std::string const &symbol ) {
	return stoe( symbol.c_str() );
}

inline const char* etos( Element e ) {
	switch(e) {
	case Element::H  : return "H";
	case Element::He : return "He";
	case Element::Li : return "Li";
	case Element::Be : return "Be";
	case Element::B  : return "B";
	case Element::C  : return "C";
	case Element::N  : return "N";
	case Element::O  : return "O";
	case Element::F  : return "F";
	case Element::Ne : return "Ne";
	case Element::Na : return "Na";
	case Element::Mg : return "Mg";
	case Element::Al : return "Al";
	case Element::Si : return "Si";
	case Element::P  : return "P";
	case Element::S  : return "S";
	case Element::Cl : return "Cl";
	case Element::Ar : return "Ar";
	case Element::K  : return "K";
	case Element::Ca : return "Ca";
	case Element::Sc : return "Sc";
	case Element::Ti : return "Ti";
	case Element::V  : return "V";
	case Element::Cr : return "Cr";
	case Element::Mn : return "Mn";
	case Element::Fe : return "Fe";
	case Element::Co : return "Co";
	case Element::Ni : return "Ni";
	case Element::Cu : return "Cu";
	case Element::Zn : return "Zn";
	case Element::Ga : return "Ga";
	case Element::Ge : return "Ge";
	case Element::As : return "As";
	case Element::Se : return "Se";
	case Element::Br : return "Br";
	case Element::Kr : return "Kr";
	case Element::Rb : return "Rb";
	case Element::Sr : return "Sr";
	case Element::Y  : return "Y";
	case Element::Zr : return "Zr";
	case Element::Nb : return "Nb";
	case Element::Mo : return "Mo";
	case Element::Tc : return "Tc";
	case Element::Ru : return "Ru";
	case Element::Rh : return "Rh";
	case Element::Pd : return "Pd";
	case Element::Ag : return "Ag";
	case Element::Cd : return "Cd";
	case Element::In : return "In";
	case Element::Sn : return "Sn";
	case Element::Sb : return "Sb";
	case Element::Te : return "Te";
	case Element::I  : return "I";
	case Element::Xe : return "Xe";
	case Element::Cs : return "Cs";
	case Element::Ba : return "Ba";
	case Element::La : return "La";
	case Element::Ce : return "Ce";
	case Element::Pr : return "Pr";
	case Element::Nd : return "Nd";
	case Element::Pm : return "Pm";
	case Element::Sm : return "Sm";
	case Element::Eu : return "Eu";
	case Element::Gd : return "Gd";
	case Element::Tb : return "Tb";
	case Element::Dy : return "Dy";
	case Element::Ho : return "Ho";
	case Element::Er : return "Er";
	case Element::Tm : return "Tm";
	case Element::Yb : return "Yb";
	case Element::Lu : return "Lu";
	case Element::Hf : return "Hf";
	case Element::Ta : return "Ta";
	case Element::W  : return "W";
	case Element::Re : return "Re";
	case Element::Os : return "Os";
	case Element::Ir : return "Ir";
	case Element::Pt : return "Pt";
	case Element::Au : return "Au";
	case Element::Hg : return "Hg";
	case Element::Tl : return "Tl";
	case Element::Pb : return "Pb";
	case Element::Bi : return "Bi";
	case Element::Po : return "Po";
	case Element::At : return "At";
	case Element::Rn : return "Rn";
	case Element::Fr : return "Fr";
	case Element::Ra : return "Ra";
	case Element::Ac : return "Ac";
	case Element::Th : return "Th";
	case Element::Pa : return "Pa";
	case Element::U  : return "U";
	case Element::Np : return "Np";
	case Element::Pu : return "Pu";
	case Element::Am : return "Am";
	case Element::Cm : return "Cm";
	case Element::Bk : return "Bk";
	case Element::Cf : return "Cf";
	case Element::Es : return "Es";
	case Element::Fm : return "Fm";
	case Element::Md : return "Md";
	case Element::No : return "No";
	case Element::Lr : return "Lr";
	case Element::Rf : return "Rf";
	case Element::Db : return "Db";
	case Element::Sg : return "Sg";
	case Element::Bh : return "Bh";
	case Element::Hs : return "Hs";
	case Element::Mt : return "Mt";
	case Element::Ds : return "Ds";
	case Element::Rg : return "Rg";
	case Element::Uub: return "Uub";
	case Element::Uut: return "Uut";
	case Element::Uuq: return "Uuq";
	case Element::Uup: return "Uup";
	case Element::Uuh: return "Uuh";
	case Element::Uus: return "Uus";
	case Element::Uuo: return "Uuo";
	default: return "Invalid";
	}
}

}

namespace std {

template<> struct hash<graphdot::Element> {
	inline std::size_t operator() ( graphdot::Element e ) const noexcept {
		return (std::size_t) e;
	}
};

}

#endif
