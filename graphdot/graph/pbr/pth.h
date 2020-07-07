#ifndef _PTH_H_ 
#define _PTH_H_

#include "mm.h"
#include "libkahypar.h"

#define PTH_TYPE_ROWNET 0
#define PTH_TYPE_COLNET 1

#define PTH_NETSTATE_LOCAL	0
#define PTH_NETSTATE_NONLOCAL	1
#define PTH_NETSTATE_SEND	2
#define PTH_NETSTATE_RECV	3

struct pthdata {

	int c;		//number of cells
	int n;		//number of nets
	int p;		//number of pins;
	int nconst;	//number of constraints
	int nad;	//number of added diagonals

	int *cwghts;	//cell weights
	int *nwghts;	//net weights
	int *xpins;	//cumulative number of pins up to each net
	int *pins;	//cells of pins

	int *gcids;	//global cell ids


	int ne;		//number of nets together with super-nets
	int pe;		//number of pins together with pins of super-nets

	int curk;

};
int analyse(struct pthdata *pth);
int pthpartition(struct pthdata *pth, int kway, double imbalance, int *partvec, int crs, int unified);
int pthpartition_gen(struct pthdata *pth, int kway, double imbalance, int *partvec, int crs, int unified);
int pthpartition_gen_tile (struct pthdata *pth, int kway, double imbalance, int *partvec, int crs,
						   int unified, int tilesize);
int kahyparpartition_gen_tile (struct pthdata *pth, int kway, double imbalance, int *partvec, int crs,
						   int unified, int tilesize, kahypar_context_t *context);
void mm2pth_colnet(struct mmdata *mm, struct pthdata *pth);
void mm2pth_rownet(struct mmdata *mm, struct pthdata *pth);
void mm2pth_finegrain(struct mmdata *mm, struct pthdata *pth, coord_t **vcs);
int pth2robpth(struct pthdata *pth, struct pthdata *robpth, int nrows, int ncols, int *xadj, int *adjncy, int *map, coord_t *vcs);
void printpth(struct pthdata *pth);
void freepth(struct pthdata *pth);

/* @OGUZ-EDIT Begin */
void
write_hygr (
	struct pthdata *pth,
	char *fname
	);
void
write_vinfo (
	int	nnzcount,
	coord_t *vcs,
	char *fname
	);
/* @OGUZ-EDIT End */

#endif
