/*-------------------------------------------------------
 		Written by Enver Kayaaslan 
		   Edited by Seher Acer
 Used for converting mmdata struct to pthdata struct
 --------------------------------------------------------*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <memory>
#include <vector>
#include <iostream>


#include "pth.h"


/* @OGUZ-EDIT Begin */
#define NO_CRD -1
/* @OGUZ-EDIT End */

// Converts mmdata to pthdata which is a col-net hypergraph
// Seher: Missing diagonals are inserted -- n_i always connects v_i
// Seher: Global  cell ids (gcids) are inserted 
void mm2pth_colnet(struct mmdata *mm, struct pthdata *pth) {

	pth -> c = mm -> N;
	pth -> n = mm -> M;
	pth -> p = mm -> realnnz;
	pth -> nconst = 1;
	pth -> nad = 0;
	
	pth -> xpins = (int *)calloc(pth->n+2, sizeof(int));
	pth -> nwghts = (int *)calloc(pth->n, sizeof(int));
	pth -> cwghts = (int *)calloc(pth->c, sizeof(int));
	pth -> gcids = (int *)calloc(pth->c, sizeof(int));
	
	int *diag = (int *)calloc(pth->c, sizeof(int));

	int i,j;
	for(i=0; i<mm->NNZ; i++) {

		if(mm->x[i] == mm->y[i])
			diag[mm->x[i]] = 1;

		pth->xpins[mm->y[i]+2]++;
		/* @OGUZ-EDIT-UNIT-CWGHTS commented out */
		/* pth->cwghts[mm->x[i]]++; */
	
		if(mm->symmetricity && mm->x[i] != mm->y[i])
		{	
			pth->xpins[mm->x[i]+2]++;
			/* @OGUZ-EDIT-UNIT-CWGHTS commented out */
			/* pth->cwghts[mm->y[i]]++; */
		}
	}
	
	for(i=0; i < pth->c; i++)
		if(diag[i] == 0)
		{	
			pth->xpins[i+2] ++;
			pth->p++;
			pth->nad++;
		}
	
	pth->pins = (int *)calloc(sizeof(int), pth->p);
	
	for(i=2; i<=pth->n; i++) 
		pth->xpins[i] += pth->xpins[i-1];

	for(i=0; i<mm->NNZ; i++) {

		pth->pins[pth->xpins[mm->y[i]+1]++] = mm->x[i];

		if(mm->symmetricity && mm->x[i] != mm->y[i]) {
			pth->pins[pth->xpins[mm->x[i]+1]++] = mm->y[i];
		}
	}

	for(i=0; i<pth->c; i++)
		if(diag[i] == 0)
			pth->pins[pth->xpins[i+1]++] = i;

	for(i=0; i<pth->n; i++) 	 	
		pth->nwghts[i] = 1;
	
	for(i=0; i<pth->c; i++)
		pth->gcids[i] = i;

	/* @OGUZ-EDIT-UNIT-CWGHTS begin */
	for (i = 0; i < pth->c; ++i)
		pth->cwghts[i] = 1;
	/* @OGUZ-EDIT-UNIT-CWGHTS end */

	pth->ne = pth->n;
	pth->pe = pth->p;

	free(diag);
}

// Converts mmdata to pthdata which is a row-net hypergraph
void mm2pth_rownet(struct mmdata *mm, struct pthdata *pth) {
	
	pth -> c = mm -> M;
	pth -> n = mm -> N;
	pth -> p = mm -> realnnz;
	pth -> nconst = 1;
	pth -> nad = 0;
	
	int *diag = (int *)calloc(pth->c, sizeof(int));
	pth -> xpins = (int *)calloc(pth->n+2, sizeof(int));
	int i,j;

	for(i=0; i<mm->NNZ; i++) {
	
		if(mm->x[i] == mm->y[i])
			diag[mm->x[i]] = 1;

		pth -> xpins[mm->x[i]+2] ++;
		if(mm->symmetricity && mm->x[i] != mm->y[i])
			pth -> xpins[mm->y[i]+2] ++;
	}

	for(i=0; i < pth->c; i++)
		if(diag[i] == 0)
		{	
			pth->xpins[i+2] ++;
			pth->p++;
			pth->nad++;
		}
	
	pth->pins = (int *)calloc(sizeof(int), pth->p);
	
	for(i=2; i<=pth->n; i++) 
		pth -> xpins[i] += pth -> xpins[i-1];

	for(i=0; i<mm->NNZ; i++) {

		pth -> pins[ pth->xpins[ mm->x[i]+1 ] ++ ] = mm->y[i];
		if(mm->symmetricity && mm->x[i] != mm->y[i]) {
			pth -> pins[ pth->xpins[mm->y[i]+1] ++ ] = mm->x[i];
		}
	}

	for(i=0; i<pth->c; i++)
		if(diag[i] == 0)
			pth->pins[pth->xpins[i+1]++] = i;

	free(diag);
}

// there are one vertex for each column, row and nonzero.
// there are one net for each column and row
// each net connects its corresponding column/row and the nonzeros contained in it
// vertex numbering: columns - rows - nonzeros
// net bumbering: columns - row
// nonzero numbering:
// 	for nonsymmetric matrices: mm struct ordering
//	for symmetric matrices: mm struct ordering - added nonzeros (mm->NNZ..mm->realnnz-1 are added ones)
void mm2pth_finegrain(struct mmdata *mm, struct pthdata *pth, coord_t **vcs) {

	int i;
	pth -> c = mm -> realnnz + mm -> N + mm -> M;
	pth -> n = mm -> N + mm -> M;
	pth -> p = mm -> realnnz * 2 + mm -> N + mm -> M;
	pth -> nconst = 1;

	pth -> xpins = (int *)calloc(pth->n+2, sizeof(int));
	pth -> pins = (int *)malloc(sizeof(int) * pth -> p);
	pth -> cwghts = (int *)calloc(pth->c, sizeof(int));
	pth -> nwghts = (int *)calloc(pth->n, sizeof(int));
	pth -> gcids = (int *)calloc(pth->c, sizeof(int));
	/* @OGUZ-EDIT Begin */
	*vcs = (coord_t *)malloc(sizeof(**vcs) * pth->c);
	coord_t *vcs_ptr = *vcs;
	for (i = 0; i < pth->c; ++i)
		 vcs_ptr[i].x = vcs_ptr[i].y = NO_CRD;
	/* @OGUZ-EDIT End */

	for(i=pth->n; i<pth->c; pth->cwghts[i++] = 1);
	for(i=0; i<pth->n; pth->nwghts[i++] = 1);

	for(i=0; i<pth->n; i++)
		pth -> xpins[i + 2] ++;

	for(i=0; i<mm->NNZ; i++) {

		pth -> xpins[mm -> y[i] + 2] ++;
		pth -> xpins[mm -> M + mm -> x[i] + 2] ++;
		
		if(mm -> symmetricity && mm -> x[i] != mm -> y[i])
		{
			pth -> xpins[mm -> x[i] + 2] ++;
			pth -> xpins[mm -> M + mm -> y[i] + 2] ++;
		}
	} 

	for(i=2; i<=pth->n+1; i++) 
		pth -> xpins[i] += pth -> xpins[i-1];

	for(i=0; i<pth->n; i++)	
		pth -> pins[ pth->xpins[i+1] ++] = i;

	int sym = 0;
	for(i=0; i<mm->NNZ; i++) {

		pth -> pins[ pth->xpins[ mm->y[i] + 1 ] ++ ] = mm->M + mm->N + i;
		pth -> pins[ pth->xpins[ mm->M + mm->x[i] + 1 ] ++ ] = mm->M + mm->N + i;

		/* @OGUZ-EDIT Begin */
		vcs_ptr[mm->M + mm->N + i].x = mm->x[i];
		vcs_ptr[mm->M + mm->N + i].y = mm->y[i];
		/* @OGUZ-EDIT End */
	
		if(mm -> symmetricity && mm -> x[i] != mm -> y[i])
		{
			pth -> pins[ pth->xpins[ mm->x[i] + 1 ] ++ ] = mm->M + mm->N + mm -> NNZ + sym;
			pth -> pins[ pth->xpins[ mm->M + mm->y[i] + 1 ] ++ ] = mm->M + mm->N + mm -> NNZ + sym;
			/* @OGUZ-EDIT Begin */
			vcs_ptr[mm->M + mm->N + mm->NNZ + sym].x = mm->y[i];
			vcs_ptr[mm->M + mm->N + mm->NNZ + sym].y = mm->x[i];
			/* @OGUZ-EDIT End */
			sym++;

		}
	}
	
	for(i = 0; i < pth->c; i++)
		pth->gcids[i] = i;

	pth->pe = pth->p;
	pth->ne = pth->n;
}

int analyse(struct pthdata *pth)
{
	int i;
	printf("There are %d nets, %d cells, %d pins..\n", pth->n, pth->c, pth->p);

	int sp = 0, tp = 0;
	for(i = 0; i < pth->n; i++)
		if(pth->xpins[i+1]-pth->xpins[i] == 1)
			sp++;
		else if(pth->xpins[i+1]-pth->xpins[i] == 2)
			tp++;
	printf("sp: %d, tp: %d\n", sp, tp);
	 
	return 0;
}

int pth2robpth(struct pthdata *pth, struct pthdata *robpth, int nrows, int ncols, int *xadj, int *adjncy, int *map, coord_t *vcs)
{
	int i, j, k;
	int dim = nrows + ncols;
	for(i = dim; i < pth->c; i++)
	{
		int col = adjncy[xadj[i-dim]];
		int row = adjncy[xadj[i-dim]+1];
		if((col < ncols && row < ncols) || (col >= ncols && row >= ncols))
			return 1;

		if(col >= ncols && row < ncols)
		{
			printf("[Warning] Swapped %d %d\n", row, col);
			int tmp = col;
			col = row;
			row = tmp;
		}

		int coldeg = pth->xpins[col+1]-pth->xpins[col];
		int rowdeg = pth->xpins[row+1]-pth->xpins[row];

		if(coldeg == 1)
			map[i] = row;
		else if(rowdeg == 1)
			map[i] = col;
		else if(rowdeg < coldeg)
			map[i] = row;
		else if(coldeg < rowdeg)
			map[i] = col;
		else
		{
			if(nrows > ncols)
				map[i] = row;
			else
				map[i] = col;
		}

		/* @OGUZ-EDIT Begin */
		vcs[i].mgvid = map[i];
		/* @OGUZ-EDIT End */
	}
	
	robpth->c = nrows+ncols;
	robpth->n = 0;
	robpth->p = 0;
	robpth->nconst = 1;

	robpth->cwghts = (int *)calloc(robpth->c, sizeof(int));
	robpth->gcids = (int *)calloc(robpth->c, sizeof(int));
	robpth->nwghts = (int *)calloc(pth->n, sizeof(int));
	robpth->xpins = (int *)calloc(pth->n+2, sizeof(int));
	robpth->pins = (int *)calloc(pth->p, sizeof(int));	
 
	robpth->xpins[0] = 0;
	for(i = 0; i < pth->n; i++)
	{		
		for(j = pth->xpins[i]; j < pth->xpins[i+1]; j++)
		{
			int v = pth->pins[j];
			if(v >= nrows+ncols)
				v = map[v];
			int exists = 0;
			for(k = robpth->xpins[robpth->n]; k < robpth->p && exists == 0; k++)
				if(robpth->pins[k] == v)
					exists = 1;
				
			if(exists == 0)
				robpth->pins[robpth->p++] = v;
		}
		robpth->n++;
		robpth->xpins[robpth->n] = robpth->p;

	}	

	for(i = 0; i < robpth->n; i++)
		robpth->nwghts[i] = 1;

	for(i = 0; i < pth->c; i++)
		if(i < nrows+ncols)
			robpth->cwghts[i] = 1;
		else
			robpth->cwghts[map[i]] ++;

	for(i = 0; i < robpth->c; i++)
		robpth->gcids[i] = i;

	robpth->pe = robpth->p;
	robpth->ne = robpth->n;

	return 0;
}

/* @OGUZ-COMMENT Do NOT use */
int pthpartition(struct pthdata *pth, int kway, double imbalance, int *partvec, int crs, int unified) {

	int cutsize=-1;
	int *partweights = (int *)calloc(sizeof(int), kway);

	/* PaToH_Parameters args; */
	/* PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_DEFAULT);		 */

	/* args._k = kway;  */
	/* args.final_imbal = imbalance; */
	/* args.crs_alg = crs; */

	/* @OGUZ-EDIT Begin */
    /* args.MemMul_CellNet = args.MemMul_CellNet * 4; */
    /* args.MemMul_Pins = args.MemMul_Pins * 2; */
    /* args.MemMul_General = args.MemMul_General * 2; */
	/* args.ref_alg = PATOH_REFALG_T_BFM; /\* BFM with tight balance 2016.09.17 *\/ */
    /* if (pth -> nconst > 1) */
    /*     args.final_imbal = 0.20f; */
    /* args.crs_alg = PATOH_CRS_HCM; */
    /* args.crs_useafteralg = PATOH_CRS_HCM; */
    /* @OGUZ-EDIT End */

	if(unified > 0 && pth->nconst == 2)
	{
		int i;
		int *weights = (int *)calloc(pth->c, sizeof(int));
		for(i = 0; i < pth->c; i++)
			weights[i] = pth->cwghts[2*i]+ pth->cwghts[2*i+1]*unified;

		/* PaToH_Alloc(&args, pth->c, pth->ne, 1, weights, pth->nwghts, pth->xpins, pth->pins); */

		/* PaToH_Part(&args, pth->c, pth->ne, 1, 0, weights, pth->nwghts, pth->xpins, pth->pins, NULL */
		/*      , partvec, partweights, &cutsize); */

		free(weights);
	}
	else
	{
		/* PaToH_Alloc(&args, pth->c, pth->ne, pth->nconst, pth->cwghts, pth->nwghts, pth->xpins, pth->pins); */

		/* PaToH_Part(&args, pth->c, pth->ne, pth->nconst, 0, pth->cwghts, pth->nwghts, pth->xpins, pth->pins, NULL */
		/*      , partvec, partweights, &cutsize); */
	}

	/* PaToH_Free(); */
	free(partweights);

	return cutsize;
}


/* @OGUZ-NOTE Disregards imbalance */
int
pthpartition_gen (
	struct pthdata *pth,
	int kway,
	double imbalance,
	int *partvec,
	int crs,
	int unified
	)
{
	assert (kway == 2);
	float tpwghts[2];
	if (pth->curk % 2 == 0)
		tpwghts[0] = tpwghts[1] = 0.5;
	else
	{
		int tmp = pth->curk / 2;
		tpwghts[0] = (float) tmp / (float) pth->curk;
		tpwghts[1] = (float) (tmp+1) / (float) pth->curk;
	}

	/* fprintf(stdout, "tpwghts %.2f %.2f", tpwghts[0], tpwghts[1]); */

	int cutsize=-1;
	int *partweights = (int *)calloc(sizeof(int), kway);

	/* PaToH_Parameters args; */
	/* PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_DEFAULT);		 */

	/* args._k = kway;  */
	/* args.final_imbal = imbalance; */
	/* args.crs_alg = crs; */

	/* @OGUZ-EDIT Begin */
    /* args.MemMul_CellNet = args.MemMul_CellNet * 4; */
    /* args.MemMul_Pins = args.MemMul_Pins * 2; */
    /* args.MemMul_General = args.MemMul_General * 2; */
	/* args.ref_alg = PATOH_REFALG_T_BFM; /\* BFM with tight balance 2016.09.17 *\/ */
    /* if (pth -> nconst > 1) */
    /*     args.final_imbal = 0.20f; */
    /* args.crs_alg = PATOH_CRS_HCM; */
    /* args.crs_useafteralg = PATOH_CRS_HCM; */
    /* @OGUZ-EDIT End */

	if(unified > 0 && pth->nconst == 2)
	{
		int i;
		int *weights = (int *)calloc(pth->c, sizeof(int));
		for(i = 0; i < pth->c; i++)
			weights[i] = pth->cwghts[2*i]+ pth->cwghts[2*i+1]*unified;

		/* PaToH_Alloc(&args, pth->c, pth->ne, 1, weights, pth->nwghts, pth->xpins, pth->pins); */

		/* PaToH_Part(&args, pth->c, pth->ne, 1, 0, weights, pth->nwghts, pth->xpins, pth->pins, NULL */
		/*      , partvec, partweights, &cutsize); */

		free(weights);
	}
	else
	{
		/* PaToH_Alloc(&args, pth->c, pth->ne, pth->nconst, pth->cwghts, pth->nwghts, pth->xpins, pth->pins); */

		/* PaToH_Part(&args, pth->c, pth->ne, pth->nconst, 0, pth->cwghts, */
		/* 		   pth->nwghts, pth->xpins, pth->pins, tpwghts, */
		/* 		   partvec, partweights, &cutsize); */
	}

	/* PaToH_Free(); */
	free(partweights);

	return cutsize;
}



void
ensure_pw (
	struct pthdata	*pth,
	int				*pvec,
	int				*partweights,
	float			*tpwghts
	)
{
	int v, n, i;
	int tpw0 = (int) tpwghts[0], tpw1 = (int) tpwghts[1];
	int from = 0, to = 1;
	if (partweights[1] > tpw1)
	{
		from = 1;
		to	 = 0;
	}

	int x = partweights[from] - tpwghts[from];

	/* fprintf(stdout, "will move %d vertices\n", x); */
	int *netinfo = (int *)calloc(2*(pth->n),  sizeof(*netinfo));
	int *vnets	 = (int *)malloc(sizeof(*vnets) * pth->xpins[pth->n]);
	int *xvnets	 = (int *)calloc(pth->c + 2, sizeof(*xvnets));
		

	// compute net information and nets of vertices
	for (n = 0; n < pth->n; ++n)
	{
		for (i = pth->xpins[n]; i < pth->xpins[n+1]; ++i)
		{
			v					  = pth->pins[i];
			netinfo[2*n+pvec[v]] += 1;
			xvnets[v+2]			 += 1;
		}
	}

	for (v = 1; v <= pth->c; ++v)
		xvnets[v] += xvnets[v-1];

	for (n = 0; n < pth->n; ++n)
	{
		for (i = pth->xpins[n]; i < pth->xpins[n+1]; ++i)
		{
			v					= pth->pins[i];
			vnets[xvnets[v+1]]  = n;
			xvnets[v+1]		   += 1;
		}
	}

	while (x > 0)
	{
		int maxg = INT_MIN, selv = -1;
		for (v = 0; v < pth->c; ++v)
		{
			if (pvec[v] == from)
			{
				int g = 0;
				for (i = xvnets[v]; i < xvnets[v+1]; ++i)
				{
					n = vnets[i];
					if (netinfo[2*n + from] == 1 && netinfo[2*n + to] > 0)
						g += pth->nwghts[n];
					else if (netinfo[2*n + from] > 1 && netinfo[2*n + to] == 0)
						g -= pth->nwghts[n];
				}

				if (g > maxg)
				{
					maxg = g;
					selv = v;
				}
			}
		}

		/* fprintf(stdout, "maxg = %d, v = %d\n", maxg, selv); */

		pvec[selv] = to;
		for (i = xvnets[selv]; i < xvnets[selv+1]; ++i)
		{
			n = vnets[i];
			netinfo[2*n + from] -= 1;
			netinfo[2*n + to] += 1;
		}

		--x;
	}

	free(netinfo);
	free(vnets);
	free(xvnets);
}




/* @OGUZ-NOTE Disregards imbalance */
int
pthpartition_gen_tile (
	struct pthdata *pth,
	int kway,
	double imbalance,
	int *partvec,
	int crs,
	int unified,
	int tilesize
	)
{
	assert (kway == 2);
	float tpwghts[2];

	if (pth->c % tilesize != 0)		/* left-over rightmost partition */
	{
		int tmp = pth->curk % 2 == 0 ? pth->curk / 2 : pth->curk / 2 + 1;
		tpwghts[0] = tmp * tilesize;
		tpwghts[1] = pth->c - tpwghts[0];
	}
	else
	{
		if (pth->curk % 2 == 0)
			tpwghts[0] = tpwghts[1] = (pth->curk / 2) * tilesize;
		else
		{
			int tmp		= pth->curk / 2;
			tpwghts[0]	= (tmp+1) * tilesize;
			tpwghts[1]	= tmp * tilesize;
		}
	}
	

	/* fprintf(stdout, "tpwghts %.0f %.0f\n", tpwghts[0], tpwghts[1]); */

	int cutsize=-1;
	int *partweights = (int *)calloc(sizeof(int), kway);

	/* PaToH_Parameters args; */
	/* PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_DEFAULT);		 */

	/* args._k = kway;  */
	/* args.final_imbal = imbalance; */
	/* args.crs_alg = crs; */

	/* @OGUZ-EDIT Begin */
    /* args.MemMul_CellNet = args.MemMul_CellNet * 4; */
    /* args.MemMul_Pins = args.MemMul_Pins * 2; */
    /* args.MemMul_General = args.MemMul_General * 2; */
	/* args.ref_alg = PATOH_REFALG_T_FM; */
    /* if (pth -> nconst > 1) */
    /*     args.final_imbal = 0.20f; */
    /* args.crs_alg = PATOH_CRS_HCM; */
    /* args.crs_useafteralg = PATOH_CRS_HCM; */
    /* @OGUZ-EDIT End */

	if(unified > 0 && pth->nconst == 2)
	{
		int i;
		int *weights = (int *)calloc(pth->c, sizeof(int));
		for(i = 0; i < pth->c; i++)
			weights[i] = pth->cwghts[2*i]+ pth->cwghts[2*i+1]*unified;

		/* PaToH_Alloc(&args, pth->c, pth->ne, 1, weights, pth->nwghts, pth->xpins, pth->pins); */

		/* PaToH_Part(&args, pth->c, pth->ne, 1, 0, weights, pth->nwghts, pth->xpins, pth->pins, NULL */
		/*      , partvec, partweights, &cutsize); */

		free(weights);
	}
	else
	{
		/* PaToH_Alloc(&args, pth->c, pth->ne, pth->nconst, pth->cwghts, pth->nwghts, pth->xpins, pth->pins); */

		int i = 0;
		do
		{
			/* fprintf(stdout, "try %d\n", i); */
			/* args.seed = i*7; */
			/* PaToH_Part(&args, pth->c, pth->ne, pth->nconst, 0, pth->cwghts, */
			/* 		   pth->nwghts, pth->xpins, pth->pins, tpwghts, */
			/* 		   partvec, partweights, &cutsize); */
			/* fprintf(stdout, "pw0 %d pw1 %d\n", partweights[0], partweights[1]); */
		}
		while (i++ < 10 &&
			   (partweights[0] != (int) tpwghts[0] ||
				partweights[1] != (int) tpwghts[1]));
		
	}

	/* fprintf(stdout, "obtained part weights %d %d\n", partweights[0], partweights[1]); */
	if (partweights[0] != (int) tpwghts[0] ||
		partweights[1] != (int) tpwghts[1])
	{
		/* fprintf(stdout, "part weights are not ensured, falling back " */
		/* 		"post-partitioning moves.\n"); */
		ensure_pw(pth, partvec, partweights, tpwghts);
		partweights[0] = partweights[1] = 0;
		int i;
		for (i = 0; i < pth->c; ++i)
			partweights[partvec[i]] += 1;

		if (partweights[0] != (int) tpwghts[0] ||
			partweights[1] != (int) tpwghts[1])
		{
			fprintf(stdout, "system error. mission abort.\n");
			exit(1);
		}
	}

	/* PaToH_Free(); */
	free(partweights);

	return cutsize;
}


/* @OGUZ-NOTE Disregards imbalance */
int
kahyparpartition_gen_tile (
	struct pthdata *pth,
	int kway,
	double imbalance,
	int *partvec,
	int crs,
	int unified,
	int tilesize,
	kahypar_context_t *context
	)
{
	assert (kway == 2);
	float tpwghts[2];

	if (pth->c % tilesize != 0)		/* left-over rightmost partition */
	{
		int tmp = pth->curk % 2 == 0 ? pth->curk / 2 : pth->curk / 2 + 1;
		tpwghts[0] = tmp * tilesize;
		tpwghts[1] = pth->c - tpwghts[0];
	}
	else
	{
		if (pth->curk % 2 == 0)
			tpwghts[0] = tpwghts[1] = (pth->curk / 2) * tilesize;
		else
		{
			int tmp		= pth->curk / 2;
			tpwghts[0]	= (tmp+1) * tilesize;
			tpwghts[1]	= tmp * tilesize;
		}
	}

	const kahypar_hypernode_id_t	ncells = pth->c;
	const kahypar_hyperedge_id_t	nnets  = pth->ne;
	const kahypar_partition_id_t	k	   = kway;
	const double					imb	   = 0.00;	// unused
	
	/* fprintf(stdout, "tpwghts %.0f %.0f\n", tpwghts[0], tpwghts[1]); */

	int *partweights = (int *)calloc(sizeof(int), k);

	// set block weights
	std::unique_ptr<kahypar_hypernode_weight_t[]> block_weights =
		std::make_unique<kahypar_hypernode_weight_t[]>(k);
	block_weights[0] = (int)tpwghts[0];
	block_weights[1] = (int)tpwghts[1];
	kahypar_set_custom_target_block_weights(k, block_weights.get(), context);

	kahypar_hyperedge_weight_t cutsize = 0;
	// std::vector<kahypar_partition_id_t> partvec(ncells, -1);

	std::unique_ptr<size_t[]> hyperedge_indices =
		std::make_unique<size_t[]>(pth->ne+1);
	for (int i = 0; i < pth->ne + 1; ++i)
		hyperedge_indices[i] = pth->xpins[i];

	std::unique_ptr<kahypar_hyperedge_id_t[]> hyperedges =
		std::make_unique<kahypar_hyperedge_id_t[]>(pth->xpins[pth->ne]);
	for (int i = 0; i < pth->xpins[pth->ne]; ++i)
		hyperedges[i] = pth->pins[i];

	std::vector<kahypar_partition_id_t> partition(ncells, -1);
		

	if(unified > 0 && pth->nconst == 2)
	{
		int i;
		int *weights = (int *)calloc(pth->c, sizeof(int));
		for(i = 0; i < pth->c; i++)
			weights[i] = pth->cwghts[2*i]+ pth->cwghts[2*i+1]*unified;

		/* PaToH_Alloc(&args, pth->c, pth->ne, 1, weights, pth->nwghts, pth->xpins, pth->pins); */

		/* PaToH_Part(&args, pth->c, pth->ne, 1, 0, weights, pth->nwghts, pth->xpins, pth->pins, NULL */
		/*      , partvec, partweights, &cutsize); */

		free(weights);
	}
	else
	{
		int i = 0;
		do
		{
			partweights[0] = partweights[1] = 0;
			/* fprintf(stdout, "try %d\n", i); */
			/* args.seed = i*7; */
			kahypar_partition(ncells, nnets, imbalance, k,
							  nullptr, // vertex weights
							  pth->nwghts, hyperedge_indices.get(),
							  hyperedges.get(), &cutsize, context, partition.data());
			/* PaToH_Part(&args, pth->c, pth->ne, pth->nconst, 0, pth->cwghts, */
			/* 		   pth->nwghts, pth->xpins, pth->pins, tpwghts, */
			/* 		   partvec, partweights, &cutsize); */

			for (int v = 0; v < ncells; ++v)
			{
				partvec[v] = partition[v];
				partweights[partvec[v]] += 1;
			}
			/* fprintf(stdout, "pw0 %d pw1 %d\n", partweights[0], partweights[1]); */
		}
		while (i++ < 10 &&
			   (partweights[0] != (int) tpwghts[0] ||
				partweights[1] != (int) tpwghts[1]));
		
	}

	/* fprintf(stdout, "obtained part weights %d %d cut %d\n", */
	/* 		partweights[0], partweights[1], cutsize); */
	if (partweights[0] != (int) tpwghts[0] ||
		partweights[1] != (int) tpwghts[1])
	{
		fprintf(stdout, "part weights are not ensured, falling back "
				"post-partitioning moves.\n");
		ensure_pw(pth, partvec, partweights, tpwghts);
		partweights[0] = partweights[1] = 0;
		int i;
		for (i = 0; i < pth->c; ++i)
			partweights[partvec[i]] += 1;

		if (partweights[0] != (int) tpwghts[0] ||
			partweights[1] != (int) tpwghts[1])
		{
			fprintf(stdout, "system error. mission abort.\n");
			exit(1);
		}
	}

	/* PaToH_Free(); */
	free(partweights);

	return cutsize;
}



void printpth(struct pthdata *pth) {

	printf("number of cells: %d\n", pth->c);
	printf("number of nets: %d\n", pth->n);
	printf("number of pins: %d\n", pth->p);

	int i;
	printf("xpins: ");
	for(i=0; i<=pth->n; i++)
		printf("%d ", pth->xpins[i]);
	printf("\n");

	printf("pins: ");
	for(i=0; i<pth->p; i++)
		printf("%d ", pth->pins[i]);
	printf("\n");
}

void freepth(struct pthdata *pth) {

	free(pth->cwghts);
	free(pth->nwghts);
	free(pth->xpins);
	free(pth->pins);
	free(pth->gcids);

	free(pth);
}


void
write_hygr (
	struct pthdata *pth,
	char *fname
	)
{
	int n, v;
	FILE *fp = fopen(fname, "w");	
	
	fprintf(fp, "%d %d %d %d %d\n", 0, pth->c, pth->n, pth->p, 3);
	fprintf(fp, "%% nets\n");
	for (n = 0; n < pth->n; ++n)
	{
		fprintf(fp, "%d ", pth->nwghts[n]);
		for (v = pth->xpins[n]; v < pth->xpins[n+1]; ++v)
			fprintf(fp, "%d ", pth->pins[v]);
		fprintf(fp, "\n");
	}
	fprintf(fp, "%% vertex weights\n");
	for (v = 0; v < pth->c; ++v)
		fprintf(fp, "%d\n", pth->cwghts[v]);

	fclose(fp);

	return;
}

void
write_vinfo (
	int	nnzcount,
	coord_t *vcs,
	char *fname
	)
{
	int v;
	FILE *fp = fopen(fname, "w");

	for (v = 0; v < nnzcount; ++v)
		fprintf(fp, "%d %d %d\n", vcs[v].x, vcs[v].y, vcs[v].mgvid);

	fclose(fp);

	return;
}
