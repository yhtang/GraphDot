#ifndef _MM_H_
#define _MM_H_

#define MM_MAXLINE 1000

struct mmdata {

	int N, M, NNZ;
	int *x;
	int *y;
	double *v;

	int symmetricity;
	int binary;
	int ndiagonal;
	int realnnz;
};


/* @OGUZ-EDIT Begin */
typedef struct _coord_t
{
	int x, y;
	int mgvid;
} coord_t;
/* @OGUZ-EDIT End */

int initialize_mm(char *file, struct mmdata *mm);
void printmm(struct mmdata *mm, char *file);
void freemm(struct mmdata *mm);

#endif 
