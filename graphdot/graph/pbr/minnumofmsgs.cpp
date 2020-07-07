#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>

#include <vector>
using std::vector;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "mm.h"
#include "pth.h"

#define BFS_ORDER 0
#define DFS_ORDER 1

int kway;
double imbalance;
int cost;
int N;
int mnom;
int order;
int mb;
int nor;
int threshold;
int adaptive;
int delayed;
int unified;
int relaxed;
int ncoresinanode;
int selective;
int tilesize;

char matrixfile[1024];
char matrixname[1024];
char partfile[1024];

int *gpartvec;	//global partitioning vector - size: N

struct pthdata **curlevel;
struct pthdata **nextlevel;

int totalcut;

void print_params ( void )
{
	fprintf(stdout,
			"matrix: %s ### "
			"k = %d ### "
			"i = %.2f ### "
			"c = %d ### "
			"m = %d ### "
			"o = %s ### "
			"e = %d ### "
			"a = %d ### "
			"d = %d ### "
			"r = %d ### "
			"b = %d ### "
			"n = %d ### "
			"u = %d ### "
			"l = %d ### "
			"s = %d ###"
			"t = %d ###\n",
			matrixname,
			kway,
			imbalance,
			cost,
			mnom,
			order == 0 ? "BFS" : "DFS",
			threshold,
			adaptive,
			delayed,
			relaxed,
			mb,
			nor,
			unified,
			ncoresinanode,
			selective,
			tilesize);
	fflush(stdout);
}


void printusage()
{
	printf("usage: mnom [options] matrixfile\n");
	printf("\t-k kway: (int) number of parts [4]\n");
	printf("\t-i imbalance: (double) desired imbalance [0.10]\n");
	printf("\t-m mnom: (int) nonzero if minimization of number of messages is active [0]\n");
	printf("\t-c cost: (int) cost of super-nets [10]\n");
	printf("\t-o order: (int) order of RB process [0]\n");
	printf("\t-e threshold: (int) eliminates super-nets whose pin-size are larger than this value [-1]\n");
	printf("\t-a adaptive: (int) nonzero if adaptive (linear) super-net costing is active [0]\n");
	printf("\t-d delayed: (int) the K value where you want to begin adding super-nets [1]\n");
	printf("\t-r relaxed: (int) the K value where you want to begin balancing messages [1]\n");
	printf("\t-b mb: (int) nonzero if message balancing is active [0]\n");
	printf("\t-n nor: (int) nonzero if second weights' avg = first weights' avg [0]\n");
	printf("\t-u unified: (int) unified balancing is active if nonzero and the value is used as alpha coefficient [0]\n");
	printf("\t-l ncoresinanode: (int) number of cores in a node, where message nets won't be added for intra-node cores if selective is on [16]\n");
	printf("\t-s selective: (int) nonzero if selective addition of message nets is on (aka don't add intra node message nets) [0]\n");
	printf("\t-p partfile: (char*) partitionfile [partition.out]\n");
	printf("\t-t tilesize: (int) k is computed from this\n");

	exit(1);
}

void getdefaults()
{
	kway = 4;
	imbalance = 0.10;
	cost = 10;
	mnom = 0;
	order = BFS_ORDER;
	threshold = -1;
	adaptive = 0;
	delayed = 1;
	relaxed = 1;
	mb = 0;
	nor = 0;
	unified = 0;
	ncoresinanode = 16;
	selective = 0;
	tilesize = 0;
	strcpy(partfile, "partition.out");
}

void parseinputs(int argc, char *argv[])
{
	int c;
	while ((c = getopt(argc, argv, "k:i:m:c:o:e:a:d:r:b:n:u:l:p:s:t:")) != -1)
	{
		switch (c)
		{
			case 'k': 	kway = atoi(optarg);
			  		break;
			case 'i':	imbalance = atof(optarg);
					break;
			case 'm':	mnom = atof(optarg);
					break;
			case 'c':	cost = atof(optarg);
					break;
			case 'o':	order = atoi(optarg);
					break;		
			case 'e':	threshold = atoi(optarg);
					break;
			case 'a':	adaptive = atoi(optarg);
					break;
			case 'd':	delayed = atoi(optarg);
					break;
			case 'r':	relaxed = atoi(optarg);
					break;
			case 'b':	mb = atoi(optarg);
					break;	
			case 'n':	nor = atoi(optarg);
					break;	
			case 'u':	unified = atoi(optarg);
					break;	
			case 'l':	ncoresinanode = atoi(optarg);
					break;
			case 's':	selective = atoi(optarg);
					break;
			case 'p':	strncpy(partfile, optarg, 1024);
				break;
			case 't':	tilesize = atoi(optarg);
				break;
		}
	}

	if (argc <= optind)
		printusage();

	strcpy(matrixfile, argv[optind]);
}

void initialize_parameters()
{
	getdefaults();
	// parseinputs(argc, argv);
}

// substrings name of matrix from file path
void substring(char *text, char out[200])
{

   	char *ptr = text;
  
   	char *prevptr = NULL;

   	while( (ptr = strstr(ptr,"/")))
   	{
        	prevptr = ptr++;
   	}
	prevptr++;

    	strncpy(out, prevptr, strlen(prevptr)-4);

}

//finds the connected components of the subhypergraph in which only super-nets exist.
//returns this imformation in compvec array of size pth->c.
//compvec[i] >= 0 is the id of the component that v_i belongs to. not a boundary vertex otherwise (-1).
//ncomp denotes the number of connected components found

//int findcomponents(struct pthdata *pth, int *compvec, int *ncomp)
int findcomponents(struct pthdata *pth, int *ncomp)
{
	//convert pins-of-nets structure to nets-of-vertices structure for super nets and their pins (boundary vertices)		
	int i, j;
	
	int nb = 0;
	int *bmap = (int *)calloc(pth->c, sizeof(int));
	int *bvers = (int *)calloc(pth->c, sizeof(int));
 
	for(i = 0; i < pth->c; i++)
		bmap[i] = -1;

	for(i = pth->n; i < pth->ne; i++)
	{
		for(j = pth->xpins[i]; j < pth->xpins[i+1]; j++)
		{
			int v = pth->pins[j];
			if(bmap[v] == -1)
			{
				bmap[v] = nb;
				bvers[nb++] = v;
			}
		}
	}	

	int *xnets = (int *)calloc(nb+2, sizeof(int));
	for(i = pth->n; i < pth->ne; i++)
	{
		for(j = pth->xpins[i]; j < pth->xpins[i+1]; j++)
		{
			int v = pth->pins[j];
			xnets[bmap[v]+2]++;
		}
	}	

	for(i = 1; i <= nb; i++)
		xnets[i+1] += xnets[i];
	
	int *nets = (int *)calloc(xnets[nb+1], sizeof(int));
	for(i = pth->n; i < pth->ne; i++)
	{
		for(j = pth->xpins[i]; j < pth->xpins[i+1]; j++)
		{
			int v = pth->pins[j];
			nets[xnets[bmap[v]+1]++] = i;
		}
	}

	//now find components
	int nc = 0, lastsrc = 0;
	
	int *comp = (int *)calloc(nb, sizeof(int));
	for(i = 0; i < nb; i++)
		comp[i] = -1;
	
	int *queue = (int *)calloc(nb+2, sizeof(int));
	int qhead = 0, qtail = 0;

	while(lastsrc != nb)
	{
		queue[qtail++] = bvers[lastsrc];
		comp[lastsrc] = nc;

		while(qhead != qtail)
		{
			int curv = queue[qhead++];
			int mapcurv = bmap[curv];

			//find nets of curv
			for(i = xnets[mapcurv]; i < xnets[mapcurv+1]; i++)
			{
				int curn = nets[i];
				//find pins of curn
				for(j = pth->xpins[curn]; j < pth->xpins[curn+1]; j++)
				{
					int newv = pth->pins[j];
					if(comp[bmap[newv]] == -1)
					{
						comp[bmap[newv]] = nc;
						queue[qtail++] = newv;
					}
				}
			}
		}

		while(comp[lastsrc] != -1 && lastsrc < nb)
			lastsrc++;

		nc++;
	}

	*ncomp = nc;
	
	/*for(i = 0; i < pth->c; i++)
		if(bmap[i] == -1)
			compvec[i] = -1;
		else
		{
			compvec[i] = comp[bmap[i]];	
		}
	*/

	free(comp);
	free(queue);
	free(nets);
	free(xnets);
	free(bmap);
	free(bvers);	
	
	return 0;	
}

// super-nets added to me just before my bisection
// they connect just source vertices that i have
// I iterate over all other pths to find nonlocal nets whose source vertex belongs to me 
int mnom_addsendnets(struct pthdata *curpth, struct pthdata *pth, struct pthdata *rpth, int curpid, int curnpths, double cost, double *max, int threshold, int ncoresinanode, int kway, int selective)
{

	int i, j, lb, ub;
	//iterate over local vertices: 1...curpth->c 
	int nsendnets = 0;
	int *sendnetids = (int *)calloc(curnpths, sizeof(int));
	int *sendnetsizes = (int *)calloc(curnpths, sizeof(int));
	int *sends = (int *)calloc(curnpths, sizeof(int));
	int totalpin = 0;

	lb = curpid;
	ub = curpid+1;

	/* printf("    in %s, lb %d ub %d\n", __FUNCTION__, lb, ub); */

	if(selective != 0 && (curnpths/2)*ncoresinanode > kway)
	{	
		int eps = (curnpths*ncoresinanode)/kway;
		lb = (curpid/eps)*eps;
		ub = (curpid/eps)*eps+ eps/2 +(curpid%eps)/2;
	}

	for(i = 0; i < curpth->c; i++)
	{
		int iorg = curpth->gcids[i];
		for(j = pth->xpins[iorg]; j < pth->xpins[iorg+1]; j++)
		{
			int receiver = gpartvec[pth->pins[j]];
			if( (receiver < lb || receiver >= ub) && sends[receiver] == 0)
			{				
				if(sendnetsizes[receiver] == 0)
					sendnetids[receiver] = nsendnets++;
				sendnetsizes[receiver]++;
				sends[receiver] = 1;
				totalpin++;
			}
		}
		for(j = 0; j < curnpths; j++)
			sends[j] = 0;

	}

	int *txpins = (int *)calloc((curpth->ne+2),sizeof(int));
	memcpy(txpins, curpth->xpins, (curpth->ne+2)*sizeof(int));

	int *tpins = (int *)calloc(curpth->pe,sizeof(int));
	memcpy(tpins, curpth->pins, curpth->pe*sizeof(int));


	free(curpth->xpins);
	free(curpth->pins);

	curpth->xpins = (int *) calloc(curpth->ne+nsendnets+2, sizeof(int));
	curpth->pins = (int *)calloc(curpth->pe+totalpin, sizeof(int));

	memcpy(curpth->xpins, txpins, (curpth->ne+1)*sizeof(int));
	memcpy(curpth->pins, tpins, curpth->pe*sizeof(int));

	memcpy(txpins, curpth->nwghts, curpth->ne*sizeof(int));
	free(curpth->nwghts);
	curpth->nwghts = (int *)calloc(curpth->ne+nsendnets,sizeof(int));
	memcpy(curpth->nwghts, txpins, curpth->ne*sizeof(int));

	free(txpins);
	free(tpins);

	
	for(i = 0; i < curnpths; i++)
		if(sendnetsizes[i] != 0)
		{	
			if((double)sendnetsizes[i]/curpth->c > *max)
				*max = (double)sendnetsizes[i]/curpth->c;
			curpth->xpins[curpth->ne + 2 + sendnetids[i] ] = sendnetsizes[i];
		}		
		
	for(i = curpth->ne+1; i <= curpth->ne + nsendnets+1; i++)
		curpth->xpins[i] += curpth->xpins[i-1]; 
	
	for(i = 0; i < curpth->c; i++)
	{
		int iorg = curpth->gcids[i];
		for(j = pth->xpins[iorg]; j < pth->xpins[iorg+1]; j++)
		{
			int receiver = gpartvec[pth->pins[j]];
			if( (receiver < lb || receiver >= ub) && sends[receiver] == 0)
			{	
				curpth->pins[curpth->xpins[curpth->ne + 1 + sendnetids[receiver]] ++] = i;
				sends[receiver] = 1;
			}
		}
		for(j = 0; j < curnpths; j++)
			sends[j] = 0;
	}

	int removed = 0;
	if(threshold > -1)
	{
		int norgpin = curpth->xpins[curpth->ne];
		int nsendpins = curpth->xpins[curpth->ne+nsendnets]-norgpin;
		int *tempxpins = (int *)calloc(nsendnets+1, sizeof(int));
		int *temppins = (int *)calloc(nsendpins, sizeof(int));
		int ns = 0, np = 0;

		for(i = curpth->ne; i < curpth->ne+nsendnets; i++)
		{
			if(curpth->xpins[i+1]-curpth->xpins[i] <= threshold)
			{
				for(j = curpth->xpins[i]; j < curpth->xpins[i+1]; j++)
					temppins[np ++] = curpth->pins[j];
				ns++;
				tempxpins[ns] = np;
			}
			else
				removed++;
		}

	
		for(i = 0; i <= ns; i++)
			curpth->xpins[curpth->ne+i] = norgpin + tempxpins[i];
	
		for(i = ns+1; i <= nsendnets; i++)
			curpth->xpins[curpth->ne+i] = 0;

		for(i = 0; i < np; i++)
			curpth->pins[norgpin+i] = temppins[i];

		free(temppins);
		free(tempxpins);

		nsendnets = ns;
	}

	int intcost = (int) round(cost*10);

	for(i = curpth->ne; i < curpth->ne + nsendnets; i++)		
		curpth->nwghts[i] = intcost;

	for(i = 0; i < curpth->n; i++)
         	curpth->nwghts[i] = 10;

	curpth->ne = curpth->ne + nsendnets;
	curpth->pe = curpth->xpins[curpth->ne];

	curpth->xpins[curpth->ne+1] = 0;

	free(sends);
	free(sendnetids);
	free(sendnetsizes);

	return removed;
}

int mnom_addrecvnets(struct pthdata *curpth, struct pthdata *pth, struct pthdata *rpth, int curpid, int curnpths, double cost, double *max, int threshold, int ncoresinanode, int kway, int selective)
{
	int i, j, lb, ub;
	// now add receive nets
	int nrecvnets = 0;
	int *recvs = (int *)calloc(curnpths, sizeof(int));
	int *recvnetids = (int *)calloc(curnpths, sizeof(int));
	int *recvnetsizes = (int *)calloc(curnpths, sizeof(int));
	int totalpin = 0;


	lb = curpid;
	ub = curpid+1;

	/* printf("    in %s, lb %d ub %d\n", __FUNCTION__, lb, ub); */

	if(selective != 0 && (curnpths/2)*ncoresinanode > kway)
	{	
		int eps = (curnpths*ncoresinanode)/kway;
		lb = (curpid/eps)*eps;
		ub = (curpid/eps)*eps+ eps/2 +(curpid%eps)/2;
	}


	for(i = 0; i < curpth->c; i++)
	{
		int iorg = curpth->gcids[i];
		for(j = rpth->xpins[iorg]; j < rpth->xpins[iorg+1]; j++)
		{
			int sender = gpartvec[rpth->pins[j]];
			if( (sender < lb || sender >= ub) && recvs[sender] == 0)
			{				
				if(recvnetsizes[sender] == 0)
					recvnetids[sender] = nrecvnets++;
				recvnetsizes[sender]++;
				recvs[sender] = 1;
				totalpin++;
			}
		}
		for(j = 0; j < curnpths; j++)
			recvs[j] = 0;

	}

	int *txpins = (int *)calloc((curpth->ne+2),sizeof(int));
	memcpy(txpins, curpth->xpins, (curpth->ne+2)*sizeof(int));

	int *tpins = (int *)calloc(curpth->pe,sizeof(int));
	memcpy(tpins, curpth->pins, curpth->pe*sizeof(int));


	free(curpth->xpins);
	free(curpth->pins);

	curpth->xpins = (int *) calloc(curpth->ne+nrecvnets+2, sizeof(int));
	curpth->pins = (int *)calloc(curpth->pe+totalpin, sizeof(int));

	memcpy(curpth->xpins, txpins, (curpth->ne+1)*sizeof(int));
	memcpy(curpth->pins, tpins, curpth->pe*sizeof(int));

	memcpy(txpins, curpth->nwghts, curpth->ne*sizeof(int));
	free(curpth->nwghts);
	curpth->nwghts = (int *)calloc(curpth->ne+nrecvnets,sizeof(int));
	memcpy(curpth->nwghts, txpins, curpth->ne*sizeof(int));

	free(txpins);
	free(tpins);

	for(i = 0; i < curnpths; i++)
		if(recvnetsizes[i] != 0)
		{	
			if((double)recvnetsizes[i]/curpth->c > *max)
				*max = (double)recvnetsizes[i]/curpth->c;
			curpth->xpins[curpth->ne + 2 + recvnetids[i] ] = recvnetsizes[i];
		}		
				
	for(i = curpth->ne+1; i <= curpth->ne + nrecvnets+1; i++)
		curpth->xpins[i] += curpth->xpins[i-1]; 
	
	for(i = 0; i < curpth->c; i++)
	{
		int iorg = curpth->gcids[i];
		for(j = rpth->xpins[iorg]; j < rpth->xpins[iorg+1]; j++)
		{
			int sender = gpartvec[rpth->pins[j]];
			if( (sender < lb || sender >= ub) && recvs[sender] == 0)
			{	
				curpth->pins[curpth->xpins[curpth->ne + 1 + recvnetids[sender]] ++] = i;
				recvs[sender] = 1;
			}
		}
		for(j = 0; j < curnpths; j++)
			recvs[j] = 0;
	}
	
	int removed = 0;

	if(threshold > -1)
	{
		int norgpin = curpth->xpins[curpth->ne];
		int nrecvpins = curpth->xpins[curpth->ne+nrecvnets]-norgpin;
		int *tempxpins = (int *)calloc(nrecvnets+1, sizeof(int));
		int *temppins = (int *)calloc(nrecvpins, sizeof(int));
		int nr = 0, np = 0;

		for(i = curpth->ne; i < curpth->ne+nrecvnets; i++)
		{
			if(curpth->xpins[i+1]-curpth->xpins[i] <= threshold)
			{
				for(j = curpth->xpins[i]; j < curpth->xpins[i+1]; j++)
					temppins[np ++] = curpth->pins[j];
				nr++;
				tempxpins[nr] = np;
			}
			else
				removed++;
		}

	
		for(i = 0; i <= nr; i++)
			curpth->xpins[curpth->ne+i] = norgpin + tempxpins[i];
	
		for(i = nr+1; i <= nrecvnets; i++)
			curpth->xpins[curpth->ne+i] = 0;

		for(i = 0; i < np; i++)
			curpth->pins[norgpin+i] = temppins[i];

		free(temppins);
		free(tempxpins);

		nrecvnets = nr;
	}

	int intcost = (int) round(cost*10);

	for(i = curpth->ne; i < curpth->ne+nrecvnets; i++)		
		curpth->nwghts[i] = intcost;


	for(i = 0; i < curpth->n; i++)
	        curpth->nwghts[i] = 10;

	curpth->ne = curpth->ne + nrecvnets;
	curpth->pe = curpth->xpins[curpth->ne];

	curpth->xpins[curpth->ne+1] = 0;

	free(recvs);
	free(recvnetids);
	free(recvnetsizes);

	return removed;
}


/* @OGUZ-TODO-UPDATE */
// updates global partition vector with the local partition vector of a bipartition
int updategpartvec( int *lpartvec, int *gcids, int lN, int pid)
{	
	int i; 
	for(i = 0; i < N; i++)
	{
		if(gpartvec[i] > pid)
			gpartvec[i] ++;
		
	}
	
	if(lpartvec != NULL)
	{	
		for(i = 0; i < lN; i++)
		{
			if(lpartvec[i] == 1)
			{
				if(gpartvec[gcids[i]] != pid)
				{	
					printf("update: pid[%d] not expected: %d..\nlid: %d, gid: %d\n", pid, gpartvec[gcids[i]], i, gcids[i]);
					return 1;
				}
				gpartvec[gcids[i]] ++;		
			}	
		}
	}
	return 0;
}

// bipartitions the given hypergraph to left and right hypergraphs with net splitting 
// a split net whose source vertex is not at the same part becomes NONLOCAL
int bisect(struct pthdata *pth, int *lpartvec, struct pthdata *leftpth, struct pthdata *rightpth, int *map, int mb)
{
	int i,j;	

	leftpth->c = 0;
	leftpth->p = 0;
	leftpth->n = 0;
	if(mb == 0)
		leftpth->nconst = 1;
	else
		leftpth->nconst = 2;
	leftpth->nad =  0;

	rightpth->c = 0;
	rightpth->p = 0;
	rightpth->n = 0;
	if(mb == 0)
		rightpth->nconst = 1;
	else
		rightpth->nconst = 2;
	rightpth->nad = 0;

	for(i = 0; i < pth->c; i++)
	{
		if(lpartvec[i] == 0)	
			map[i] = leftpth->c++;
		else	
			map[i] = rightpth->c++;
	}

	leftpth->cwghts = (int *)calloc(leftpth->c*leftpth->nconst,sizeof(int));
	rightpth->cwghts = (int *)calloc(rightpth->c*rightpth->nconst,sizeof(int));
	leftpth->gcids = (int *)calloc(leftpth->c,sizeof(int));
	rightpth->gcids = (int *)calloc(rightpth->c,sizeof(int));

	for(i = 0; i < pth->c; i++)
	{	
		if(lpartvec[i] == 0)
		{	
			leftpth->gcids[map[i]] = pth->gcids[i];
			leftpth->cwghts[map[i]*leftpth->nconst] = pth->cwghts[i*pth->nconst];
		}
		else
		{	
			rightpth->gcids[map[i]] = pth->gcids[i];
			rightpth->cwghts[map[i]*leftpth->nconst] = pth->cwghts[i*pth->nconst];
		}

	}

	int leftpin = 0, rightpin = 0, leftnet = 0, rightnet = 0;
	for(i = 0; i < pth->n; i++)
	{
		int ls = leftpin;
		int rs = rightpin;
		for(j = pth->xpins[i]; j < pth->xpins[i+1]; j++)
		{
			int c = pth->pins[j];
			if(lpartvec[c] == 0)	
				leftpin ++;	
			else
				rightpin ++;		
		}
		if(leftpin > ls)
			leftnet++;
		if(rightpin > rs)
			rightnet++;
	 }

	leftpth->xpins = (int *)calloc(leftnet+2, sizeof(int));
	rightpth->xpins = (int *)calloc(rightnet+2, sizeof(int));
	leftpth->pins = (int *)calloc(leftpin, sizeof(int));
	rightpth->pins = (int *)calloc(rightpin, sizeof(int));
	leftpth->nwghts = (int *)calloc(leftnet+2, sizeof(int));
	rightpth->nwghts = (int *)calloc(rightnet+2, sizeof(int));

	for(i = 0; i < pth->n; i++)
	{
		for(j = pth->xpins[i]; j < pth->xpins[i+1]; j++)
		{
			int c = pth->pins[j];
			if(lpartvec[c] == 0)	
				leftpth->pins[leftpth->p ++] = map[c];	
			else
				rightpth->pins[rightpth->p ++] = map[c];
		
		}
		
		if(leftpth->xpins[leftpth->n] < leftpth->p) 
		{	
			leftpth->xpins[leftpth->n+1] = leftpth->p;
			leftpth->nwghts[leftpth->n] = pth->nwghts[i];
			leftpth->n++;
		}
		// add right split net to right pth
		if(rightpth->xpins[rightpth->n] < rightpth->p) 
		{			
			rightpth->xpins[rightpth->n+1] = rightpth->p;
			rightpth->nwghts[rightpth->n] = pth->nwghts[i];
			rightpth->n++;
		}

	}
	leftpth->pe = leftpth->p;	//now there's no super-nets
	leftpth->ne = leftpth->n;	//super-nets will be added just before bisection
	rightpth->pe = rightpth->p;
	rightpth->ne = rightpth->n;
	
	return 0;

}


int mnom_setsecondweights(struct pthdata *pth, int mb, int nor, int sym, int ns)
{
	int i, j;
	double *sw = (double *)calloc(pth->c, sizeof(double));

	if(sym)
	{	
		for(i = pth->n; i < pth->ne; i++)
		{
			int deg = pth->xpins[i+1]-pth->xpins[i];
			int cost = pth->nwghts[i]/2;

			for(j = pth->xpins[i]; j < pth->xpins[i+1]; j++)
			{
				sw[pth->pins[j]] += (double) cost/deg; 
			}
		}
	}
	else
	{
		for(i = pth->n; i < pth->n + ns; i++)
		{
			int deg = pth->xpins[i+1]-pth->xpins[i];
			int cost = pth->nwghts[i];

			for(j = pth->xpins[i]; j < pth->xpins[i+1]; j++)
			{
				sw[pth->pins[j]] += (double) cost/deg; 
			}
		}
	}

	double min = DBL_MAX;
	double max = DBL_MIN;
	for(i = 0; i < pth->c; i++)
	{
		if(sw[i] != 0 && sw[i] < min)
			min = sw[i];
		if(sw[i] > max)
			max = sw[i];
	}
	if(mb == 1 )
	{
		if(min <= 1)
			for(i = 0; i < pth->c; i++)
				pth->cwghts[2*i+1] = (int) (sw[i]/min);
		else
			for(i = 0; i < pth->c; i++)
				pth->cwghts[2*i+1] = (int) sw[i];
	}
	else if(mb == 2)	
	{
		for(i = 0; i < pth->c; i++)
				pth->cwghts[2*i+1] = (int) (sw[i]/min);
	}
	else
	{
		for(i = 0; i < pth->c; i++)
			pth->cwghts[2*i+1] = (int) ((sw[i]/max)*mb);

	}

	if(nor)
	{	
		double ratio = 1, firstsum = 0, secondsum = 0; 
		int secondcnt = 0;
		for(i = 0; i < pth->c; i++)
			firstsum += pth->cwghts[2*i];
		for(i = 0; i < pth->c; i++)
		{	
			if(pth->cwghts[2*i+1] > 0)
			{	
				secondcnt++;
				secondsum += pth->cwghts[2*i+1];
			}
		}	
		
		if(secondcnt != 0 && secondsum != 0)
		{
			ratio = (firstsum*secondcnt)/(secondsum*pth->c);
			for(i = 0; i < pth->c; i++)
				pth->cwghts[2*i+1] = (int) (pth->cwghts[2*i+1]*ratio);
		}
	}
		
 	free(sw);

	return 0;
}


// partitions the given hypergraph into kway parts with recursive bisection in bfs ordering
/* int */
/* partition_rb_bfs_gen( */
/* 	struct pthdata *pth, */
/* 	struct pthdata *rpth, */
/* 	int kway, */
/* 	int cost, */
/* 	double corrimb, */
/* 	int sym, */
/* 	int *efnd, */
/* 	int mb, */
/* 	int nor, */
/* 	double *max, */
/* 	int threshold, */
/* 	int *fs, */
/* 	int *rm, */
/* 	int adaptive, */
/* 	int delayed, */
/* 	int unified, */
/* 	int relaxed */
/* 	) */
/* { */
/* 	int i, j; */
/* 	int curnpth = 1, l = 0; */
/* 	double mymax = 0;  */
/* 	int firstseen = -1; */

/* 	curlevel = (struct pthdata **)calloc(curnpth, sizeof(struct pthdata *)); */
/* 	nextlevel = (struct pthdata **)calloc(curnpth*2, sizeof(struct pthdata *)); */

/* 	/\* bound is ceiling in the more general case *\/ */
/* 	pth->curk = kway; */
/* 	int kway_bound = kway; */
/* 	if (kway & (kway - 1)) */
/* 		kway_bound = (int) pow(2, ((int) log2(kway)) + 1); */
/* 	/\* printf("bound is %d\n", kway_bound); *\/ */
	
/* 	curlevel[0] = pth;	 */

/* 	int rbheight = (int) log2(kway); */
/* 	rbheight--; */

/* 	double deltacost = (double) cost/rbheight; */
/* 	double runningcost = 0; */

/* 	int lastid; */

/* 	while(curnpth != kway_bound) */
/*  	{ */
/* 		lastid = 0; */
/* 		/\* printf("curnpth = %d\n", curnpth); *\/ */
		
/* 		if(curnpth > 1 && runningcost < 2) */
/* 			runningcost = 2; */

/* 		int comp = 0, compmin = kway, compmax = 0; */
/* 		int snet = 0, snetmin = 2*kway, snetmax = 0;  */
/* 		int spin = 0, spinmin = 2*N, spinmax = 0, sno = 0; */
/* 		long long pin = 0, no = 0; */

/* 		for(i = 0; i < curnpth; i++) */
/* 		{ */
/* 			struct pthdata *curpth = (struct pthdata *)curlevel[i]; */
			
/* 			/\* printf("  processing hygr %d, curk = %d, lastid = %d\n", *\/ */
/* 			/\* 	   i, curpth->curk, lastid); *\/ */
			
/* 			if (curpth->curk == 1) */
/* 			{ */
/* 				++lastid; */
/* 				continue; */
/* 			} */

/* 			if(curpth != NULL) */
/* 			{ */
/* 				struct pthdata *leftpth = (struct pthdata *)calloc(1, sizeof(struct pthdata)); */
/* 				struct pthdata *rightpth = (struct pthdata *)calloc(1, sizeof(struct pthdata)); */

/* 				nextlevel[2*i] = leftpth; */
/* 				nextlevel[2*i+1] = rightpth; */
				
/* 				if(curnpth > 1 && mnom > 0 && delayed <= curnpth)  */
/* 				{ */
/* 					int ns = -1; */
/* 					if(sym) */
/* 						(*rm) += mnom_addsendnets(curpth, pth, rpth, lastid, 2*curnpth, adaptive > 0 ? 2*runningcost: 2*cost, &mymax, threshold, ncoresinanode, kway, selective); */
/* 					else */
/* 					{ */
/* 						(*rm) += mnom_addsendnets(curpth, pth, rpth, lastid, 2*curnpth, adaptive > 0? runningcost : cost, &mymax, threshold, ncoresinanode, kway, selective); */
/* 						ns = curpth->ne-curpth->n; */
/* 						(*rm) += mnom_addrecvnets(curpth, pth, rpth, lastid, 2*curnpth, adaptive > 0? runningcost : cost, &mymax, threshold, ncoresinanode, kway, selective); */
/*                     } */
/* 					if(firstseen == -1 && curpth->ne-curpth->n) */
/* 						firstseen = l;				 */
						
/* 					spin += curpth->pe-curpth->p; */
/* 					sno += curpth->ne-curpth->n; */

/* 					pin += curpth->p; */
/* 					no += curpth->n; */
					
/* 					if(mb > 0 && relaxed <= curnpth) */
/* 						mnom_setsecondweights(curpth, mb, nor, sym, ns);			 */

						
/* 				} */
/* 				int *lpartvec = (int *)calloc(curpth->c, sizeof(int)); */

/* 				/\* printf("    partitioning ... "); *\/ */
/* 				totalcut += pthpartition_gen(curpth, 2, corrimb, lpartvec, 12, unified); */
/* 				/\* printf("\n"); *\/ */

/* 				/\* printf("    updating global partvec with lastid %d\n", lastid); *\/ */
/* 				if(updategpartvec(lpartvec, curpth->gcids, curpth->c, lastid)) */
/* 					return 1; */

/* 				lastid += 2; */
/* 				leftpth->curk = curpth->curk / 2; */
/* 				rightpth->curk = leftpth->curk + curpth->curk % 2; */
			
/* 				int *map = (int *)calloc(curpth->c, sizeof(int)); */

/* 				int multi = 0; */
/* 				if(mb > 0 && curnpth >= relaxed/2 && curnpth >= delayed/2) */
/* 					multi = 1; */

/* 				if(bisect(curpth, lpartvec, leftpth, rightpth, map, multi)) */
/* 					return 1; */

/* 				if(leftpth->c == 0) */
/* 				{ */
/* 					fprintf(stdout, "may cause trouble, leftpth->c = 0\n"); */
/* 					freepth(leftpth); */
/* 					nextlevel[2*i] = NULL; */
/* 				} */
/* 				if(rightpth->c == 0) */
/* 				{ */
/* 					fprintf(stdout, "may cause trouble, rightpth->c = 0\n"); */
/* 					freepth(rightpth); */
/* 					nextlevel[2*i+1] = NULL; */
/* 				}		 */

/* 				free(map); */
/* 				free(lpartvec); */
/* 				if(curnpth > 1) */
/* 					freepth(curpth); */
/* 			} */
/* 			else */
/* 			{ */
/* 				fprintf(stdout, "This may cause trouble, check this branch!\n"); */
/* 				nextlevel[2*i] = NULL; */
/* 				nextlevel[2*i+1] = NULL; */
/* 				if(updategpartvec(NULL, NULL, 0, 2*i)) */
/* 					return 1; */
/* 			} */
			
/* 		} */

/* 		/\*if(mnom != 0 && curnpth > 1) */
/* 			printf("L=%d [P=%d]\t%.8lf %d %d\t%.8lf %d %d\t%.8lf %d %d\n", l, curnpth,  */
/* 				(double)comp/curnpth, (compmin == kway)?0:compmin, compmax, */
/* 				(double)snet/curnpth, (snetmin == 2*kway)?0:snetmin, snetmax, */
/* 				(sno==0)?0:(double)spin/sno, (spinmin == 2*N)?0:spinmin, spinmax); */
/* 		*\/ */


/* 		/\* if(mnom != 0 && curnpth > 1) *\/ */
/* 		/\* 	printf("L=%d [P=%d]\t%.8lf\t%.8lf\n", l, curnpth, *\/ */
/* 		/\* 		(sno==0)?0:(double)spin/sno, (no==0)?0:(double)pin/no); *\/ */

/* 		l++; */

/* 		curnpth = curnpth*2; */
		
/* 		if(curnpth != kway) */
/* 		{ */
		
/* 			free(curlevel); */
/* 			curlevel = nextlevel; */
/* 			nextlevel = (struct pthdata **)calloc(curnpth*2, sizeof(struct pthdata *)); */

/* 		} */
/* 		runningcost += deltacost; */
	
/* 	} */

/* 	free(curlevel); */


/* 	int ec = 0; */
/* 	for(i = 0; i < curnpth; i++) */
/* 	{	 */
/* 		struct pthdata *freeme = (struct pthdata *)nextlevel[i]; */
/* 		if(freeme != NULL) */
/* 			freepth(freeme); */
/* 		else */
/* 			ec++; */
/* 	} */
	
/* 	free(nextlevel);  */

/* 	*efnd = ec;	 */
/* 	*max = mymax; */
/* 	*fs = firstseen; */
/* 	return 0; */
/* } */


// partitions the given hypergraph into kway parts with recursive bisection in bfs ordering
// uses tilesize
int
partition_rb_bfs_gen_tile(
	struct pthdata *pth,
	struct pthdata *rpth,
	int kway,
	int cost,
	double corrimb,
	int sym,
	int *efnd,
	int mb,
	int nor,
	double *max,
	int threshold,
	int *fs,
	int *rm,
	int adaptive,
	int delayed,
	int unified,
	int relaxed,
	kahypar_context_t *context
	)
{
	int i, j;
	int curnpth = 1, l = 0;
	double mymax = 0; 
	int firstseen = -1;

	curlevel = (struct pthdata **)calloc(curnpth, sizeof(struct pthdata *));
	nextlevel = (struct pthdata **)calloc(curnpth*2, sizeof(struct pthdata *));

	/* bound is ceiling in the more general case */
	pth->curk = kway;
	int kway_bound = kway;
	if (kway & (kway - 1))
		kway_bound = (int) pow(2, ((int) log2(kway)) + 1);
	/* printf("bound is %d\n", kway_bound); */
	
	curlevel[0] = pth;	

	int rbheight = (int) log2(kway);
	rbheight--;

	double deltacost = (double) cost/rbheight;
	double runningcost = 0;

	int lastid;

	while(curnpth != kway_bound)
 	{
		lastid = 0;
		/* printf("curnpth = %d\n", curnpth); */
		
		if(curnpth > 1 && runningcost < 2)
			runningcost = 2;

		int comp = 0, compmin = kway, compmax = 0;
		int snet = 0, snetmin = 2*kway, snetmax = 0; 
		int spin = 0, spinmin = 2*N, spinmax = 0, sno = 0;
		long long pin = 0, no = 0;

		for(i = 0; i < curnpth; i++)
		{
			struct pthdata *curpth = (struct pthdata *)curlevel[i];
			
			/* printf("  processing hygr %d, curk = %d, lastid = %d\n", */
			/* 	   i, curpth->curk, lastid); */
			
			if (curpth->curk == 1)
			{
				++lastid;
				continue;
			}

			if(curpth != NULL)
			{
				struct pthdata *leftpth = (struct pthdata *)calloc(1, sizeof(struct pthdata));
				struct pthdata *rightpth = (struct pthdata *)calloc(1, sizeof(struct pthdata));

				nextlevel[2*i] = leftpth;
				nextlevel[2*i+1] = rightpth;
				
				if(curnpth > 1 && mnom > 0 && delayed <= curnpth) 
				{
					int ns = -1;
					if(sym)
						(*rm) += mnom_addsendnets(curpth, pth, rpth, lastid, 2*curnpth, adaptive > 0 ? 2*runningcost: 2*cost, &mymax, threshold, ncoresinanode, kway, selective);
					else
					{
						(*rm) += mnom_addsendnets(curpth, pth, rpth, lastid, 2*curnpth, adaptive > 0? runningcost : cost, &mymax, threshold, ncoresinanode, kway, selective);
						ns = curpth->ne-curpth->n;
						(*rm) += mnom_addrecvnets(curpth, pth, rpth, lastid, 2*curnpth, adaptive > 0? runningcost : cost, &mymax, threshold, ncoresinanode, kway, selective);
                    }
					if(firstseen == -1 && curpth->ne-curpth->n)
						firstseen = l;				
						
					spin += curpth->pe-curpth->p;
					sno += curpth->ne-curpth->n;

					pin += curpth->p;
					no += curpth->n;
					
					if(mb > 0 && relaxed <= curnpth)
						mnom_setsecondweights(curpth, mb, nor, sym, ns);			

						
				}
				int *lpartvec = (int *)calloc(curpth->c, sizeof(int));

				/* printf("    partitioning ... "); */
				totalcut += kahyparpartition_gen_tile(curpth, 2, corrimb, lpartvec, 12, unified, tilesize, context);
				/* printf("\n"); */

				/* printf("    updating global partvec with lastid %d\n", lastid); */
				if(updategpartvec(lpartvec, curpth->gcids, curpth->c, lastid))
					return 1;

				lastid += 2;
				rightpth->curk = curpth->curk / 2;
				leftpth->curk = rightpth->curk + curpth->curk % 2;
			
				int *map = (int *)calloc(curpth->c, sizeof(int));

				int multi = 0;
				if(mb > 0 && curnpth >= relaxed/2 && curnpth >= delayed/2)
					multi = 1;

				if(bisect(curpth, lpartvec, leftpth, rightpth, map, multi))
					return 1;

				if(leftpth->c == 0)
				{
					fprintf(stdout, "may cause trouble, leftpth->c = 0\n");
					freepth(leftpth);
					nextlevel[2*i] = NULL;
				}
				if(rightpth->c == 0)
				{
					fprintf(stdout, "may cause trouble, rightpth->c = 0\n");
					freepth(rightpth);
					nextlevel[2*i+1] = NULL;
				}		

				free(map);
				free(lpartvec);
				if(curnpth > 1)
					freepth(curpth);
			}
			else
			{
				fprintf(stdout, "This may cause trouble, check this branch!\n");
				nextlevel[2*i] = NULL;
				nextlevel[2*i+1] = NULL;
				if(updategpartvec(NULL, NULL, 0, 2*i))
					return 1;
			}
			
		}

		/*if(mnom != 0 && curnpth > 1)
			printf("L=%d [P=%d]\t%.8lf %d %d\t%.8lf %d %d\t%.8lf %d %d\n", l, curnpth, 
				(double)comp/curnpth, (compmin == kway)?0:compmin, compmax,
				(double)snet/curnpth, (snetmin == 2*kway)?0:snetmin, snetmax,
				(sno==0)?0:(double)spin/sno, (spinmin == 2*N)?0:spinmin, spinmax);
		*/


		/* if(mnom != 0 && curnpth > 1) */
		/* 	printf("L=%d [P=%d]\t%.8lf\t%.8lf\n", l, curnpth, */
		/* 		(sno==0)?0:(double)spin/sno, (no==0)?0:(double)pin/no); */

		l++;

		curnpth = curnpth*2;
		
		if(curnpth != kway)
		{
		
			free(curlevel);
			curlevel = nextlevel;
			nextlevel = (struct pthdata **)calloc(curnpth*2, sizeof(struct pthdata *));

		}
		runningcost += deltacost;
	
	}

	free(curlevel);


	int ec = 0;
	for(i = 0; i < curnpth; i++)
	{	
		struct pthdata *freeme = (struct pthdata *)nextlevel[i];
		if(freeme != NULL)
			freepth(freeme);
		else
			ec++;
	}
	
	free(nextlevel); 

	*efnd = ec;	
	*max = mymax;
	*fs = firstseen;
	return 0;
}

/* // partitions the given hypergraph into curkway parts with recursive bisection in dfs ordering */
/* int partition_rb_dfs(struct pthdata *curpth, struct pthdata *pth, struct pthdata *rpth, int curkway, int cost, double corrimb, int pid, int sym, int *efnd, int mb, int nor, double *max, int threshold) */
/* { */
/* 	int i; */
/* 	int *lpartvec = (int *)calloc(curpth->c, sizeof(int)); */
/* 	double mymax = 0; */

/* 	if(curkway != kway && mnom > 0) */
/* 	{ */
/* 		int ns = -1;	 */
/* 		if(sym) */
/* 			mnom_addsendnets(curpth, pth, rpth, pid, kway, 2*cost, &mymax, threshold, 2, kway, 0); */
/* 		else */
/* 		{ */
/* 			mnom_addsendnets(curpth, pth, rpth, pid, kway, cost, &mymax, threshold, 2, kway, 0); */
/* 			ns = pth->ne-pth->n; */
/* 			mnom_addrecvnets(curpth, pth, rpth, pid, kway, cost, &mymax, threshold, 2, kway, 0); */
/* 		} */
/* 		if(mb > 0) */
/* 			mnom_setsecondweights(curpth, mb, nor, sym, ns); */
/* 	} */

/* 	totalcut += pthpartition(curpth, 2, corrimb, lpartvec, 12, 0);	 */

/* 	if(updategpartvec(lpartvec, curpth->gcids, curpth->c, pid)) */
/* 		return 1; */

/* 	if(curkway == 2) */
/* 	{ */
/* 		free(lpartvec); */
/* 		return 0; */
/* 	}	 */

/* 	int *map = (int *)calloc(curpth->c, sizeof(int)); */
/* 	struct pthdata *pthleft = (struct pthdata *)calloc(1, sizeof(struct pthdata)); */
/* 	struct pthdata *pthright = (struct pthdata *)calloc(1, sizeof(struct pthdata)); */

/* 	if(bisect(curpth, lpartvec, pthleft, pthright, map, mb)) */
/* 		return 1; */

/* 	if(pthleft->c == 0 ||pthright->c == 0) */
/* 	{ */
/* 		*efnd = 1; */
/* 		free(map); */
/* 		free(lpartvec); */
/* 		return 0;	 */
/* 	} */

/* 	int lefnd = 0, refnd = 0;  */
/* 	partition_rb_dfs(pthleft, pth, rpth, curkway/2, cost, corrimb, pid, sym, &lefnd, mb, nor, &mymax, threshold); */
/* 	freepth(pthleft); */

/* 	if(lefnd == 1) */
/* 	{ */
/* 		*efnd = 1; */
/* 		free(map); */
/* 		free(lpartvec); */
/* 		return 0;	 */
/* 	}	 */

/* 	partition_rb_dfs(pthright, pth, rpth, curkway/2, cost, corrimb, pid+curkway/2, sym, &refnd, mb, nor, &mymax, threshold); */
/* 	freepth(pthright); */
		
/* 	if(refnd == 1) */
/* 		*efnd = 1;	 */

/* 	free(map); */
/* 	free(lpartvec); */
/* 	*max = mymax; */

/* 	return 0; */
/* } */

int calculateallimbalances(int *maxcount, int *totalcount, int *maxvol, int *totalvol, double *compimb, struct pthdata *pth, int *gpartvec, int *intra, int *inter, int ncoresinanode)
{
	int i,j, ec = 0;
	*inter = 0;
	*intra = 0;
	int **sends = (int **)calloc(kway, sizeof(int *));
	for(i = 0; i < kway; i++)
		sends[i] = (int *)calloc(kway, sizeof(int));
	
	int *parts = (int *)calloc(kway, sizeof(int));

	for(i = 0; i < pth->n; i++)
	{
		for(j = pth->xpins[i]; j < pth->xpins[i+1]; j++)
			parts[gpartvec[pth->pins[j]]] = 1;

		for(j = 0; j < kway; j++)
			if(parts[j] != 0)		
				sends[gpartvec[i]][j]++;
		
		for(j = 0; j < kway; j++)
			parts[j] = 0;
	}

	int *sendvols = (int *)calloc(kway, sizeof(int));
	int *sendcounts = (int *)calloc(kway, sizeof(int));
	for(i = 0; i < kway; i++)
		for(j = 0; j < kway; j++)
			if(i != j)
			{	
				sendvols[i] += sends[i][j];
				if(sends[i][j] != 0)
				{	
					sendcounts[i] ++;
				
					if( (i / ncoresinanode) == (j / ncoresinanode) )
						(*intra)++;
					else
						(*inter)++;
				}
			}

	*totalvol = 0, *totalcount = 0, *maxvol = 0, *maxcount = 0;
	for(i = 0; i < kway; i++)
	{
		if(sendvols[i] > *maxvol)
			*maxvol = sendvols[i]; 
		*totalvol += sendvols[i];

		if(sendcounts[i] > *maxcount)
			*maxcount = sendcounts[i];
		*totalcount += sendcounts[i];
	}

	int totalpw = 0, maxpw = 0;	
	int *partweights = (int *)calloc(kway, sizeof(int));
	for(i = 0; i < N; i++)
	{	
		partweights[gpartvec[i]] += pth->cwghts[i*pth->nconst];
	}
	
	for(i = 0; i < kway; i++)
 	{
		if(partweights[i] > maxpw)
			maxpw = partweights[i];
		totalpw += partweights[i];	
	}
	*compimb = ((double) maxpw)*kway/totalpw-1;

	for(i = 0; i < kway; i++)
		free(sends[i]);
	free(sends);
	free(sendvols);
	free(sendcounts);
	free(parts);
	free(partweights);

	return 0;
}


void writevec(char *fout, int *gpartvec, int n)
{
	int i;
	FILE *f = fopen(fout, "w");
	for(i = 0; i < n; i++)
		fprintf(f, "%d\n", gpartvec[i]);
	fclose(f);
	
}


py::array_t<int>
run (
	py::array_t<int>	 rids,
	py::array_t<int>	 cids,
	int					 nrows,
	int					 ncols,
	int					 nnzs,
	int					 tile_size,
	int					 msg_net_cost,
	std::string			&context_file
	)
{
	int i;

	initialize_parameters();
	tilesize = tile_size;
	cost	 = msg_net_cost;

	// convert to matrix market format
	struct mmdata *mm = (struct mmdata *) calloc(1, sizeof(struct mmdata));
	// initialize_mm(matrixfile, mm);

	py::buffer_info rids_buf = rids.request(),
		cids_buf = cids.request();

	mm->symmetricity = 0;
	mm->ndiagonal	 = 0;		// does not matter for general matrices
	mm->x			 = (int *) rids_buf.ptr;
	mm->y			 = (int *) cids_buf.ptr;
	mm->v			 = NULL;
	mm->binary		 = 1;
	mm->N			 = nrows;
	mm->M			 = ncols;
	mm->NNZ			 = nnzs;
	mm->realnnz      = nnzs;

	printf("nrows = %d, ncols = %d\n", mm->N, mm->M);

	// printf("rows: ");
	// for (int i = 0; i < nnzs; ++i)
	// 	printf("%d ", mm->x[i]);
	// printf("\ncols: ");
	// for (int i = 0; i < nnzs; ++i)
	// 	printf("%d ", mm->y[i]);
	// printf("\n");
	
	
	if(mm->N != mm->M)
	{
		printf("%s is not square..\n", matrixname);
		freemm(mm);
		return py::array_t<int>(0);;
	}
	
	struct pthdata *pth = (struct pthdata *) calloc(1, sizeof(struct pthdata));
	mm2pth_colnet(mm, pth);

	struct pthdata *rpth = (struct pthdata *) calloc(1, sizeof(struct pthdata));
	mm2pth_rownet(mm, rpth);

	gpartvec = (int *)calloc(pth->c, sizeof(int));
	N = pth->c;

	if (tilesize != 0)
	{
		kway = (pth->c + tilesize - 1) / tilesize;
		/* printf("computing number of parts using tilesize, k = %d\n", kway); */
	}

	double	corrimb	   = pow (1.0+imbalance,
							  1.0/ceil(log((double)kway)/log(2.0)))-1.0;
	int		emptycount = 0;
	int		err		   = 0;
	totalcut		   = 0;
	double	max		   = 0;
	int		fs		   = -1;
	int		rm		   = 0;

	if(mnom == 0 && mb == 1)
	{
		printf("[Warning] Nonexisting super nets can not be balanced.. I'm treating as b = 0\n");
		mb = 0;
	}
	if(mb == 0 && nor == 1)
	{
		printf("[Warning] Nonexisting second weights can not be normalized.. I'm treating as n = 0\n");
		nor = 0;
	}
	if(mb == 0 && unified > 0)
	{
		printf("[Warning] Message balancing should be active for unified weighting.. I'm treating as b = 1\n");
		mb = 1;
	}
	if(mb == 0 && relaxed > 1)
	{
		printf("[Warning] Message balancing should be active for relaxed weighting.. I'm treating as b = 1\n");
		mb = 1;
	}


	// @TODO
	/* Partitioning with KaHyPar */
	kahypar_context_t* context = kahypar_context_new();
	kahypar_configure_context_from_file(context,  context_file.c_str() );

	/* @OGUZ-EDIT Begin */
	struct timespec ts_start;
	struct timespec ts_end;
	clock_gettime(CLOCK_MONOTONIC, &ts_start);
	/* clock_t start, end; */
	/* start = clock(); */
	/* @OGUZ-EDIT End */

	if(order == BFS_ORDER)
		err =
			partition_rb_bfs_gen_tile(pth, rpth, kway, cost, corrimb,
									  mm->symmetricity, &emptycount,
									  mb, nor, &max,
									  threshold, &fs, &rm, adaptive,
									  delayed, unified, relaxed, context);
	else
	{
		printf("cannot partition using DFS.\n");
		exit(1);
		/* err = partition_rb_dfs(pth, pth, rpth, kway, cost, corrimb, 0,
		   mm->symmetricity, &emptycount, mb, nor, &max, threshold); */
	}

	/* @OGUZ-EDIT Begin */
	clock_gettime(CLOCK_MONOTONIC, &ts_end);
	double elapsed_sec = 1e9*(double)(ts_end.tv_sec-ts_start.tv_sec) +
		(double)(ts_end.tv_nsec - ts_start.tv_nsec);
	elapsed_sec = elapsed_sec/1e9;
	/* end = clock(); */ 
	/* @OGUZ-EDIT End */
	

	if(err)
	{
		printf("error occured\n");
		freepth(pth);
		freemm(mm);
		free(gpartvec);

		return py::array_t<int>(0);
	}	

	/* if(emptycount != 0) */
	/* 	printf("[Warning] There is at least one empty part (matrix = %s)\n", matrixname); */

	int maxdegree, maxcount, totalcount, maxvol, totalvol, inter, intra;
	double compimb;

	calculateallimbalances(&maxcount, &totalcount, &maxvol, &totalvol, &compimb,
						   pth, gpartvec, &intra, &inter, ncoresinanode);

	if( mnom == 0 && totalcut != totalvol)
		printf("Total cut in the hypergraph is not equal to the computed total volume..\n");
	else
	{
		print_params();
		/* @OGUZ-EDIT Begin */
		printf("%s %d %d %f %f %d %d %d %d %d %d %f %d %d %d %.3f %d %d\n", matrixname, N, mm->realnnz, 
							compimb, (double)maxvol*kway/totalvol-1, 
							totalvol/kway, maxvol, totalvol, 
		       totalcount/kway, maxcount, totalcount, max, fs, rm, emptycount, elapsed_sec, inter, intra);
		/* @OGUZ-EDIT End */
	}

	

	// writevec(partfile, gpartvec, pth->c);

	/* @OGUZ-EDIT Begin */
	int *pw = (int *)calloc(kway, sizeof(*pw));
	for (i = 0; i < pth->c; ++i)
		pw[gpartvec[i]] += 1;
	int pwmin = 1e9;
	int pwmax = -1;
	for (i = 0; i < kway; ++i)
	{
		if (pw[i] < pwmin)
			pwmin = pw[i];
		if (pw[i] > pwmax)
			pwmax = pw[i];
		/* fprintf(stdout, "P %4d -> %d\n", i, pw[i]); */
	}
	fprintf(stdout, "pwmin %d pwmax %d\n", pwmin, pwmax);
	free(pw);
	/* @OGUZ-EDIT End */


	py::array_t<int> pvec({mm->N});
	py::buffer_info pvec_buf = pvec.request();
	int *tmp = (int *)pvec_buf.ptr;
	for (i = 0; i < pth->c; ++i)
		tmp[i] = gpartvec[i];

	// free(rpth->xpins);
	// free(rpth->pins);
	// free(rpth);

	// freepth(pth);
	// freemm(mm);
	// free(gpartvec);

	return pvec;
}


PYBIND11_MODULE(pbr, m) {
	m.def("run", &run, "runs the reordering algorithm");
}
