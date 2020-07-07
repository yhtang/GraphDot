/*--------------------------------------------- 
 	Written by Enver Kayaaslan 
 Used for reading .mtx files into mmdata struct
 ----------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mm.h"

#define STRINGSIZE 1000


// reads .mtx files into mmdata struct
// indices start from 0
int initialize_mm(char *file, struct mmdata *mm) {

	char filename[STRINGSIZE];
	char line[MM_MAXLINE];
	char *saveptr, *token;
	sprintf(filename, "%s", file);
	
	FILE *f = fopen(filename, "r");
	
	mm -> symmetricity = 0;
	
	fgets(line, MM_MAXLINE, f);
	if(line[0] == '%') {
	
		token = strtok_r(line, " \n", &saveptr);
		while(token != NULL) {
			if(!strcmp(token, "symmetric")) {
				mm -> symmetricity = 1;
				break;
			}
			token = strtok_r(NULL, " \n", &saveptr);
		}

		fgets(line, MM_MAXLINE, f);	
		while(line[0] == '%') {
			fgets(line, MM_MAXLINE, f);	
		}
	}
		
	sscanf(line, " %d %d %d", &mm->N, &mm->M, &mm->NNZ);
	
	mm -> ndiagonal = 0;
	mm -> x = (int *)malloc(mm->NNZ * sizeof(int));
	mm -> y = (int *)malloc(mm->NNZ * sizeof(int));
	mm -> v = NULL;	

	fgets(line, MM_MAXLINE, f);
	double v;
	if(2 == sscanf(line, " %d %d %lf", &mm -> x[0], &mm -> y[0], &v)) 
		mm -> binary = 1;
	else {
		mm->binary = 0;
		mm -> v = (double *)malloc(mm->NNZ * sizeof(double));	
		mm->v[0] = v;
	}
	mm -> x[0] --;
	mm -> y[0] --;
	if(mm->symmetricity && mm -> x[0] == mm -> y[0])
		mm -> ndiagonal ++;
	
	int i;
	for(i=1; i<mm->NNZ; i++) {

		fgets(line, MM_MAXLINE, f);
		if(! mm -> binary) 
			sscanf(line, " %d %d %lf", &mm -> x[i], &mm -> y[i], &mm -> v[i]);
		else	
			sscanf(line, " %d %d", &mm -> x[i], &mm -> y[i]);

		mm -> x[i] --;
		mm -> y[i] --;

		if(mm->symmetricity && mm -> x[i] == mm -> y[i])
			mm -> ndiagonal ++;
	}

	mm -> realnnz = (!mm -> symmetricity)? mm->NNZ: 2 * mm->NNZ - mm->ndiagonal;

	fclose(f);
	return 0;
}


void printmm(struct mmdata *mm, char *filename) {

	FILE *f = fopen(filename, "w");
	fprintf(f, "%c%cMatrixMarket matrix coordinate %s symmetric\n", '%', '%', mm->binary? "pattern": "real");
	fprintf(f, "%d %d %d\n", mm->N, mm->M, mm->NNZ);
	int i;
	for(i=0; i<mm->NNZ; i++)
		fprintf(f, "%d %d\n", mm->x[i]+1, mm->y[i]+1);
	fclose(f);
}

void freemm(struct mmdata *mm) {

	free(mm->x);
	free(mm->y);
	free(mm->v);
	free(mm);
}
