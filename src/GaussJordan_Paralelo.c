/* Victor Forbes - 9293394
   Gabriel Camargo - 9293456
   Marcos Camargo - 9278045
   Felipe Alegria - 9293501 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

#define true 1
#define false 0

#define NUM_THREADS 2

#define MATRIX_PATH "input/matriz5.txt"
#define	VECTOR_PATH "input/vetor5.txt"
#define RESULT_PATH "output/resultado5.txt"

typedef struct{
	double val;
	int row;
} pair;

/* Imprime a matriz aumentada. */
void print_matrix(double *A, int m, int n){
	int i, j;

	for (i = 0; i < m; i++){
		for (j = 0; j < n + 1; j++){
			printf("%.0lf ", A[i*(n+1) + j]);
		}

		printf("\n");
	}
}

/* Aloca uma matriz de doubles MxN em um vetor. */
double *malloc_matrix(int m, int n){
	double *A;

	A = (double *) malloc(m * n * sizeof(double));

	return A;
}

/* Troca o conteúdo de duas variáveis apontadas por a e b com tamanho size. */
void swap(void *a, void *b, int size){
	void *c;

	if (a != b){
		c = malloc(size);

		memcpy(c, a, size);
		memcpy(a, b, size);
		memcpy(b, c, size);

		free(c);
	}
}

/* Lê uma matriz A e um vetor b (Ax = b) e concatena ambos em uma Matriz Aumentada A. */
double *read_augmented_matrix(int *m, int *n){
	int *length, offset, len, i, j;
	double *A, aux;
	char **line;
	FILE *fp;

	// Inicializando.
	line = NULL;
	length = NULL;
	*m = *n = 0;

	// Abrindo o arquivo com a matriz A.
	fp = fopen(MATRIX_PATH, "r");

	// Lendo todas as linhas do arquivo. Obtendo o número M de linhas da matriz.
	do{
		line = (char **)realloc(line, (*m + 1) * sizeof(char *));
		length = (int *)realloc(length, (*m + 1) * sizeof(int));
		fscanf(fp, "%m[^\r\n]%n%*[\r\n]", line + *m, length + *m);
		(*m)++;
	}while (!feof(fp));

	// Fechando o arquivo com a matriz A.
	fclose(fp);

	// Obtendo o número N de colunas da matriz.
	for (offset = 0; offset < length[0]; offset += len + 1, (*n)++){
		sscanf(line[0] + offset, "%lf%n", &aux, &len);
	}

	// Aloca uma matriz (M + 1) x (N + 1) para que a Matriz Aumentada seja 1-based.
	A = malloc_matrix(*m + 1, *n + 1);

	// Obtendo os valores da matriz A.
	for (i = 0; i < *m; i++){
		for (j = 0, offset = 0; j < *n; j++, offset += len + 1){
			sscanf(line[i] + offset, "%lf%n", &A[i*(*n+1)+j], &len);
		}
	}

	// Liberando a memória alocada para ler o arquivo com a matriz A.
	for (i = 0; i < *m; i++){
		free(line[i]);
	}

	free(line);
	free(length);

	// Abrindo o arquivo com o vetor b.
	fp = fopen(VECTOR_PATH, "r");

	// Lendo os valores do vetor b direto para a última coluna (coluna n + 1) da Matriz Aumentada.
	for (i = 0; i < *m; i++){
		fscanf(fp, "%lf", &A[i*(*n+1)+(*n)]);
	}

	// Fechando o arquivo com o vetor b.
	fclose(fp);

	return A;
}

/* Imprime a solução do sistema linear Ax = b em um arquivo. */
void print_solution(double *A, int m, int n){
	FILE *fp;
	int i;

	fp = fopen(RESULT_PATH, "w");

	for (i = 0; i < m; i++){
		fprintf(fp, "%.3lf\n", A[i*(n+1) + n]);
	}

	fclose(fp);
}

int find_proc(int *start_row, int cur_row, int num_proc){
	int i = 0;
	for(i = 0; i < num_proc; i++){
		if(start_row[i+1] > cur_row){
			return i;
		}
	}
	return 0;
}

int main(int argc, char *argv[]){
	// variaveis de uso geral
	int m, n, p, i, j, k, l;
	double *A, *subA, s, t;

	// variaveis relacionadas ao MPI
	int my_rank, num_proc, root = 0, tag = 0;
	int num_elements, chunk_size, remainder, *sendcounts, *displs, *start_row, *num_rows, sum, sum_rows;
	pair pivot;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

	omp_set_nested(true);

	// Obtendo a Matriz Aumentada.
	if (my_rank == 0) {
		A = read_augmented_matrix(&m, &n);
		// print_matrix(A, m, n);

		// Recuperando o tempo inicial.
		s = omp_get_wtime();
	}
	
	// enviando os valores de m e n para todos os processos	
	MPI_Bcast(&m, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, root, MPI_COMM_WORLD);
	
	double *rcvbuf = (double*) malloc((n+1) * sizeof(double));
	
	// alocando a matriz para os demais processos
	if(my_rank != 0){
		A = malloc_matrix(m + 1, n + 1);
	}

	subA = malloc_matrix(m + 1, n + 1);

	/////////////////////////////////////////////////////////////////

	// Calculando novos valores do chunk para realizar a subtração das linhas
	chunk_size = m / num_proc;
	 
	// quantos chunks terao 1 elemento a mais
	remainder = m % num_proc;
	// posicao de inicio de cada chunk no buffer
	displs = (int*) calloc (num_proc+1, sizeof(int));
	// vetor com o tamanho de cada chunk
	sendcounts = (int*) calloc (num_proc, sizeof(int));
	// primeira linha de cada processo
	start_row = (int*) calloc(num_proc, sizeof(int));
	// numero de linhas de cada processo
	num_rows = (int*) calloc(num_proc, sizeof(int));

	// preenchendo os vetores sendcounts, displs, num_rows e sum_rows
	sum = 0, sum_rows = 0;
	for(k = 0; k < num_proc; k++){
		if(k < remainder){
			sendcounts[k] = (chunk_size + 1) * (n+1);
			num_rows[k] = chunk_size+1;
		} else {
			sendcounts[k] = chunk_size * (n+1);
			num_rows[k] = chunk_size;
		}

		displs[k] = sum;
		sum += sendcounts[k];
		start_row[k] = sum_rows;
		
		if(k < remainder){
			sum_rows += chunk_size+1;
		}
		else{
			sum_rows += chunk_size;
		}
	}
	displs[k] = sum;
	start_row[k] = sum_rows;

	//  Realizando Scatter para dividir as linhas da matriz entre os processos
	MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, subA, (chunk_size + 1) * (n+1), MPI_DOUBLE, root, MPI_COMM_WORLD);

	// [DEBUG] imprimindo as linhas de cada processo
	// for(i = 0; i < num_rows[my_rank]; i++){
	// 	// printf("num_rows[%d] = %d\n", my_rank, num_rows[my_rank]);
	// 	for(j = 0; j < n+1; j++){
	// 		printf("%.1lf ", subA[i*(n+1)+j]);
	// 	}
	// 	printf("\n");
	// }

	////////////////////////////////////////////////////////

	// double pivot = 0;
	// Eliminação de Gauss-Jordan.
	for (i = 0, j = 0; i < m && j < n+1; i++, j++){ 
		// processo responsavel pela linha atual
		int cur_rank = find_proc(start_row, i, num_proc);
		// printf("curr_rank = %d\n", cur_rank);

		// [DEBUG] imprimindo as linhas de cada processo
		// printf("rank = %d\n", my_rank);
		// for(k = 0; k < num_rows[my_rank]; k++){
		// 	// printf("num_rows[%d] = %d\n", my_rank, num_rows[my_rank]);
		// 	printf("(%d) ", start_row[my_rank]+k);
		// 	for(int l = 0; l < n+1; l++){
		// 		printf("%.1lf ", subA[k*(n+1)+l]);
		// 	}
		// 	printf("\n");
		// }

		// int x; scanf("%d", &x);

		int cur_val, cur_row, pivot_row;
		if(my_rank == cur_rank){
			// linha dentro do da sub-matriz
			cur_row = i - start_row[my_rank];
			// valor da posicao [i,j] da matriz original
			cur_val = subA[cur_row*(n+1)+j];

			// printf("cur_row = %d, cur_val = %d\n", cur_row, cur_val);
		}


		// o processo responsavel pela linha atual manda o valor do pivot atual para os demais
		MPI_Bcast(&cur_val, 1, MPI_DOUBLE, cur_rank, MPI_COMM_WORLD);
		pivot.val = cur_val;

		// se o valor de A[i][j] for 0, precisamos de um novo pivot
		if(cur_val == 0){
			// numero de linhas no processo atual
			pair local_pivot;
			local_pivot.val = 0, local_pivot.row = -1;
			pivot.val = 0, pivot.row = -1;
			// encontrando um pivo nao nulo
			for(k = 0; k < num_rows[my_rank]; k++){
				// printf("%lf, %d\n", subA[k*(n+1)+j], k);
				if(fabs(subA[k*(n+1)+j]) > local_pivot.val){
					local_pivot.val = subA[k*(n+1)+j];
					local_pivot.row = start_row[my_rank] + k;
				}
			}

			MPI_Allreduce(&local_pivot, &pivot, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

			// se só há zeros na coluna, aumentamos o j e vamos para a proxima iteracao
			if(pivot.val == 0){
				// printf("caiu aqui\n");
				i--;
				MPI_Barrier(MPI_COMM_WORLD);
				continue;
			}

			// printf("pivot = (%lf,%d)\n", pivot.val, pivot.row);

			// procesos que contem a linha do pivot
			int pivot_rank = find_proc(start_row, pivot.row, num_proc);
			// printf("pivot_rank = %d\n", pivot_rank);
			

			if(my_rank == cur_rank && cur_rank == pivot_rank){
				// linha do pivot na sub-matriz
				pivot_row = pivot.row - start_row[pivot_rank];
				cur_row = i - start_row[my_rank];
				// trocando as linhas:
				memcpy(rcvbuf, &subA[cur_row*(n+1)], (n+1) * sizeof(double));
				memcpy(&subA[cur_row*(n+1)], &subA[pivot_row*(n+1)], (n+1) * sizeof(double));
				memcpy(&subA[pivot_row*(n+1)], rcvbuf, (n+1) * sizeof(double));
			}
			else{
				// no processo que contem a linha atual
				if(my_rank == cur_rank){
					// mando a linha atual para o processo que contem o pivot
					MPI_Send(&subA[cur_row*(n+1)], n+1, MPI_DOUBLE, pivot_rank, tag, MPI_COMM_WORLD);
					// recebo a linha do pivot no rcvbuf
					MPI_Recv(rcvbuf, n+1, MPI_DOUBLE, pivot_rank, tag, MPI_COMM_WORLD, &status);
					// copio a linha do pivot para a linha atual
					memcpy(&subA[cur_row*(n+1)], rcvbuf, (n+1) * sizeof(double));
				}

				// no processo que contem a linha do pivot
				if(my_rank == pivot_rank){
					// descobrindo a linha da sub-matriz em que o pivot esta
					pivot_row = pivot.row - start_row[pivot_rank];
					// printf("pivot_row = %d\n", pivot_row);
					// mando a linha do pivot para o processo que contem a antiga linha atual
					MPI_Send(&subA[pivot_row*(n+1)], n+1, MPI_DOUBLE, cur_rank, tag, MPI_COMM_WORLD);
					// recebo a linha atual do processo que a contem
					MPI_Recv(rcvbuf, n+1, MPI_DOUBLE, cur_rank, tag, MPI_COMM_WORLD, &status);
					// copio a antiga linha atual para a linha do pivot
					memcpy(&subA[pivot_row*(n+1)], rcvbuf, (n+1) * sizeof(double));
				}
			}
		}
		
		// if(j > n){
		// 	MPI_Barrier(MPI_COMM_WORLD);
		// 	break;
		// }

		// normalizacao: dividindo a linha do pivot pelo valor do pivot
		if(my_rank == cur_rank){
			// cur_row = i - start_row[my_rank];
			// printf("pivot.val = %.1lf\n", pivot.val);
			for(k = j+1; k < n+1; k++){
				subA[cur_row*(n+1)+k] /= pivot.val;
			}
			subA[cur_row*(n+1)+j] = 1.0;
		}



		// Enviando a linha do pivot para todos processos.
		if(my_rank == cur_rank){
			MPI_Bcast(&subA[cur_row*(n+1)], n+1, MPI_DOUBLE, cur_rank, MPI_COMM_WORLD);
			memcpy(rcvbuf, &subA[cur_row*(n+1)], (n+1) * sizeof(double));
		} else {
			MPI_Bcast(rcvbuf, n+1, MPI_DOUBLE, cur_rank, MPI_COMM_WORLD);
			// printf("linha do pivot = ");
			// for(k = 0; k < n+1; k++){
			// 	printf("%.1lf ", rcvbuf[k]);
			// }
			// printf("\n");
		}
	
		// if(my_rank == cur_rank){	
		// 	for(k = 0; k < num_proc; k++){
		// 		MPI_Send(&subA[cur_row*(n+1)], n+1, MPI_DOUBLE, )
		// 	}
		// }

		for(k = 0; k < num_rows[my_rank]; k++){
			int real_row = start_row[my_rank]+k;
			if(real_row != i && subA[k*(n+1)+j] != 0.0){
				// #pragma omp parallel for default(shared) private(l) num_threads(NUM_THREADS)
				for (l = j + 1; l < n + 1; l++){
					subA[k*(n+1)+l] -= subA[k*(n+1)+j] * rcvbuf[l];	
				}
				// Atualizando o valor de A[k][j]. O valor de A[k][j] é 0 após a subtração.
				subA[k*(n+1)+j] = 0.0;
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}
	
	// Usamos gatherv para reunir todas as sub-matrizes na matriz A do processo 0.
	MPI_Gatherv(subA, sendcounts[my_rank], MPI_DOUBLE, A, sendcounts, displs, MPI_DOUBLE, root, MPI_COMM_WORLD);

	if (my_rank == 0){
		// Imprimindo a Matriz Aumentada final.
		// printf("\nMatriz Aumentada final:\n");
		print_matrix(A, m, n);
		// Recuperando o tempo final.
		t = omp_get_wtime();
		printf("Tempo: %.5lfs\n", t - s);
		// Imprimindo a solução em um arquivo.
		print_solution(A, m, n);
		// Liberando a memória alocada para a Matriz Aumentada.
		free(A);
	}

	MPI_Finalize();

	return 0;
}