/* Victor Forbes - 9293394
   Gabriel Camargo - 9293456
   Marcos Camargo - 9278045
   Felipe Alegria - 9293501 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

#define true 1
#define false 0

/* Imprime a matriz aumentada. */
void print_matrix(double **A, int m, int n){
	int i, j;

	for (i = 1; i <= m; i++){
		for (j = 1; j <= n + 1; j++){
			printf("%.0lf ", A[i][j]);
		}

		printf("\n");
	}
}

/* Aloca uma matriz de doubles MxN. */
double **malloc_matrix(int m, int n){
	double **A;
	int i;

	A = (double **)malloc(m * sizeof(double *));

	for (i = 0; i < m; i++){
		A[i] = (double *)malloc(n * sizeof(double));
	}

	return A;
}

/* Libera a memória alocada para uma matriz de doubles MxN. */
void free_matrix(double **A, int m, int n){
	int i;

	for (i = 0; i < m; i++){
		free(A[i]);
	}

	free(A);
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
double **read_augmented_matrix(int *m, int *n){
	int *length, offset, len, i, j;
	double **A, aux;
	char **line;
	FILE *fp;

	// Inicializando.
	line = NULL;
	length = NULL;
	*m = *n = 0;

	// Abrindo o arquivo com a matriz A.
	fp = fopen("input/matriz5.txt", "r");

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

	// Aloca uma matriz (M + 1) x (N + 2) para que a Matriz Aumentada seja 1-based.
	A = malloc_matrix(*m + 1, *n + 2);

	// Obtendo os valores da matriz A.
	for (i = 1; i <= *m; i++){
		for (j = 1, offset = 0; j <= *n; j++, offset += len + 1){
			sscanf(line[i - 1] + offset, "%lf%n", A[i] + j, &len);
		}
	}

	// Liberando a memória alocada para ler o arquivo com a matriz A.
	for (i = 0; i < *m; i++){
		free(line[i]);
	}

	free(line);
	free(length);

	// Abrindo o arquivo com o vetor b.
	fp = fopen("input/vetor5.txt", "r");

	// Lendo os valores do vetor b direto para a última coluna (coluna n + 1) da Matriz Aumentada.
	for (i = 1; i <= *m; i++){
		fscanf(fp, "%lf", A[i] + *n + 1);
	}

	// Fechando o arquivo com o vetor b.
	fclose(fp);

	return A;
}

/* Imprime a solução do sistema linear Ax = b em um arquivo. */
void print_solution(double **A, int m, int n){
	FILE *fp;
	int i;

	fp = fopen("resultado5.txt", "w");

	for (i = 1; i <= m; i++){
		fprintf(fp, "%.3lf\n", A[i][n + 1]);
	}

	fclose(fp);
}

int main(int argc, char *argv[]){
	// variaveis de uso geral
	int m, n, p, i, j, k, l;
	double **A, s, t;

	// variaveis relacionadas ao MPI
	int my_rank, num_proc, root = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

	omp_set_nested(true);

	// Obtendo a Matriz Aumentada.
	if (my_rank == 0) {
		A = read_augmented_matrix(&m, &n);

		// Recuperando o tempo inicial.
		s = omp_get_wtime();
	}
		
	MPI_Bcast(&m, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, root, MPI_COMM_WORLD);
	if(my_rank != 0){
		// Aparentemente precisa estar alocado para funcionar (não sabemos o porque D:)
		A = malloc_matrix(m + 1, n + 2);
	}

	// Imprimindo a Matriz Aumentada Inicial.
	// printf("Matriz Aumentada inicial:\n");
	// print_matrix(A, m, n);

	double pivot = 0;
	// Eliminação de Gauss-Jordan.
	for (i = 1, j = 1; i <= m; i++, j++){
		if (my_rank == 0 ){
			// Buscando o próximo pivô A[i][j].
			while (j <= n + 1 && A[i][j] == 0.0){
				// Buscando uma linha abaixo de A[i][j] tal que A[k][j] != 0.
				p = 0;
				#pragma omp parallel for default(shared) private(k) reduction(max: p) num_threads(2)
				for (k = i + 1; k <= m; k++){
					if (A[k][j] == 0.0){
						p = 0;
					}
					else{
						p = k;
					}
				}

				// Checando se um pivô foi encontrado.
				if (p > 0){ // Se uma linha foi encontrada.
					// printf("VAI SWAPAR %d com %d na coluna %d\n", i, p, j);
					// int buceta; scanf("%d", &buceta);
					swap(A + i, A + p, sizeof(double *));
				}
				else{ // Se uma linha não foi encontrada.
					j++;
				}
			}

			// Não há mais pivôs.
			if (j > n + 1){
				break;
			}

			pivot = A[i][j];

			// Atualizando o valor do pivô. O valor do pivô é 1 após a divisão.
			A[i][j] = 1.0;

		}

		// Enviando o pivot para todos os processos
		MPI_Bcast(&pivot, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

		// Realizando scatter para dividir a linha entre os processos e realizar a divisão pelo pivot

		int num_elements, chunk_size, remainder, *sendcounts, *displs, sum;
		// numero de elementos serem divididos pelo pivot
		num_elements = n-j+1;
		// tamanho de cada chunk
		chunk_size = num_elements/num_proc;
		// numero de elementos que sobram ao dividir o numero de elementos pelo numero de chunks
		remainder = num_elements % num_proc;			
		// vetor com o tamanho dos chunks
		sendcounts = (int*) calloc (num_proc, sizeof(int));
		// vetor com as posicoes onde cada chuk comeca
		displs = (int*) calloc (num_proc+1, sizeof(int));
		// Vetor que receberá os valores
		double *rcvbuf = (double*) malloc((chunk_size+1) * sizeof(double));

		// os 'remainder' primeiros chunks terao um elementos a mais que os demais
		sum = 0;
		for(k = 0; k < remainder; k++){
			sendcounts[k] = chunk_size+1;
			displs[k] = sum;
			sum += sendcounts[k];
		}
		// os demais chunks terao 'chunk_size' elementos
		for(k = remainder; k < num_proc; k++){
			sendcounts[k] = chunk_size;
			displs[k] = sum;
			sum += sendcounts[k];
		}

		// Enviando um chunk para cada processo
		if(my_rank < num_elements){
			MPI_Scatterv(&A[i][j+1], sendcounts, displs, MPI_DOUBLE, rcvbuf, chunk_size+1, MPI_DOUBLE, root, MPI_COMM_WORLD);
					
			// Dividindo a i-ésima linha pelo pivô A[i][j].
			#pragma omp parallel for default(shared) private(k) num_threads(2)
			for (k = 0; k < sendcounts[my_rank]; k++){
				rcvbuf[k] /= pivot;
			}

			// Usamos o Gatherv para unir a linha que dividimos.
			MPI_Gatherv(rcvbuf, sendcounts[my_rank], MPI_DOUBLE, &A[i][j+1], sendcounts, displs, MPI_DOUBLE, root, MPI_COMM_WORLD);

		}

		// Liberando memória alocada.
		free(rcvbuf);
		// free(displs);

		// Enviando a linha do pivot para todos processos.
		MPI_Bcast(A[i], n+2, MPI_DOUBLE, root, MPI_COMM_WORLD);
		
		// Enviando a linha a ser subtraida para os processos do mesmo grupo de comunicação.
		int tag = 0;
		MPI_Status status;
		
		chunk_size = m/num_proc;
		remainder = m%num_proc;
		sum = 1;
		for(k = 0; k < num_proc; k++){
			if(k < remainder){
				sendcounts[k] = chunk_size + 1;
			} else {
				sendcounts[k] = chunk_size;
			}
			displs[k] = sum;
			sum += sendcounts[k];
			// printf("sendcounts[%d] = %d\n", k, sendcounts[k]);
			// printf("displs[%d] = %d\n", k, displs[k]);
		}
		displs[k] = sum;
		// printf("displs[%d] = %d\n", k, displs[k]);

		if(my_rank == 0){ // enviar as linhas para os respectivos processos
			for(k = 1; k < num_proc; k++){ // para cada processo exeto o root
				for(l = displs[k]; l < displs[k+1]; l++){ // envio sendcounts[k] linhas (ou menos, se tiver resto)
					MPI_Send(A[l], n+2, MPI_DOUBLE, k, tag, MPI_COMM_WORLD);
				} 
			}
		}
		else{ // recebendo as linhas a serem subtraidas da linha do pivo
			for(l = displs[my_rank]; l < displs[my_rank+1]; l++){ // recebo sendcounts[my_rank] linhas (ou menos, se tiver resto)
				MPI_Recv(A[l], n+2, MPI_DOUBLE, root, tag, MPI_COMM_WORLD, &status);
			}
		}


		// Para todos processos, subtraimos as linhas de seus chunks
		for(k = displs[my_rank]; k < displs[my_rank+1]; k++){
			if(k != i && A[k][j] != 0){
				#pragma omp parallel for default(shared) private(l) num_threads(2)
				for (l = j + 1; l <= n + 1; l++){
					A[k][l] -= A[k][j] * A[i][l];
				}
				A[k][j] = 0.0;
			}
		}


		if(my_rank != 0){ // enviando as linhas modificadas de volta ao processo mestre
			// printf("displs[%d] = %d, displs[%d] = %d\n", my_rank, displs[my_rank], my_rank+1, displs[my_rank+1]);
			for(k = displs[my_rank]; k < displs[my_rank+1]; k++){ // recebo sendcounts[k] linhas (ou menos, se tiver resto)
				// printf("mandou a linha %d na iteracao %d\n", k, i);
				MPI_Send(A[k], n+2, MPI_DOUBLE, root, tag, MPI_COMM_WORLD);
			}
		}
		else{ // recebendo as linhas modificadas
			for(k = 1; k < num_proc; k++){ // para cada processo
				for(l = displs[k]; l < displs[k+1]; l++){ // envio chunk_size linhas (ou menos, se tiver resto)
					// printf("vai receber a linha %d na iteracao %d... k = %d\n", l, i, k);
					MPI_Recv(A[l], n+2, MPI_DOUBLE, k, tag, MPI_COMM_WORLD, &status);
					// printf("recebeu a linha %d na iteracao %d\n", l, i);
				}
			}
		}
		
		// if(my_rank == 0) 		printf("passou o scatter\n");

		free(displs);
		free(sendcounts);


		// if(my_rank == 0){
		// 	// Eliminando todas as outras entradas da j-ésima coluna.
		// 	for (k = 1; k <= m; k++){
		// 		// Se não for a i-ésima linha (linha do pivô) e possui entrada A[k][j] != 0.
		// 		if (k != i && A[k][j] != 0.0){
		// 			// Subtraindo a linha i da linha k.
		// 			// #pragma omp parallel for default(shared) private(l) num_threads(2)
		// 			for (l = j + 1; l <= n + 1; l++){
		// 				/* 	j = coluna do pivo
		// 					i = linha do pivo	

		// 					k = linha atual
		// 					l = percorre a linha
		// 				*/
		// 				A[k][l] -= A[k][j] * A[i][l];
		// 			}

		// 			// Atualizando o valor de A[k][j]. O valor de A[k][j] é 0 após a subtração.
		// 			A[k][j] = 0.0;
		// 		}
		// 	}
		// }
	}

	if (my_rank == 0){
		// Imprimindo a Matriz Aumentada final.
		printf("\nMatriz Aumentada final:\n");
		print_matrix(A, m, n);
		// Recuperando o tempo final.
		t = omp_get_wtime();
		printf("Tempo: %.5lfs\n", t - s);
		// Imprimindo a solução em um arquivo.
		print_solution(A, m, n);
		// Liberando a memória alocada para a Matriz Aumentada.
		free_matrix(A, m + 1, n + 2);
	}

	MPI_Finalize();

	return 0;
}