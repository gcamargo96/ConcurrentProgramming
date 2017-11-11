/* Victor Forbes - 9293394
   Gabriel Camargo - 9293456
   Marcos Camargo - 9278045
   Felipe Alegria - 9293501 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

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
	fp = fopen("../input/matriz2.txt", "r");

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
	fp = fopen("../input/vetor2.txt", "r");

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

	fp = fopen("resultado2.txt", "w");

	for (i = 1; i <= m; i++){
		fprintf(fp, "%.3lf\n", A[i][n + 1]);
	}

	fclose(fp);
}

int main(int argc, char *argv[]){
	int m, n, i, j, k, l;
	double **A, s, t;

	// Obtendo a Matriz Aumentada.
	A = read_augmented_matrix(&m, &n);

	// Imprimindo a Matriz Aumentada Inicial.
	// printf("Matriz Aumentada inicial:\n");
	// print_matrix(A, m, n);

	// Recuperando o tempo inicial.
	s = omp_get_wtime();

	// Eliminação de Gauss-Jordan.
	for (i = 1, j = 1; i <= m; i++, j++){
		// Buscando o próximo pivô A[i][j].
		while (j <= n + 1 && A[i][j] == 0.0){
			// Buscando uma linha abaixo de A[i][j] tal que A[k][j] != 0.]
			for (k = i + 1; k <= m; k++){
				if (A[k][j] != 0.0){
					break;
				}
			}

			// Checando se um pivô foi encontrado.
			if (k <= m){ // Se uma linha foi encontrada.
				// printf("VAI SWAPAR %d com %d na coluna %d\n", i, k, j);
					// int buceta; scanf("%d", &buceta);
				swap(A + i, A + k, sizeof(double *));
			}
			else{ // Se uma linha não foi encontrada.
				j++;
			}
		}

		// Não há mais pivôs.
		if (j > n + 1){
			break;
		}

		// Dividindo a i-ésima linha pelo pivô A[i][j].
		for (k = j + 1; k <= n + 1; k++){
			A[i][k] /= A[i][j];
		}

		// Atualizando o valor do pivô. O valor do pivô é 1 após a divisão.
		A[i][j] = 1.0;

		// Eliminando todas as outras entradas da j-ésima coluna.
		for (k = 1; k <= m; k++){
			// Se não for a i-ésima linha (linha do pivô) e possui entrada A[k][j] != 0.
			if (k != i && A[k][j] != 0.0){
				// Subtraindo a linha i da linha k;
				for (l = j + 1; l <= n + 1; l++){
					A[k][l] -= A[k][j] * A[i][l];
				}

				// Atualizando o valor de A[k][j]. O valor de A[k][j] é 0 após a subtração.
				A[k][j] = 0.0;
			}
		}
	}

	// Recuperando o tempo inicial.
	t = omp_get_wtime();

	// Imprimindo a Matriz Aumentada final.
	printf("\nMatriz Aumentada final:\n");
	print_matrix(A, m, n);
	printf("Tempo: %.5lfs\n", t - s);

	// Imprimindo a solução em um arquivo.
	print_solution(A, m, n);

	// Liberando a memória alocada para a Matriz Aumentada.
	free_matrix(A, m + 1, n + 2);

	return 0;
}