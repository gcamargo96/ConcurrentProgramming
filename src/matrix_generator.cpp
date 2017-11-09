#include <bits/stdc++.h>

using namespace std;

#define N (1 << 13)

int mat[N][N];

void preenche(int x){
	int i, j;

	for (i = 0; i < x; i++){
		for (j = 0; j < x; j++){
			mat[i][j + x] = mat[i][j];
			mat[i + x][j] = mat[i][j];
			mat[i + x][j + x] = -mat[i][j];
		}
	}
}

int main(){
	int k, i, j;

	cin >> k;

	mat[0][0] = 1;

	for (i = 0; i < k; i++){
		preenche(1 << i);
	}

	for (i = 0; i < (1 << k); i++){
		for (j = 0; j < (1 << k); j++){
			if (mat[i][j] == 1){
				cout << "1 ";
			}
			else{
				cout << "-1 ";
			}
		}

		cout << endl;
	}

	return 0;
}
