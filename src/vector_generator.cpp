#include <bits/stdc++.h>

using namespace std;

int main(){
	int k, i;

	srand(time(NULL));
	
	scanf("%d", &k);
	
	for(i = 0; i < (1 << k); i++){
		printf("%d\n", rand() % 1000);
	}
	
	return 0;
}