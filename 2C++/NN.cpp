#include <bits/stdc++.h>
#include <vector>
using namespace std;

/*	
Auto loop example
vector<int> x;
for(auto i : x) {
	cout<<x.back()<< " " ;
	x.pop_back();
}*/
// m x n matrix
struct Matrix {
	int n;
	int m;
	Matrix(int m, int n, int init = 0) {
		Matrix::n  = n; Matrix::m = m;
		for(int i = 0; i < m; i ++) {
			vector<float> row;
			for(int j = 0; j < n; row.push_back(init), j++);
			skel.push_back(row);
		}
	}
	friend ostream& operator<<(ostream& os, const Matrix& M);
private:
	vector< vector<float> > skel;
};

ostream& operator<<(ostream& os, const Matrix& M) {
	for(int i = 0; i < M.m; i++, cout<<endl) 
		for(j : M.skel[i]) 
			cout<<j<<" ";
	cout<<M.skel[3][2];
}

signed main(int argc, char *argv[]) {
	if(argc < 4) 
		cout<<"You have to pass the filename AND learning rate"<<endl;
	else {
		int m = atoi(argv[2]);
		int n = atoi(argv[3]);
		Matrix dataset(m, n);
		cout<<dataset;
	}
}