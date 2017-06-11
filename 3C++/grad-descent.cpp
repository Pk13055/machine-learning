// This is a simple implementation of gradient descent with takes params as follows:
// It takes the size of the dataset and the inputs as the size of the houses in sq feet.
// It then takes the number of queries and gives predicted prices on the basis of the learnt values
// This algorithm uses the square error hypothesis to calculate the predicted prices for the house
#include <iostream>
#include <vector>
#include <limits.h>
#include <math.h>
#define PRE 0.0001
using namespace std;

class gradientDescent {
	int m;
	long long int learning_rate;
	double t0, t1;
	void calculate_params();
	double h(float x, double t_0, double t_1) { return t_0 + t_1 * x; }
public:
	gradientDescent(int m_p, int l_r = 0.01) { 
		m = m_p; 
		learning_rate = l_r; 
		t0 = t1 = 0;
	} 
	void getHouses();
	double queryHouse(float size);
protected:
	struct house { 
		float size; 
		float cost; 
		house(float s, float c) { size = s; cost = c; }
	};
	vector< house > house_list;
};

void gradientDescent::getHouses() {
	for(int i = 0; i < m; i++) {
		int s, c;
		cout<<"Enter the size, cost of house "<<i + 1<<" : ";
		cin>>s>>c;
		house_list.push_back(house(s, c));
	}
	cout<<endl<<"Houses successfully entered"<<endl;
	for(auto i = house_list.begin(); i != house_list.end(); i++)
		cout<<(*i).size<<" : "<<(*i).cost<<endl;
	calculate_params();
}

// Gradient descent here is used with linear regression to provide for a supervised machine learning algorithm
void gradientDescent::calculate_params() {
	do {
		double temp_sum[2] = {0, 0};
		for(int i = 0; i < m; i++) {
			temp_sum[0] += ( h(house_list[i].size, t0, t1) - house_list[i].cost);
			temp_sum[1] += ((h(house_list[i].size, t0, t1) - house_list[i].cost) * house_list[i].size);
		}
		cout<<temp_sum[0]<<" : "<<temp_sum[1]<<endl;
		temp_sum[0] /= m; temp_sum[0] *= learning_rate;
		temp_sum[1] /= m; temp_sum[1] *= learning_rate;
		cout<<temp_sum[0]<<" : "<<temp_sum[1]<<endl;
		double temp0 = t0 - temp_sum[0];
		double temp1 = t1 - temp_sum[1];
		if(fabs(temp0 - t0) < PRE && fabs(temp1 - t1) < PRE) break;
		else { t0 = temp0; t1 = temp1; }
	} while(true);
	t0 = t0; t1 = t1;
}
/*
void gradientDescent::calculate_params() {
	double t0 = 0, t1 = 0;
	while(1) {
		double temp_sum[2] = {0, 0}, temp0, temp1;
		for(int i = 0; i < m; i++) {
			temp_sum[0] += (h(house_list[i].size, t0, t1) - house_list[i].cost);
			temp_sum[1] += ((h(house_list[i].size, t0, t1) - house_list[i].cost) * house_list[i].size);
		}

	}
}
*/
double gradientDescent::queryHouse(float house) { return h(house, t0, t1); }

signed main() {
	int m;
	cout<<"Enter dataset size : "; cin>>m;
	gradientDescent g(m);
	g.getHouses();
	cout<<"Enter the number of queries : "; cin>>m;
	while(m--) {
		float temp;
		cout<<"Enter the size of the house : ";
		cin>>temp;
		cout<<"Predicted cost of the house : "<<g.queryHouse(temp)<<endl;
	}
}